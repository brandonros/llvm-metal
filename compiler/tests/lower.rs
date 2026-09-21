//! Hand-written kernels through the whole compiler and onto the GPU, one
//! feature of `lower` each.
#![cfg(target_os = "macos")]
use inkwell::{context::Context, memory_buffer::MemoryBuffer};
use llvm_metal_compiler::compile_bitcode;
use std::path::PathBuf;

const OPERATIONS: &str = "
    declare i32 @llvm_metal.thread_index()
    declare i64 @llvm_metal.length(i32)
    declare i8 @llvm_metal.load(i32, i64)
    declare void @llvm_metal.store(i32, i64, i8)
    declare void @abort() noreturn
";

/// Run `kernel.k` from `body` on `threads` threads over an input and an output
/// buffer, and return the output, or the runtime's error.
fn run(
    test: &str,
    body: &str,
    threads: usize,
    input: &[u8],
    output: usize,
) -> Result<Vec<u8>, String> {
    let context = Context::create();
    let source = format!("{OPERATIONS}{body}");
    let buffer = MemoryBuffer::create_from_memory_range_copy(source.as_bytes(), test);
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|error| error.to_string())?;
    let bitcode = module.write_bitcode_to_memory().as_slice().to_vec();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/lower")
        .join(test);
    let compiled =
        compile_bitcode(&[bitcode], "k", &directory).map_err(|error| format!("{error:?}"))?;
    let pipeline = llvm_metal_runtime::Pipeline::load(&compiled.library, "k")?;
    let mut input = pipeline.buffer_from(input)?;
    let mut output = pipeline.buffer::<u8>(output)?;
    let slots = compiled.bindings.buffers;
    pipeline.run(threads, slots, &mut [&mut input, &mut output])?;
    Ok(output.read(<[u8]>::to_vec))
}

#[test]
#[ignore = "requires an Apple GPU"]
fn operations_reach_the_buffers_the_thread_and_the_lengths() {
    // output[thread] = input[thread] + length(input)
    let body = "define void @kernel.k() {
        %thread = call i32 @llvm_metal.thread_index()
        %index = zext i32 %thread to i64
        %byte = call i8 @llvm_metal.load(i32 0, i64 %index)
        %length = call i64 @llvm_metal.length(i32 0)
        %short = trunc i64 %length to i8
        %sum = add i8 %byte, %short
        call void @llvm_metal.store(i32 1, i64 %index, i8 %sum)
        ret void }";
    assert_eq!(
        run("operations", body, 4, &[10, 20, 30, 40], 4),
        Ok(vec![14, 24, 34, 44])
    );
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_helper_receives_the_launch() {
    let body = "define void @kernel.k() { call void @copy(i64 2)  ret void }
        define internal void @copy(i64 %index) {
        %byte = call i8 @llvm_metal.load(i32 0, i64 %index)
        call void @llvm_metal.store(i32 1, i64 0, i8 %byte)
        ret void }";
    assert_eq!(run("helper", body, 1, &[1, 2, 3], 1), Ok(vec![3]));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_constant_is_read_from_the_threads_copy() {
    let body = "@table = constant [4 x i8] c\"\\05\\06\\07\\08\"
        define void @kernel.k() {
        %index = call i8 @llvm_metal.load(i32 0, i64 0)
        %wide = zext i8 %index to i64
        %place = getelementptr [4 x i8], ptr @table, i64 0, i64 %wide
        %byte = load i8, ptr %place
        call void @llvm_metal.store(i32 1, i64 0, i8 %byte)
        ret void }";
    assert_eq!(run("constant", body, 1, &[2], 1), Ok(vec![7]));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn an_address_held_by_a_constant_is_relocated() {
    let body = "@text = constant [3 x i8] c\"abc\"
        @slice = constant { ptr, i64 } { ptr getelementptr (i8, ptr @text, i64 1), i64 2 }
        define void @kernel.k() {
        %start = load ptr, ptr @slice
        %byte = load i8, ptr %start
        call void @llvm_metal.store(i32 1, i64 0, i8 %byte)
        ret void }";
    assert_eq!(run("relocation", body, 1, &[0], 1), Ok(vec![b'b']));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn a_panic_fails_the_launch() {
    let body = "define void @kernel.k() {
        %byte = call i8 @llvm_metal.load(i32 0, i64 0)
        %bad = icmp eq i8 %byte, 1
        br i1 %bad, label %fail, label %fine
        fail:  call void @abort()  unreachable
        fine:  ret void }";
    assert_eq!(run("panic-no", body, 1, &[0], 1), Ok(vec![0]));
    assert_eq!(
        run("panic-yes", body, 1, &[1], 1),
        Err("a thread panicked".into())
    );
}
