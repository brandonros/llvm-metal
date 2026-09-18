use inkwell::{
    context::Context,
    values::{AnyValue, InstructionOpcode},
};
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};

fn interface() -> KernelInterface {
    KernelInterface {
        schema: 1,
        entry: "kernel".into(),
        calling_convention: "C".into(),
        invocations: Some(1),
        dispatch: Dispatch::Single,
        aliasing: "disjoint".into(),
        arguments: vec![BufferArgument {
            name: "data".into(),
            kind: "buffer".into(),
            access: Access::ReadWrite,
            bytes: 16,
            alignment: 8,
        }],
    }
}
// A backward constant-table walk retains a constant-expression incoming pointer
// after optimization. The legacy writer used to emit its GEP before the PHI in
// the loop header, violating both PHI grouping and the incoming-edge definition.
const SOURCE: &str = r#"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
@table = private constant [4 x i64] [i64 1, i64 5, i64 7, i64 9], align 8
define void @kernel(ptr %p) {
entry:
 %raw = load i32, ptr %p, align 4
 %bounded = and i32 %raw, 3
 %count = add i32 %bounded, 1
 br label %loop
loop:
 %cursor = phi ptr [getelementptr inbounds (i8, ptr @table, i64 32), %entry], [%previous, %loop]
 %index = phi i32 [0, %entry], [%next, %loop]
 %sum = phi i64 [0, %entry], [%accumulated, %loop]
 %previous = getelementptr inbounds i8, ptr %cursor, i64 -8
 %value = load i64, ptr %previous, align 8
 %accumulated = add i64 %sum, %value
 %next = add i32 %index, 1
 %done = icmp eq i32 %next, %count
 br i1 %done, label %exit, label %loop
exit:
 %out = getelementptr i8, ptr %p, i64 8
 store i64 %accumulated, ptr %out, align 8
 ret void
}
"#;

#[test]
fn constant_expression_phi_operands_become_incoming_edge_instructions() {
    let context = Context::create();
    let module = parse_ir(&context, SOURCE.as_bytes(), "phi-constant").unwrap();
    let original = module.print_to_string().to_string();
    let (air, _) = legalize(&module, &interface()).unwrap();
    air.verify().unwrap();
    let mut pointer_phis = 0;
    for function in air.get_functions() {
        for block in function.get_basic_blocks() {
            for instruction in block
                .get_instructions()
                .filter(|i| i.get_opcode() == InstructionOpcode::Phi)
            {
                let text = instruction.print_to_string().to_string();
                if text.contains("phi ptr addrspace(2)") {
                    pointer_phis += 1;
                    assert!(!text.contains("getelementptr"), "{text}");
                }
            }
        }
    }
    assert!(
        pointer_phis > 0,
        "test must retain the constant-space loop PHI"
    );
    assert_eq!(module.print_to_string().to_string(), original);
}

#[test]
#[ignore = "requires pinned llvm-downgrade from the Nix shell; no GPU needed"]
fn legacy_writer_roundtrip_preserves_phi_grouping_and_dominance() {
    let context = Context::create();
    let module = parse_ir(&context, SOURCE.as_bytes(), "phi-constant").unwrap();
    let original = module.print_to_string().to_string();
    let compiled = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
    let reparsed =
        llvm_metal_compiler::parse_bitcode(&context, &compiled.air_bitcode, "legacy-phi").unwrap();
    reparsed.verify().unwrap();
    assert_eq!(module.print_to_string().to_string(), original);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn constant_pointer_loop_phi_reads_expected_table_values_on_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    let context = Context::create();
    let module = parse_ir(&context, SOURCE.as_bytes(), "phi-constant").unwrap();
    let compiled = llvm_metal_compiler::compile::compile(&module, &interface()).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("kernel.metallib");
    std::fs::write(&path, compiled.metallib).unwrap();
    let kernel = Kernel::load(&path, &compiled.bindings).unwrap();
    for raw in [0u32, 1, 2, 3, 4, u32::MAX] {
        let mut bytes = vec![0xa5; 528];
        bytes[256..260].copy_from_slice(&raw.to_le_bytes());
        let mut expected = bytes.clone();
        let sum: u64 = [9u64, 7, 5, 1].iter().take((raw as usize & 3) + 1).sum();
        expected[264..272].copy_from_slice(&sum.to_le_bytes());
        let mut buffers = [Buffer { bytes, offset: 256 }];
        // SAFETY: input bounds the walk to the four constant elements; exactly
        // one aligned output word is written within the guarded 16-byte record.
        unsafe {
            kernel.run(&mut buffers, 1, 1).unwrap();
        }
        assert_eq!(buffers[0].bytes, expected);
    }
}
