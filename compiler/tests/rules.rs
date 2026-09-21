//! One kernel per rule that breaks it and nothing else.
use inkwell::{context::Context, memory_buffer::MemoryBuffer};
use llvm_metal_compiler::{Error, Rule, program};

/// The rules `body` breaks, as a module whose entry is `kernel.k`.
fn broken(body: &str) -> Vec<Rule> {
    let context = Context::create();
    let buffer = MemoryBuffer::create_from_memory_range_copy(body.as_bytes(), "rule");
    let module = context.create_module_from_ir(buffer).unwrap();
    let bitcode = module.write_bitcode_to_memory().as_slice().to_vec();
    match program(&context, &[bitcode], "k") {
        Ok(_) => Vec::new(),
        Err(Error::Refused(violations)) => violations.into_iter().map(|v| v.rule).collect(),
        Err(Error::Input(error)) => panic!("{error}"),
    }
}

#[test]
fn a_kernel_that_breaks_no_rule_is_accepted() {
    let rules = broken(
        "declare i8 @llvm_metal.load(i32, i64)
         define void @kernel.k() { %b = call i8 @llvm_metal.load(i32 0, i64 0)  ret void }",
    );
    assert_eq!(rules, []);
}

#[test]
fn known_function() {
    let rules = broken(
        "declare void @elsewhere()  define void @kernel.k() { call void @elsewhere()  ret void }",
    );
    assert_eq!(rules, [Rule::KnownFunction]);
}

#[test]
fn constant_slot() {
    let rules = broken(
        "declare i8 @llvm_metal.load(i32, i64)
         define void @kernel.k() { call void @read(i32 0)  ret void }
         define void @read(i32 %slot) { %b = call i8 @llvm_metal.load(i32 %slot, i64 0)  ret void }",
    );
    assert_eq!(rules, [Rule::ConstantSlot]);
}

#[test]
fn direct_call() {
    let rules = broken(
        "define void @kernel.k() { %f = alloca ptr  %g = load ptr, ptr %f  call void %g()  ret void }",
    );
    assert_eq!(rules, [Rule::DirectCall]);
}

#[test]
fn no_recursion() {
    let rules = broken(
        "define void @kernel.k() { call void @a()  ret void }
         define void @a() { call void @b()  ret void }
         define void @b() { call void @a()  ret void }",
    );
    assert_eq!(rules, [Rule::NoRecursion, Rule::NoRecursion]);
}

#[test]
fn native_integer() {
    let rules = broken("define void @kernel.k() { %x = add i128 1, 2  ret void }");
    assert_eq!(rules, [Rule::NativeInteger]);
}

#[test]
fn no_integer_to_pointer() {
    let rules = broken(
        "define void @kernel.k() { %p = alloca i8  %n = ptrtoint ptr %p to i64  %q = inttoptr i64 %n to ptr  ret void }",
    );
    assert_eq!(rules, [Rule::NoIntegerToPointer]);
}

#[test]
fn constant_data() {
    let mutable = broken(
        "@state = global i32 0  define void @kernel.k() { store i32 1, ptr @state  ret void }",
    );
    assert_eq!(mutable, [Rule::ConstantData]);
    let function = broken(
        "@table = constant ptr @kernel.k
         define void @kernel.k() { %f = load ptr, ptr @table  ret void }",
    );
    assert_eq!(function, [Rule::ConstantData]);
    let constant = broken(
        "@name = constant [2 x i8] c\"ab\"  @place = constant { ptr, i32 } { ptr @name, i32 7 }
         define void @kernel.k() { %p = load ptr, ptr @place  ret void }",
    );
    assert_eq!(constant, []);
}

#[test]
fn a_panic_is_an_operation_and_what_only_it_reached_is_gone() {
    let context = Context::create();
    let source = "declare void @core_panic(ptr) noreturn
         @message = constant ptr @formatter
         define void @formatter() { call void @unknown()  ret void }
         declare void @unknown()
         define i32 @checked(i1 %bad) { br i1 %bad, label %fail, label %fine
           fail:  call void @core_panic(ptr @message)  unreachable
           fine:  ret i32 7 }
         define void @kernel.k() { %x = call i32 @checked(i1 false)  ret void }";
    let buffer = MemoryBuffer::create_from_memory_range_copy(source.as_bytes(), "panic");
    let module = context.create_module_from_ir(buffer).unwrap();
    let bitcode = module.write_bitcode_to_memory().as_slice().to_vec();
    let program = program(&context, &[bitcode], "k").unwrap();
    let text = program.print_to_string().to_string();
    assert!(
        text.contains("call void @llvm_metal.panic()\n  ret i32 0"),
        "{text}"
    );
    assert!(
        !text.contains("formatter") && !text.contains("core_panic"),
        "{text}"
    );
}
