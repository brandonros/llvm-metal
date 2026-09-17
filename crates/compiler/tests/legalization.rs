use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};

fn interface() -> KernelInterface {
    KernelInterface {
        schema: 1,
        entry: "kernel".into(),
        calling_convention: "C".into(),
        invocations: Some(1),
        dispatch: llvm_metal_abi::Dispatch::Single,
        aliasing: "disjoint".into(),
        arguments: vec![BufferArgument {
            name: "data".into(),
            kind: "buffer".into(),
            access: Access::ReadWrite,
            bytes: 4,
            alignment: 4,
        }],
    }
}

fn source(body: &str) -> String {
    format!(
        "target triple = \"nvptx64-nvidia-cuda\"\ntarget datalayout = \"e-p:64:64-i64:64-i128:128\"\ndefine void @kernel(ptr %p) {{\n{body}\nret void\n}}"
    )
}

#[test]
fn legalization_changes_buffer_address_space_and_preserves_input() {
    let context = Context::create();
    let module = parse_ir(
        &context,
        source(
            "%x = load i32, ptr %p, align 4\n%y = add i32 %x, 42\nstore i32 %y, ptr %p, align 4",
        )
        .as_bytes(),
        "input",
    )
    .unwrap();
    let original = module.print_to_string().to_string();
    let (air, bindings) = legalize(&module, &interface()).unwrap();
    let result = air.print_to_string().to_string();
    assert!(result.contains("ptr addrspace(1)"));
    assert!(!result.contains("addrspacecast"));
    assert_eq!(air.get_global_metadata_size("air.kernel"), 1);
    assert_eq!(bindings.buffers[0].minimum_bytes, 4);
    assert_eq!(module.print_to_string().to_string(), original);
}

#[test]
fn rejects_unimplemented_semantics_without_mutating_input() {
    let context = Context::create();
    for body in [
        "%x = atomicrmw add ptr %p, i32 1 monotonic",
        "%x = load volatile i32, ptr %p",
        "%x = load i128, ptr %p",
        "%x = load float, ptr %p",
        "call void asm sideeffect \"\", \"\"()",
        "%n = ptrtoint ptr %p to i64",
    ] {
        let module = parse_ir(&context, source(body).as_bytes(), "unsupported").unwrap();
        let before = module.print_to_string().to_string();
        assert!(legalize(&module, &interface()).is_err(), "{body}");
        assert_eq!(module.print_to_string().to_string(), before);
    }
}

#[test]
fn rejects_wrong_target_layout_signature_and_invocation_contract() {
    let context = Context::create();
    let valid = source("%x = load i32, ptr %p");
    for text in [
        valid.replace("nvptx64-nvidia-cuda", "aarch64-apple-darwin"),
        valid.replace("p:64:64", "p:32:32"),
        valid
            .replace("ptr %p", "i32 %p")
            .replace("%x = load i32, i32 %p", ""),
    ] {
        let module = parse_ir(&context, text.as_bytes(), "unsupported").unwrap();
        assert!(legalize(&module, &interface()).is_err());
    }
    let module = parse_ir(&context, valid.as_bytes(), "valid").unwrap();
    let mut contract = interface();
    contract.invocations = Some(2);
    assert!(legalize(&module, &contract).is_err());
}

#[test]
fn constant_tables_and_codegen_flags_are_retargeted() {
    let context = Context::create();
    let text = source(
        "%i = load i32, ptr %p\n%j = and i32 %i, 3\n%q = getelementptr [4 x i32], ptr @table, i32 0, i32 %j\n%v = load i32, ptr %q\nstore i32 %v, ptr %p",
    ) + "\n@table = constant [4 x i32] [i32 3, i32 5, i32 7, i32 9]\n!llvm.module.flags = !{!0, !1}\n!0 = !{i32 8, !\"PIC Level\", i32 2}\n!1 = !{i32 7, !\"PIE Level\", i32 2}\n";
    let module = parse_ir(&context, text.as_bytes(), "table").unwrap();
    let (air, _) = legalize(&module, &interface()).unwrap();
    let ir = air.print_to_string().to_string();
    assert!(ir.contains("addrspace(2) constant"));
    assert!(!ir.contains("addrspacecast"));
    assert!(!ir.contains("PIC Level"));
    assert!(!ir.contains("PIE Level"));
    let mutable = text.replace("@table = constant", "@table = global");
    assert!(
        legalize(
            &parse_ir(&context, mutable.as_bytes(), "mutable").unwrap(),
            &interface()
        )
        .is_err()
    );
}

#[test]
fn unsigned_three_way_comparison_is_lowered() {
    let context = Context::create();
    let text = source(
        "%x = load i32, ptr %p\n%c = call i32 @llvm.ucmp.i32.i32(i32 %x, i32 2147483648)\nstore i32 %c, ptr %p",
    ) + "\ndeclare i32 @llvm.ucmp.i32.i32(i32, i32)\n";
    let module = parse_ir(&context, text.as_bytes(), "compare").unwrap();
    let (air, _) = legalize(&module, &interface()).unwrap();
    let ir = air.print_to_string().to_string();
    assert!(!ir.contains("llvm.ucmp"));
    assert!(ir.contains("icmp ult"));
    assert!(ir.contains("icmp ugt"));
}

#[test]
fn volatile_intrinsics_and_invalid_device_operations_are_rejected() {
    let context = Context::create();
    for (body, declaration) in [
        (
            "call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 4, i1 true)",
            "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)",
        ),
        (
            "%x = call i64 @llvm_metal.linear_thread_index()",
            "declare i64 @llvm_metal.linear_thread_index()",
        ),
        (
            "%x = call i32 @llvm_metal.atomic_add_device_u32(ptr %p)",
            "declare i32 @llvm_metal.atomic_add_device_u32(ptr)",
        ),
        (
            "%x = call i32 @llvm_metal.linear_thread_index()",
            "declare i32 @llvm_metal.linear_thread_index()",
        ),
    ] {
        let text = format!("{}\n{declaration}", source(body));
        let module = parse_ir(&context, text.as_bytes(), "unsupported").unwrap();
        assert!(legalize(&module, &interface()).is_err(), "{text}");
    }
}

#[test]
fn wide_unsupported_operations_fail_without_mutating_input() {
    let context = Context::create();
    for body in [
        "%x = load i128, ptr %p",
        "store volatile i128 1, ptr %p",
        "store atomic i128 1, ptr %p seq_cst, align 16",
        "%x = load i64, ptr %p\n%w = zext i64 %x to i128\n%r = udiv i128 %w, 3\nstore i128 %r, ptr %p",
        "%x = load i64, ptr %p\n%w = zext i64 %x to i128\n%r = lshr i128 123, %w\nstore i128 %r, ptr %p",
        "%x = load i64, ptr %p\n%w = zext i64 %x to i128\n%r = lshr i128 %w, 128\nstore i128 %r, ptr %p",
    ] {
        let module = parse_ir(&context, source(body).as_bytes(), "wide-refusal").unwrap();
        let before = module.print_to_string().to_string();
        let error = legalize(&module, &interface()).unwrap_err();
        assert!(error.contains("unsupported i128"), "{error}");
        assert_eq!(module.print_to_string().to_string(), before);
    }
}

#[test]
fn private_volatile_barrier_survives_helper_inlining() {
    let context = Context::create();
    let text =
        source("%x = load i8, ptr %p\n%y = call i8 @barrier(i8 %x) noinline\nstore i8 %y, ptr %p")
            + "
        define internal i8 @barrier(i8 %x) noinline {
            %slot = alloca i8
            store i8 %x, ptr %slot
            %value = load volatile i8, ptr %slot
            ret i8 %value
        }";
    let module = parse_ir(&context, text.as_bytes(), "barrier").unwrap();
    let (air, _) = legalize(&module, &interface()).unwrap();
    let text = air.print_to_string().to_string();
    assert!(text.contains("load volatile i8"));
    assert!(air.get_function("barrier").is_none());
}

#[test]
fn private_volatile_copies_preserve_accesses_and_reject_device_endpoints() {
    let context = Context::create();
    let declaration = "\ndeclare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)\n";
    let text = source(
        "%a = alloca [4 x i8]\n%b = alloca [4 x i8]\nstore i32 305419896, ptr %a, align 1\ncall void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 4, i1 true)\n%x = load i32, ptr %b, align 1\nstore i32 %x, ptr %p, align 1",
    ) + declaration;
    let module = parse_ir(&context, text.as_bytes(), "private-copy").unwrap();
    let (air, _) = legalize(&module, &interface()).unwrap();
    let text = air.print_to_string().to_string();
    assert_eq!(text.matches("load volatile i8").count(), 4);
    assert_eq!(text.matches("store volatile i8").count(), 4);
    for copy in [
        "call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %a, i64 4, i1 true)",
        "call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %p, i64 4, i1 true)",
        "call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 257, i1 true)",
    ] {
        let text = source(&format!(
            "%a = alloca [257 x i8]\n%b = alloca [257 x i8]\n{copy}"
        )) + declaration;
        let module = parse_ir(&context, text.as_bytes(), "unsupported-copy").unwrap();
        assert!(legalize(&module, &interface()).is_err());
    }
}
