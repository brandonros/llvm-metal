use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{
    air::{InliningPolicy, legalize_with_policy},
    parse_ir,
};

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
fn source() -> &'static str {
    r#"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
declare i64 @llvm.umax.i64(i64, i64)
define internal i64 @leaf(i64 %x) {
  %a = mul i64 %x, 17
  %b = call i64 @llvm.umax.i64(i64 %a, i64 100)
  ret i64 %b
}
define internal i64 @nested(i64 %x) {
  %a = call i64 @leaf(i64 %x)
  %b = add i64 %x, 3
  %c = call i64 @leaf(i64 %b)
  %d = xor i64 %a, %c
  ret i64 %d
}
define void @kernel(ptr %p) {
  %x = load i64, ptr %p, align 8
  %a = call i64 @nested(i64 %x)
  %y = add i64 %x, 11
  %b = call i64 @nested(i64 %y)
  %z = add i64 %a, %b
  %q = getelementptr i64, ptr %p, i64 1
  store i64 %z, ptr %q, align 8
  ret void
}
"#
}
#[test]
fn scalar_calls_survive_and_helpers_receive_intrinsic_legalization() {
    let context = Context::create();
    let input = parse_ir(&context, source().as_bytes(), "calls").unwrap();
    let before = input.print_to_string().to_string();
    let (air, _) =
        legalize_with_policy(&input, &interface(), InliningPolicy::RetainScalar).unwrap();
    air.verify().unwrap();
    assert_eq!(
        air.get_functions()
            .filter(|f| f.count_basic_blocks() != 0)
            .count(),
        3
    );
    let text = air.print_to_string().to_string();
    assert!(text.contains("call i64 @nested"));
    assert!(text.contains("call i64 @leaf"));
    assert!(!text.contains("llvm.umax"));
    assert_eq!(input.print_to_string().to_string(), before);
    let (baseline, _) = legalize_with_policy(&input, &interface(), InliningPolicy::All).unwrap();
    assert_eq!(
        baseline
            .get_functions()
            .filter(|f| f.count_basic_blocks() != 0)
            .count(),
        1
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn retained_nested_calls_survive_writer_and_execute_on_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for policy in [InliningPolicy::All, InliningPolicy::RetainScalar] {
        let context = Context::create();
        let input = parse_ir(&context, source().as_bytes(), "calls").unwrap();
        let artifact =
            llvm_metal_compiler::compile::compile_with_policy(&input, &interface(), policy)
                .unwrap();
        let decoded =
            llvm_metal_compiler::parse_bitcode(&context, &artifact.air_bitcode, "AIR").unwrap();
        assert_eq!(
            decoded
                .get_functions()
                .filter(|f| f.count_basic_blocks() != 0)
                .count(),
            if policy == InliningPolicy::All { 1 } else { 3 }
        );
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        for x in [0u64, 1, 5, 6, 100, u32::MAX as u64, u64::MAX - 3, u64::MAX] {
            let mut buffers = [Buffer {
                bytes: vec![0xa5; 528],
                offset: 256,
            }];
            buffers[0].bytes[256..264].copy_from_slice(&x.to_le_bytes());
            // SAFETY: one invocation; initialized aligned 16-byte region and guards.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            let leaf = |v: u64| v.wrapping_mul(17).max(100);
            let nested = |v: u64| leaf(v) ^ leaf(v.wrapping_add(3));
            let expected = nested(x).wrapping_add(nested(x.wrapping_add(11)));
            assert_eq!(&buffers[0].bytes[264..272], &expected.to_le_bytes());
            assert_eq!(&buffers[0].bytes[256..264], &x.to_le_bytes());
            assert!(
                buffers[0].bytes[..256]
                    .iter()
                    .chain(&buffers[0].bytes[272..])
                    .all(|b| *b == 0xa5)
            );
        }
    }
}

fn pointer_source() -> String {
    source().split("declare i64").next().unwrap().to_owned()
        + r#"
@table = constant [2 x i64] [i64 37, i64 91]
define internal i64 @read(ptr %p) noinline {
  %v = load i64, ptr %p, align 8
  ret i64 %v
}
define internal i64 @nested_read(ptr %p) noinline {
  %v = call i64 @read(ptr %p)
  %r = add i64 %v, 2
  ret i64 %r
}
define internal void @wipe(ptr %p) noinline {
  store volatile i64 0, ptr %p, align 8
  ret void
}
define void @kernel(ptr %p) {
  %slot = alloca i64, align 8
  store i64 7, ptr %slot, align 8
  %a = call i64 @nested_read(ptr %p)
  %b = call i64 @nested_read(ptr %slot)
  %c = call i64 @nested_read(ptr getelementptr ([2 x i64], ptr @table, i64 0, i64 1))
  call void @wipe(ptr %slot)
  %zero = load volatile i64, ptr %slot, align 8
  %ab = add i64 %a, %b
  %abc = add i64 %ab, %c
  %sum = add i64 %abc, %zero
  %out = getelementptr i64, ptr %p, i64 1
  store i64 %sum, ptr %out, align 8
  ret void
}
"#
}
fn struct_pointer_source() -> String {
    pointer_source().replace("@table = constant [2 x i64] [i64 37, i64 91]", "@table = constant <{ [2 x i64], [7 x i8] }> <{ [2 x i64] [i64 37, i64 91], [7 x i8] undef }>, align 8")
}

#[test]
fn constant_struct_with_undefined_padding_preserves_retained_table_access() {
    let context = Context::create();
    let input = parse_ir(
        &context,
        struct_pointer_source().as_bytes(),
        "constant-struct",
    )
    .unwrap();
    let artifact = llvm_metal_compiler::compile::compile(&input, &interface()).unwrap();
    let air =
        llvm_metal_compiler::parse_bitcode(&context, &artifact.air_bitcode, "struct-air").unwrap();
    assert!(air.get_function("nested_read.metal.2").is_some());
    assert!(
        air.get_global("table")
            .unwrap()
            .get_initializer()
            .unwrap()
            .is_struct_value()
    );
}

#[test]
fn helpers_specialize_for_device_private_and_constant_pointers() {
    let context = Context::create();
    let input = parse_ir(&context, pointer_source().as_bytes(), "pointers").unwrap();
    let (air, _) = legalize_with_policy(&input, &interface(), InliningPolicy::Selective).unwrap();
    air.verify().unwrap();
    let text = air.print_to_string().to_string();
    for name in [
        "read.metal.0",
        "read.metal.1",
        "read.metal.2",
        "nested_read.metal.0",
        "nested_read.metal.1",
        "nested_read.metal.2",
        "wipe.metal.0",
    ] {
        assert!(air.get_function(name).is_some(), "missing {name}: {text}");
    }
    assert!(!text.contains("addrspacecast"));
    assert!(text.contains("store volatile i64 0"));
    assert!(text.contains("load volatile i64"));
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn specialized_pointer_calls_execute_on_gpu() {
    use llvm_metal_runtime::{Buffer, Kernel};
    for source in [pointer_source(), struct_pointer_source()] {
        let context = Context::create();
        let input = parse_ir(&context, source.as_bytes(), "pointers").unwrap();
        let artifact = llvm_metal_compiler::compile::compile_with_policy(
            &input,
            &interface(),
            InliningPolicy::Selective,
        )
        .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("kernel.metallib");
        std::fs::write(&path, artifact.metallib).unwrap();
        let kernel = Kernel::load(&path, &artifact.bindings).unwrap();
        for x in [0u64, 100, u64::MAX] {
            let mut buffers = [Buffer {
                bytes: vec![0xa5; 528],
                offset: 256,
            }];
            buffers[0].bytes[256..264].copy_from_slice(&x.to_le_bytes());
            // SAFETY: one invocation with disjoint initialized storage and output guards.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            assert_eq!(
                &buffers[0].bytes[264..272],
                &x.wrapping_add(104).to_le_bytes()
            );
            assert_eq!(&buffers[0].bytes[256..264], &x.to_le_bytes());
            assert!(
                buffers[0].bytes[..256]
                    .iter()
                    .chain(&buffers[0].bytes[272..])
                    .all(|b| *b == 0xa5)
            );
        }
    }
}

#[test]
fn recursive_indirect_escaping_and_mismatched_calls_are_rejected() {
    let header = source().split("declare i64").next().unwrap();
    let cases = [
        (
            "define internal i64 @a(i64 %x) { %r = call i64 @a(i64 %x)\nret i64 %r }\ndefine void @kernel(ptr %p) { %r = call i64 @a(i64 1)\nstore i64 %r, ptr %p\nret void }",
            "recursive",
        ),
        (
            "define internal i64 @a(i64 %x) { %r = call i64 @b(i64 %x)\nret i64 %r }\ndefine internal i64 @b(i64 %x) { %r = call i64 @a(i64 %x)\nret i64 %r }\ndefine void @kernel(ptr %p) { %r = call i64 @a(i64 1)\nstore i64 %r, ptr %p\nret void }",
            "recursive",
        ),
        (
            "define void @kernel(ptr %p) { %f = load ptr, ptr %p\ncall void %f()\nret void }",
            "indirect",
        ),
        (
            "define internal i64 @a(i64 %x) { ret i64 %x }\ndefine void @kernel(ptr %p) { store ptr @a, ptr %p\nret void }",
            "escapes",
        ),
        (
            "define internal fastcc i64 @a(i64 %x) { ret i64 %x }\ndefine void @kernel(ptr %p) { %r = call i64 @a(i64 1)\nstore i64 %r, ptr %p\nret void }",
            "calling convention",
        ),
        (
            "define internal i64 @a(i64 %x) { ret i64 %x }\ndefine void @kernel(ptr %p) { %r = call i64 @a(i64 1) [\"deopt\"(i32 7)]\nstore i64 %r, ptr %p\nret void }",
            "operand bundles",
        ),
    ];
    for (body, error) in cases {
        let context = Context::create();
        let input = parse_ir(&context, format!("{header}{body}").as_bytes(), "negative").unwrap();
        let before = input.print_to_string().to_string();
        let actual =
            legalize_with_policy(&input, &interface(), InliningPolicy::Selective).unwrap_err();
        assert!(actual.contains(error), "wanted {error}; got {actual}");
        assert_eq!(input.print_to_string().to_string(), before);
    }
}

#[test]
fn required_wide_and_thread_context_inlining_is_preserved() {
    let context = Context::create();
    let header = source().split("declare i64").next().unwrap();
    let body = r#"
declare i32 @llvm_metal.linear_thread_index()
define internal i128 @wide(i128 %x) noinline { %r = add i128 %x, 7
ret i128 %r }
define internal i32 @index() noinline { %n = call i32 @llvm_metal.linear_thread_index()
ret i32 %n }
define internal i32 @nested_index() noinline { %n = call i32 @index()
ret i32 %n }
define void @kernel(ptr %p) {
 %i = call i32 @nested_index()
 %n = zext i32 %i to i128
 %wide = call i128 @wide(i128 %n)
 %r = trunc i128 %wide to i64
 store i64 %r, ptr %p
 ret void
}
"#;
    let input = parse_ir(&context, format!("{header}{body}").as_bytes(), "required").unwrap();
    let mut contract = interface();
    contract.dispatch = Dispatch::Grid1d;
    contract.invocations = None;
    let (air, _) = legalize_with_policy(&input, &contract, InliningPolicy::Selective).unwrap();
    let text = air.print_to_string().to_string();
    assert_eq!(
        air.get_functions()
            .filter(|f| f.count_basic_blocks() != 0)
            .count(),
        1
    );
    assert!(!text.contains("i128"));
    assert!(!text.contains("llvm_metal.linear_thread_index"));
}

#[test]
fn internal_fastcc_and_every_call_site_are_retargeted_together() {
    let context = Context::create();
    let text = source()
        .replace("define internal i64", "define internal fastcc i64")
        .replace("call i64 @leaf", "call fastcc i64 @leaf")
        .replace("call i64 @nested", "call fastcc i64 @nested");
    let input = parse_ir(&context, text.as_bytes(), "fastcc").unwrap();
    let (air, _) =
        legalize_with_policy(&input, &interface(), InliningPolicy::RetainScalar).unwrap();
    assert!(air.get_function("leaf").is_some());
    assert!(air.get_functions().all(|f| f.get_call_conventions() == 0));
    for i in air
        .get_functions()
        .flat_map(|f| f.get_basic_blocks())
        .flat_map(|b| b.get_instructions())
    {
        if let Ok(call) = inkwell::values::CallSiteValue::try_from(i) {
            assert_eq!(call.get_call_convention(), 0);
        }
    }
}

#[test]
fn private_pointer_joins_are_supported_but_unproven_loaded_pointers_are_refused() {
    let context = Context::create();
    let joined = pointer_source().replace("%a = call i64 @nested_read(ptr %p)", "%other = alloca i64, align 8\nstore i64 8, ptr %other\n%x = load i64, ptr %p\n%cond = icmp eq i64 %x, 0\n%selected = select i1 %cond, ptr %slot, ptr %other\n%a = call i64 @nested_read(ptr %selected)");
    let input = parse_ir(&context, joined.as_bytes(), "private-join").unwrap();
    legalize_with_policy(&input, &interface(), InliningPolicy::Selective).unwrap();
    let unknown = pointer_source().replace(
        "%a = call i64 @nested_read(ptr %p)",
        "%unknown = load ptr, ptr %p\n%a = call i64 @nested_read(ptr %unknown)",
    );
    let input = parse_ir(&context, unknown.as_bytes(), "unknown-pointer").unwrap();
    assert!(
        legalize_with_policy(&input, &interface(), InliningPolicy::Selective)
            .unwrap_err()
            .contains("unresolved pointer flow")
    );
}

#[test]
fn legacy_writer_drops_modern_facts_but_preserves_sret_abi() {
    let context = Context::create();
    let header = source().split("declare i64").next().unwrap();
    let input = format!(
        r#"{header}
define internal void @write(ptr sret(i64) writable initializes((0, 8)) captures(none) %p, i64 %x) noinline {{
 store i64 %x, ptr %p, align 8
 ret void
}}
define void @kernel(ptr %p) {{
 %x = load i64, ptr %p, align 8
 %slot = alloca i64, align 8
 call void @write(ptr sret(i64) writable initializes((0, 8)) captures(none) %slot, i64 %x)
 %r = load i64, ptr %slot, align 8
 %out = getelementptr i64, ptr %p, i64 1
 store i64 %r, ptr %out, align 8
 ret void
}}
"#
    );
    let input = parse_ir(&context, input.as_bytes(), "modern-facts").unwrap();
    let artifact = llvm_metal_compiler::compile::compile_with_policy(
        &input,
        &interface(),
        InliningPolicy::Selective,
    )
    .unwrap();
    assert!(artifact.air_ir.contains("sret(i64)"));
    assert!(!artifact.air_ir.contains("captures("));
    assert!(!artifact.air_ir.contains("initializes("));
    let decoded =
        llvm_metal_compiler::parse_bitcode(&context, &artifact.air_bitcode, "legacy").unwrap();
    assert!(decoded.get_function("write.metal.0").is_some());
}

#[test]
fn pointer_return_interfaces_remain_on_the_required_inlining_path() {
    let context = Context::create();
    let text = pointer_source().replace(
        "%a = call i64 @nested_read(ptr %p)",
        "%next = call ptr @offset(ptr %p)\n%a = call i64 @nested_read(ptr %next)",
    ) + "\ndefine internal ptr @offset(ptr %p) noinline { %q = getelementptr i64, ptr %p, i64 1\nret ptr %q }\n";
    let input = parse_ir(&context, text.as_bytes(), "pointer-return").unwrap();
    let (air, _) = legalize_with_policy(&input, &interface(), InliningPolicy::Selective).unwrap();
    assert!(air.get_function("offset").is_none());
    assert!(air.get_function("nested_read.metal.1").is_some());
}

#[test]
fn preparation_marks_retained_boundaries_for_later_legalization() {
    let context = Context::create();
    let input = parse_ir(&context, source().as_bytes(), "prepare").unwrap();
    let prepared =
        llvm_metal_compiler::calls::prepare(&input, "kernel", InliningPolicy::RetainScalar)
            .unwrap();
    for name in ["leaf", "nested"] {
        assert!(
            prepared
                .get_function(name)
                .unwrap()
                .get_enum_attribute(
                    inkwell::attributes::AttributeLoc::Function,
                    inkwell::attributes::Attribute::get_named_enum_kind_id("noinline")
                )
                .is_some()
        );
    }
    let (air, _) =
        legalize_with_policy(&prepared, &interface(), InliningPolicy::RetainScalar).unwrap();
    assert!(air.get_function("leaf").is_some());
}

#[test]
fn byval_interfaces_inline_before_pointer_specialization() {
    let context = Context::create();
    let text = pointer_source().replace("@read(ptr %p)", "@read(ptr byval(i64) %p)");
    let input = parse_ir(&context, text.as_bytes(), "byval").unwrap();
    let (air, _) = legalize_with_policy(&input, &interface(), InliningPolicy::Selective).unwrap();
    assert!(!air.print_to_string().to_string().contains("byval("));
    assert!(air.get_function("read.metal.1").is_none());
    assert!(air.get_function("nested_read.metal.1").is_some());
}

#[test]
fn public_api_and_cli_default_to_retention_with_full_inlining_opt_out() {
    let context = Context::create();
    let input = parse_ir(&context, pointer_source().as_bytes(), "default-policy").unwrap();
    let artifact = llvm_metal_compiler::compile::compile(&input, &interface()).unwrap();
    let decoded =
        llvm_metal_compiler::parse_bitcode(&context, &artifact.air_bitcode, "default-air").unwrap();
    assert!(decoded.get_function("nested_read.metal.1").is_some());
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("input.ll");
    let contract = directory.path().join("interface.json");
    std::fs::write(&source, pointer_source()).unwrap();
    std::fs::write(&contract, serde_json::to_vec(&interface()).unwrap()).unwrap();
    for all in [false, true] {
        let output = directory.path().join(if all { "all" } else { "default" });
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_llvm-metalc"));
        command
            .arg("compile")
            .arg(&source)
            .arg("--interface")
            .arg(&contract)
            .arg("--output")
            .arg(&output);
        if all {
            command.args(["--inlining", "all"]);
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let bytes = std::fs::read(output.join("kernel.air.bc")).unwrap();
        let module = llvm_metal_compiler::parse_bitcode(&context, &bytes, "cli-air").unwrap();
        assert_eq!(
            module
                .get_functions()
                .filter(|f| f.count_basic_blocks() != 0)
                .count()
                > 1,
            !all
        );
    }
}
