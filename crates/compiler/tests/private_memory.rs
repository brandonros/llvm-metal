use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};

const HEADER: &str =
    "target triple = \"nvptx64-nvidia-cuda\"\ntarget datalayout = \"e-p:64:64-i64:64-i128:128\"\n";
const INTRINSICS: &str = "\ndeclare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)\ndeclare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)\n";
fn interface(bytes: usize) -> KernelInterface {
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
            bytes,
            alignment: 8,
        }],
    }
}
fn source(body: &str) -> String {
    format!("{HEADER}define void @kernel(ptr %p) {{\n{body}\nret void\n}}\n{INTRINSICS}")
}

fn wide_source(alignment: u32, volatile_load: bool) -> String {
    let volatile = if volatile_load { "volatile " } else { "" };
    let offset = if alignment == 1 { 1 } else { 0 };
    source(&format!(
        r#"
      %storage = alloca [48 x i8], align 16
      %slot = getelementptr i8, ptr %storage, i64 {offset}
      %lo = load i64, ptr %p, align 8
      %hi_in = getelementptr i8, ptr %p, i64 8
      %hi = load i64, ptr %hi_in, align 8
      store i64 %lo, ptr %slot, align {alignment}
      %hi_slot = getelementptr i8, ptr %slot, i64 8
      store i64 %hi, ptr %hi_slot, align 1
      %wide = load {volatile}i128, ptr %slot, align {alignment}
      %sum = add i128 %wide, 18446744073709551617
      %destination = getelementptr i8, ptr %slot, i64 16
      store volatile i128 %sum, ptr %destination, align {alignment}
      %result = load i128, ptr %destination, align {alignment}
      %out = getelementptr i8, ptr %p, i64 16
      store i128 %result, ptr %out, align 8
    "#
    ))
}

#[test]
fn private_wide_accesses_split_and_keep_volatile_semantics() {
    for alignment in [1, 8, 16] {
        for volatile in [false, true] {
            let context = Context::create();
            let module = parse_ir(
                &context,
                wide_source(alignment, volatile).as_bytes(),
                "private-wide",
            )
            .unwrap();
            let original = module.print_to_string().to_string();
            let (air, _) = legalize(&module, &interface(32)).unwrap();
            air.verify().unwrap();
            let ir = air.print_to_string().to_string();
            assert!(!ir.contains("i128"));
            assert_eq!(ir.matches("store volatile i64").count(), 2);
            assert_eq!(
                ir.matches("load volatile i64").count(),
                if volatile { 2 } else { 0 }
            );
            assert_eq!(module.print_to_string().to_string(), original);
        }
    }
}

// Copy through two private arrays with different initialized 16-byte guards.
// Export both arrays so GPU tests inspect the source, destination and both guards.
fn aggregate_source(length: usize) -> String {
    let allocation = length + 32;
    let destination_dump = length * 2 + 32;
    source(&format!(
        r#"
      %a = alloca [{allocation} x i8], align 8
      %b = alloca [{allocation} x i8], align 8
      call void @llvm.memset.p0.i64(ptr %a, i8 90, i64 {allocation}, i1 false)
      call void @llvm.memset.p0.i64(ptr %b, i8 150, i64 {allocation}, i1 false)
      %from = getelementptr i8, ptr %a, i64 16
      %to = getelementptr i8, ptr %b, i64 16
      call void @llvm.memcpy.p0.p0.i64(ptr %from, ptr %p, i64 {length}, i1 false)
      call void @llvm.memcpy.p0.p0.i64(ptr %to, ptr %from, i64 {length}, i1 true)
      %source_dump = getelementptr i8, ptr %p, i64 {length}
      %destination_dump = getelementptr i8, ptr %p, i64 {destination_dump}
      call void @llvm.memcpy.p0.p0.i64(ptr %source_dump, ptr %a, i64 {allocation}, i1 false)
      call void @llvm.memcpy.p0.p0.i64(ptr %destination_dump, ptr %b, i64 {allocation}, i1 false)
    "#
    ))
}

#[test]
fn bounded_private_aggregate_copies_expand_without_losing_volatile_accesses() {
    for length in [0usize, 257, 520, 1072, 4096] {
        let context = Context::create();
        let module = parse_ir(
            &context,
            aggregate_source(length).as_bytes(),
            "private-aggregate",
        )
        .unwrap();
        let original = module.print_to_string().to_string();
        let (air, _) = legalize(&module, &interface(length * 3 + 64)).unwrap();
        air.verify().unwrap();
        let ir = air.print_to_string().to_string();
        assert_eq!(ir.matches("load volatile i8").count(), length);
        assert_eq!(ir.matches("store volatile i8").count(), length);
        assert_eq!(module.print_to_string().to_string(), original);
    }
}

#[test]
fn device_atomic_unbounded_and_oversize_accesses_remain_rejected() {
    let prelude = "%a = alloca [4112 x i8], align 16\n%b = alloca [4112 x i8], align 16\n";
    for body in [
        "%wide = load i128, ptr %p, align 8",
        "%wide = load volatile i128, ptr %p, align 8",
        "store volatile i128 1, ptr %p, align 8",
        "%wide = load atomic i128, ptr %a seq_cst, align 16",
        "store atomic i128 1, ptr %a seq_cst, align 16",
        "%wide = atomicrmw add ptr %a, i128 1 monotonic, align 16",
        "call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 4097, i1 true)",
        "%length = load i64, ptr %p, align 8\ncall void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 %length, i1 true)",
        "call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %a, i64 520, i1 true)",
        "call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %p, i64 1072, i1 true)",
        "%condition = load i1, ptr %p\n%mixed = select i1 %condition, ptr %a, ptr %p\ncall void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %mixed, i64 520, i1 true)",
        "call void @llvm.memset.p0.i64(ptr %a, i8 0, i64 520, i1 true)",
    ] {
        let context = Context::create();
        let text = source(&format!("{prelude}{body}"));
        let module = parse_ir(&context, text.as_bytes(), "private-refusal").unwrap();
        let original = module.print_to_string().to_string();
        assert!(legalize(&module, &interface(12352)).is_err(), "{body}");
        assert_eq!(module.print_to_string().to_string(), original);
    }
}

#[cfg(target_os = "macos")]
fn gpu_kernel(source: &str, bytes: usize) -> (llvm_metal_runtime::Kernel, tempfile::TempDir) {
    let context = Context::create();
    let module = parse_ir(&context, source.as_bytes(), "private-memory-gpu").unwrap();
    let artifact = llvm_metal_compiler::compile::compile(&module, &interface(bytes)).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("kernel.metallib");
    std::fs::write(&path, artifact.metallib).unwrap();
    (
        llvm_metal_runtime::Kernel::load(&path, &artifact.bindings).unwrap(),
        directory,
    )
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn private_wide_reads_carry_and_volatile_writes_match_gpu() {
    use llvm_metal_runtime::Buffer;
    for (alignment, volatile) in [(1, false), (8, true), (16, true)] {
        let (kernel, _directory) = gpu_kernel(&wide_source(alignment, volatile), 32);
        for value in [
            0u128,
            1,
            u64::MAX as u128,
            1 << 64,
            1 << 127,
            u128::MAX,
            0x123456789abcdef0fedcba9876543210,
        ] {
            let mut bytes = vec![0xa5; 544];
            bytes[256..272].copy_from_slice(&value.to_le_bytes());
            let mut expected = bytes.clone();
            expected[272..288]
                .copy_from_slice(&value.wrapping_add(18446744073709551617).to_le_bytes());
            let mut buffers = [Buffer { bytes, offset: 256 }];
            // SAFETY: one initialized, guarded 32-byte record, private offsets
            // in the reviewed module fit its local 48-byte allocation.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            assert_eq!(buffers[0].bytes, expected);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn rsa_sized_private_volatile_copies_preserve_payloads_and_guards_on_gpu() {
    use llvm_metal_runtime::Buffer;
    for length in [520usize, 1072] {
        let used = length * 3 + 64;
        let (kernel, _directory) = gpu_kernel(&aggregate_source(length), used);
        for fill in [0u8, 255, 83] {
            let mut bytes = vec![0xa5; used + 512];
            for (i, byte) in bytes[256..256 + length].iter_mut().enumerate() {
                *byte = fill.wrapping_add((i as u8).wrapping_mul(37));
            }
            let payload = bytes[256..256 + length].to_vec();
            let mut expected = bytes.clone();
            for (offset, guard) in [(256 + length, 90), (256 + length * 2 + 32, 150)] {
                expected[offset..offset + length + 32].fill(guard);
                expected[offset + 16..offset + 16 + length].copy_from_slice(&payload);
            }
            let mut buffers = [Buffer { bytes, offset: 256 }];
            // SAFETY: all source/destination ranges fit the ABI record; the
            // volatile transfer is disjoint and entirely within local arrays.
            unsafe {
                kernel.run(&mut buffers, 1, 1).unwrap();
            }
            assert_eq!(buffers[0].bytes, expected, "length={length}, fill={fill}");
        }
    }
}

fn constant_payload() -> Vec<u8> {
    (0..160)
        .map(|n| (n as u8).wrapping_mul(37).wrapping_add(11))
        .collect()
}
fn constant_definition() -> String {
    let values = constant_payload()
        .into_iter()
        .map(|v| format!("i8 {v}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("\n@table = private constant [160 x i8] [{values}], align 8\n")
}
fn constant_copy_source() -> String {
    source(
        r#"
      %storage = alloca [160 x i8], align 8
      call void @llvm.memset.p0.i64(ptr %storage, i8 150, i64 160, i1 false)
      %destination = getelementptr i8, ptr %storage, i64 16
      %constant = getelementptr i8, ptr @table, i64 16
      call void @llvm.memcpy.p0.p0.i64(ptr %destination, ptr %constant, i64 128, i1 true)
      call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %storage, i64 160, i1 false)
      %last = getelementptr i8, ptr @table, i64 159
      %byte = load volatile i8, ptr %last, align 1
      %out = getelementptr i8, ptr %p, i64 160
      store i8 %byte, ptr %out, align 1
    "#,
    ) + &constant_definition()
}

#[test]
fn defined_constant_sources_keep_volatile_reads_and_private_writes() {
    let context = Context::create();
    let module = parse_ir(&context, constant_copy_source().as_bytes(), "constant-copy").unwrap();
    let original = module.print_to_string().to_string();
    let (air, _) = legalize(&module, &interface(161)).unwrap();
    air.verify().unwrap();
    let ir = air.print_to_string().to_string();
    assert_eq!(ir.matches("load volatile i8").count(), 129);
    assert_eq!(ir.matches("store volatile i8").count(), 128);
    assert!(ir.contains("addrspace(2) constant"));
    assert!(!ir.contains("addrspacecast"), "{ir}");
    assert_eq!(module.print_to_string().to_string(), original);
}

#[test]
fn constant_destinations_and_untrusted_global_sources_are_refused() {
    let good = constant_copy_source();
    let copies = [
        // Writing through a constant pointer is never legalized as private memory.
        good.replace(
            "ptr %destination, ptr %constant, i64 128, i1 true",
            "ptr %constant, ptr %destination, i64 128, i1 true",
        ),
        good.replace("@table = private constant", "@table = private global"),
        good.replace(
            "@table = private constant",
            "@table = externally_initialized constant",
        ),
        source(
            "%a = alloca [128 x i8]\ncall void @llvm.memcpy.p0.p0.i64(ptr %a, ptr @unknown, i64 128, i1 true)",
        ) + "\n@unknown = external constant [128 x i8]\n",
        source("store volatile i8 0, ptr @table") + &constant_definition(),
        source("store volatile i128 0, ptr @table, align 8") + &constant_definition(),
        source("%v = load volatile i8, ptr @table\nstore i8 %v, ptr %p")
            + &constant_definition().replace(
                "@table = private constant",
                "@table = externally_initialized constant",
            ),
    ];
    for text in copies {
        let context = Context::create();
        let module = parse_ir(&context, text.as_bytes(), "constant-memory-refusal").unwrap();
        let original = module.print_to_string().to_string();
        assert!(legalize(&module, &interface(161)).is_err(), "{text}");
        assert_eq!(module.print_to_string().to_string(), original);
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Apple GPU and pinned llvm-downgrade"]
fn constant_to_private_volatile_copy_preserves_payload_and_guards_on_gpu() {
    use llvm_metal_runtime::Buffer;
    let (kernel, _directory) = gpu_kernel(&constant_copy_source(), 161);
    let bytes = vec![0xa5; 161 + 512];
    let mut expected = bytes.clone();
    expected[256..416].fill(150);
    expected[272..400].copy_from_slice(&constant_payload()[16..144]);
    expected[416] = constant_payload()[159];
    let mut buffers = [Buffer { bytes, offset: 256 }];
    // SAFETY: the 128-byte constant range is defined, its private destination is
    // guarded within a 160-byte allocation, and the output has 161 writable bytes.
    unsafe {
        kernel.run(&mut buffers, 1, 1).unwrap();
    }
    assert_eq!(buffers[0].bytes, expected);
}

#[test]
fn unresolved_constant_pointer_cast_is_rejected_before_serialization() {
    let text = source(
        r#"
        %condition = load i1, ptr %p
        %choice = select i1 %condition, ptr inttoptr (i64 1 to ptr), ptr %p
        %byte = load i8, ptr %choice
        %out = getelementptr i8, ptr %p, i64 1
        store i8 %byte, ptr %out
    "#,
    );
    let context = Context::create();
    let module = parse_ir(&context, text.as_bytes(), "constant-cast-refusal").unwrap();
    let original = module.print_to_string().to_string();
    let error = legalize(&module, &interface(2)).unwrap_err();
    assert!(
        error.contains("constant pointer escaped address-space inference"),
        "{error}"
    );
    assert_eq!(module.print_to_string().to_string(), original);
}
