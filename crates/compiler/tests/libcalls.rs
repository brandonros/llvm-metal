use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};
fn interface() -> KernelInterface {
    KernelInterface { schema: 1, entry: "kernel".into(), calling_convention: "C".into(), invocations: Some(1), dispatch: Dispatch::Single, aliasing: "disjoint".into(), arguments: [("a", Access::Read, 33), ("b", Access::Read, 33), ("result", Access::ReadWrite, 16)].into_iter().map(|(name,access,bytes)| BufferArgument {name:name.into(),kind:"buffer".into(),access,bytes,alignment:1}).collect() }
}
fn source(name: &str) -> String { format!(r#"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
declare i32 @{name}(ptr, ptr, i64)
define void @kernel(ptr %a, ptr %b, ptr %out) {{
 %n = load i64, ptr %out, align 1
 %value = call i32 @{name}(ptr %a, ptr %b, i64 %n)
 %p = getelementptr i8, ptr %out, i64 8
 store i32 %value, ptr %p, align 1
 ret void
}}
"#) }
#[test]
fn comparisons_are_defined_without_mutating_input() {
 let context=Context::create();
 for name in ["memcmp","bcmp"] {
 let module=parse_ir(&context,source(name).as_bytes(),"compare").unwrap();
 let original=module.print_to_string().to_string();
 let (air,_) = legalize(&module,&interface()).unwrap();
 air.verify().unwrap();
 assert!(!air.print_to_string().to_string().contains(&format!("@{name}(")));
 assert_eq!(module.print_to_string().to_string(),original);
 }
}
#[test]
fn incompatible_comparison_declaration_is_rejected() {
 let context=Context::create();
 let text=source("memcmp").replace("i32 @memcmp", "i64 @memcmp").replace("store i32 %value", "store i64 %value");
 let module=parse_ir(&context,text.as_bytes(),"bad").unwrap();
 assert!(legalize(&module,&interface()).unwrap_err().contains("invalid memcmp declaration"));
}
#[cfg(target_os="macos")]
#[test]
#[ignore="requires Apple GPU and pinned llvm-downgrade"]
fn comparison_order_lengths_and_unsigned_bytes_match_gpu() {
 use llvm_metal_runtime::{Buffer,Kernel};
 let context=Context::create();
 for name in ["memcmp","bcmp"] {
 let module=parse_ir(&context,source(name).as_bytes(),"compare").unwrap();
 let artifact=llvm_metal_compiler::compile::compile(&module,&interface()).unwrap();
 let dir=tempfile::tempdir().unwrap();
 let path=dir.path().join("kernel.metallib");
 std::fs::write(&path,artifact.metallib).unwrap();
 let kernel=Kernel::load(&path,&artifact.bindings).unwrap();
 for length in [0usize,1,7,16,32,33] {
 for index in [0usize,6,15,31,32] {
 for (left,right) in [(0u8,255u8),(128,127),(255,0),(42,42)] {
 let mut buffers:Vec<_>=[33,33,16].into_iter().map(|n|Buffer { bytes:vec![0xa5;n+512],offset:256 }).collect();
 buffers[0].bytes[256+index]=left;
 buffers[1].bytes[256+index]=right;
 buffers[2].bytes[256..264].copy_from_slice(&(length as u64).to_le_bytes());
 let mut expected:Vec<_>=buffers.iter().map(|b|b.bytes.clone()).collect();
 let value=if index<length {i32::from(left)-i32::from(right)} else {0};
 expected[2][264..268].copy_from_slice(&value.to_le_bytes());
 // SAFETY: reviewed byte-only comparison, disjoint guarded input ranges of at least n bytes.
 unsafe {kernel.run(&mut buffers,1,1).unwrap();}
 for (actual,expected) in buffers.iter().zip(expected) {assert_eq!(actual.bytes,expected,"{name}: {length}/{index}/{left}/{right}");}
 }
 }
 }
 }
}
