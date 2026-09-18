use inkwell::context::Context;
use llvm_metal_abi::{Access, BufferArgument, Dispatch, KernelInterface};
use llvm_metal_compiler::{air::legalize, parse_ir};
fn interface() -> KernelInterface {
    KernelInterface {schema:1,entry:"kernel".into(),calling_convention:"C".into(),invocations:Some(1),dispatch:Dispatch::Single,aliasing:"disjoint".into(),arguments:vec![BufferArgument{name:"data".into(),kind:"buffer".into(),access:Access::ReadWrite,bytes:48,alignment:8}]}
}
fn source(op:&str)->String {format!(r#"
target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-i128:128"
define void @kernel(ptr %p) {{
 %lo = load i64, ptr %p, align 8
 %p1 = getelementptr i8, ptr %p, i64 8
 %hi = load i64, ptr %p1, align 8
 %p2 = getelementptr i8, ptr %p, i64 16
 %countlo = load i64, ptr %p2, align 8
 %p3 = getelementptr i8, ptr %p, i64 24
 %counthi = load i64, ptr %p3, align 8
 %l = zext i64 %lo to i128
 %h = zext i64 %hi to i128
 %hs = shl i128 %h, 64
 %value = or i128 %l, %hs
 %nl = zext i64 %countlo to i128
 %nh = zext i64 %counthi to i128
 %nhs = shl i128 %nh, 64
 %n = or i128 %nl, %nhs
 %result = {op} i128 %value, %n
 %out = getelementptr i8, ptr %p, i64 32
 store i128 %result, ptr %out, align 8
 ret void
}}
"#)}
#[test]
fn dynamic_wide_shifts_legalize_without_mutating_input() {
 let context=Context::create();
 for op in ["shl","lshr","ashr"] {
 let module=parse_ir(&context,source(op).as_bytes(),"wide-shift").unwrap();
 let original=module.print_to_string().to_string();
 let (air,_) = legalize(&module,&interface()).unwrap();
 air.verify().unwrap();
 assert!(!air.print_to_string().to_string().contains("i128"));
 assert_eq!(module.print_to_string().to_string(),original);
 }
}
#[cfg(target_os="macos")]
#[test]
#[ignore="requires Apple GPU and pinned llvm-downgrade"]
fn every_valid_wide_shift_matches_gpu() {
 use llvm_metal_runtime::{Buffer,Kernel};
 let context=Context::create();
 for op in ["shl","lshr","ashr"] {
 let module=parse_ir(&context,source(op).as_bytes(),"wide-shift").unwrap();
 let artifact=llvm_metal_compiler::compile::compile(&module,&interface()).unwrap();
 let dir=tempfile::tempdir().unwrap();
 let path=dir.path().join("kernel.metallib");
 std::fs::write(&path,artifact.metallib).unwrap();
 let kernel=Kernel::load(&path,&artifact.bindings).unwrap();
 for value in [0u128,1,u64::MAX as u128,1<<127,u128::MAX,0x123456789abcdef0fedcba9876543210] {
 for n in 0..128u32 {
 let mut bytes=vec![0xa5;560];
 bytes[256..272].copy_from_slice(&value.to_le_bytes());
 bytes[272..288].copy_from_slice(&(n as u128).to_le_bytes());
 let result=match op {"shl"=>value<<n,"lshr"=>value>>n,_=>((value as i128)>>n) as u128};
 let mut expected=bytes.clone();
 expected[288..304].copy_from_slice(&result.to_le_bytes());
 let mut buffers=[Buffer{bytes,offset:256}];
 // SAFETY: known integer-only kernel and initialized guarded 48-byte record.
 unsafe {kernel.run(&mut buffers,1,1).unwrap();}
 assert_eq!(buffers[0].bytes,expected,"{op}/{value:x}/{n}");
 }
 }
 }
}
