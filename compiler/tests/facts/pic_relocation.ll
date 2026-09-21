; rustc marks its modules position-independent. With that flag at 2, this
; module kills Apple's compiler service (null dereference in a machine function
; pass; pipeline creation reports XPC_ERROR_CONNECTION_INTERRUPTED; M5, macOS
; 27). It is what `lower` makes of a kernel with an empty body whose constants
; hold one address. `emit` clears the flag, which means nothing on a GPU.

@llvm_metal.constants = internal addrspace(2) constant <{ [18 x i8], [6 x i8], <{ ptr, [16 x i8] }> }> <{ [18 x i8] c"kernel/src/lib.rs\00", [6 x i8] zeroinitializer, <{ ptr, [16 x i8] }> <{ ptr null, [16 x i8] c"\11\00\00\00\00\00\00\00a\00\00\00.\00\00\00" }> }>, align 8

define internal void @kernel.shallenge(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, i32 %4, ptr %5) {
  ret void
}

define void @probe(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, i32 %4) {
  %constants = alloca [48 x i8], align 8
  call void @llvm.memcpy.p0.p2.i64(ptr %constants, ptr addrspace(2) @llvm_metal.constants, i64 48, i1 false)
  %6 = getelementptr i8, ptr %constants, i64 0
  %7 = getelementptr i8, ptr %constants, i64 24
  store ptr %6, ptr %7, align 1
  call void @kernel.shallenge(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, i32 %4, ptr %constants)
  ret void
}

declare void @llvm.memcpy.p0.p2.i64(ptr noalias writeonly captures(none), ptr addrspace(2) noalias readonly captures(none), i64, i1 immarg)

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"PIC Level", i32 2}
