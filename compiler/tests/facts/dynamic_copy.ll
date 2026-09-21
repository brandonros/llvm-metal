; `copy_from_slice` and `fill`: a copy and a fill whose length is only known
; when the kernel runs.
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %from = alloca [16 x i8], align 8
  %to = alloca [16 x i8], align 8
  %length = load i64, ptr addrspace(1) %input, align 8
  store i64 %length, ptr %from, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %to, i8 0, i64 %length, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %to, ptr align 8 %from, i64 %length, i1 false)
  %copied = load i64, ptr %to, align 8
  store i64 %copied, ptr addrspace(1) %output, align 8
  ret void
}
