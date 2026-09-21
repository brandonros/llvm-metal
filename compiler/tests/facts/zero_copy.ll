; A copy whose run-time length is zero: `input` holds the length, and `to`
; keeps its 42 when nothing is written.
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %from = alloca [16 x i8], align 8
  %to = alloca [16 x i8], align 8
  %length = load i64, ptr addrspace(1) %input, align 8
  store i64 7, ptr %from, align 8
  store i64 42, ptr %to, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %to, ptr align 8 %from, i64 %length, i1 false)
  %result = load i64, ptr %to, align 8
  store i64 %result, ptr addrspace(1) %output, align 8
  ret void
}
