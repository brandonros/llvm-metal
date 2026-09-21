; An intrinsic declared as LLVM 22 declares it. `captures` did not exist when
; the bitcode encoding Apple reads was defined.
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %from = alloca i64, align 8
  %to = alloca i64, align 8
  %value = load i64, ptr addrspace(1) %input, align 8
  store i64 %value, ptr %from, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %to, ptr align 8 %from, i64 8, i1 false)
  %copied = load i64, ptr %to, align 8
  store i64 %copied, ptr addrspace(1) %output, align 8
  ret void
}
