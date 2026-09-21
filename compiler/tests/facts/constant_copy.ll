; A table in constant memory copied to thread memory in one call, then indexed
; through an ordinary pointer.
@table = internal addrspace(2) constant [4 x i64] [i64 3, i64 5, i64 7, i64 11], align 8

declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)

define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %copy = alloca [4 x i64], align 8
  call void @llvm.memcpy.p0.p2.i64(ptr align 8 %copy, ptr addrspace(2) align 8 @table, i64 32, i1 false)
  %index = load i64, ptr addrspace(1) %input, align 8
  %element = getelementptr [4 x i64], ptr %copy, i64 0, i64 %index
  %value = load i64, ptr %element, align 8
  store i64 %value, ptr addrspace(1) %output, align 8
  ret void
}
