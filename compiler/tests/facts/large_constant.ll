; A table too large to become immediates: Metal must give it storage.
@table = internal addrspace(2) constant [64 x i64] [i64 0, i64 3, i64 6, i64 9, i64 12, i64 15, i64 18, i64 21, i64 24, i64 27, i64 30, i64 33, i64 36, i64 39, i64 42, i64 45, i64 48, i64 51, i64 54, i64 57, i64 60, i64 63, i64 66, i64 69, i64 72, i64 75, i64 78, i64 81, i64 84, i64 87, i64 90, i64 93, i64 96, i64 99, i64 102, i64 105, i64 108, i64 111, i64 114, i64 117, i64 120, i64 123, i64 126, i64 129, i64 132, i64 135, i64 138, i64 141, i64 144, i64 147, i64 150, i64 153, i64 156, i64 159, i64 162, i64 165, i64 168, i64 171, i64 174, i64 177, i64 180, i64 183, i64 186, i64 189], align 8

declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)

define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %copy = alloca [64 x i64], align 8
  call void @llvm.memcpy.p0.p2.i64(ptr align 8 %copy, ptr addrspace(2) align 8 @table, i64 512, i1 false)
  %index = load i64, ptr addrspace(1) %input, align 8
  %element = getelementptr [64 x i64], ptr %copy, i64 0, i64 %index
  %value = load i64, ptr %element, align 8
  store i64 %value, ptr addrspace(1) %output, align 8
  ret void
}
