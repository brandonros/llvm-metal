; What rustc's modules carry: a position-independence level, which changes how
; a compiler addresses globals. So the probe reads one.
@table = internal addrspace(2) constant [4 x i64] [i64 3, i64 5, i64 7, i64 11], align 8

define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %index = load i64, ptr addrspace(1) %input, align 8
  %element = getelementptr [4 x i64], ptr addrspace(2) @table, i64 0, i64 %index
  %value = load i64, ptr addrspace(2) %element, align 8
  store i64 %value, ptr addrspace(1) %output, align 8
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
