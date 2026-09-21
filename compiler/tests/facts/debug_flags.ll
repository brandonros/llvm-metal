; What is left of a rustc module once its debug info is stripped: the two
; flags that announce debug info, and a call between functions.
define internal i64 @twice(i64 %value) {
  %doubled = shl i64 %value, 1
  ret i64 %doubled
}

define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %value = load i64, ptr addrspace(1) %input, align 8
  %result = call i64 @twice(i64 %value)
  store i64 %result, ptr addrspace(1) %output, align 8
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
