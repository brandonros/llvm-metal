; What inlining a function with `noalias` parameters leaves behind: memory
; accesses tagged with a scope whose declaration call is gone.
define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %local = alloca i64, align 8
  %value = load i64, ptr addrspace(1) %input, align 8
  store i64 %value, ptr %local, align 8, !noalias !0
  %kept = load i64, ptr %local, align 8, !noalias !0
  store i64 %kept, ptr addrspace(1) %output, align 8
  ret void
}

!0 = !{!1}
!1 = distinct !{!1, !2, !"callee: argument 0"}
!2 = distinct !{!2, !"callee"}
