; An address that must exist as an integer: nothing can fold it away, because
; it leaves the kernel. The local is also written through, so it stays memory.
define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %local = alloca [16 x i8], align 8
  %value = load i64, ptr addrspace(1) %input, align 8
  store volatile i64 %value, ptr %local, align 8
  %address = ptrtoint ptr %local to i64
  %kept = load volatile i64, ptr %local, align 8
  %sum = add i64 %address, %kept
  %result = sub i64 %sum, %address
  store volatile i64 %address, ptr addrspace(1) %output, align 8
  store volatile i64 %result, ptr addrspace(1) %output, align 8
  ret void
}
