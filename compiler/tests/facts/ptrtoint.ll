; The length of a slice as Rust computes it: the difference of two addresses.
define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %local = alloca [16 x i8], align 8
  %count = load i64, ptr addrspace(1) %input, align 8
  %end = getelementptr i8, ptr %local, i64 %count
  %first = ptrtoint ptr %local to i64
  %last = ptrtoint ptr %end to i64
  %length = sub i64 %last, %first
  store i64 %length, ptr addrspace(1) %output, align 8
  ret void
}
