; A reference to a value that holds no data: Rust allocates nothing for it and
; still passes its address.
define internal i64 @method(ptr %self, i64 %value) noinline {
  %twice = shl i64 %value, 1
  ret i64 %twice
}

define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %handle = alloca [0 x i8], align 1
  %value = load i64, ptr addrspace(1) %input, align 8
  %result = call i64 @method(ptr %handle, i64 %value)
  store i64 %result, ptr addrspace(1) %output, align 8
  ret void
}
