; How Rust returns a small struct such as a `Range<usize>`: two words by value.
define internal { i64, i64 } @pair(i64 %first, i64 %second) noinline {
  %one = insertvalue { i64, i64 } poison, i64 %first, 0
  %both = insertvalue { i64, i64 } %one, i64 %second, 1
  ret { i64, i64 } %both
}

define void @probe(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %value = load i64, ptr addrspace(1) %input, align 8
  %pair = call { i64, i64 } @pair(i64 %value, i64 31)
  %first = extractvalue { i64, i64 } %pair, 0
  %second = extractvalue { i64, i64 } %pair, 1
  %sum = add i64 %first, %second
  store i64 %sum, ptr addrspace(1) %output, align 8
  ret void
}
