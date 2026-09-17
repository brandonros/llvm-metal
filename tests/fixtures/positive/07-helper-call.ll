; Contract: (a * 3) xor (b * 3), with wrapping multiplication in a helper.
define internal i64 @triple(i64 %x) noinline {
entry:
  %value = mul i64 %x, 3
  ret i64 %value
}
define i64 @fixture(i64 %a, i64 %b) {
entry:
  %x = call i64 @triple(i64 %a)
  %y = call i64 @triple(i64 %b)
  %result = xor i64 %x, %y
  ret i64 %result
}
