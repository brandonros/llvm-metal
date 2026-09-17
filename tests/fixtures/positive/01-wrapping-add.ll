; Contract: fixture(a, b) = (a + b) mod 2^64. No nsw/nuw promises.
define i64 @fixture(i64 %a, i64 %b) {
entry:
  %sum = add i64 %a, %b
  ret i64 %sum
}
