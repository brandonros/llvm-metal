; Contract: rotate a left by (b mod 64), including b = 0 and 64.
declare i64 @llvm.fshl.i64(i64, i64, i64)
define i64 @fixture(i64 %a, i64 %b) {
entry:
  %rotated = call i64 @llvm.fshl.i64(i64 %a, i64 %a, i64 %b)
  ret i64 %rotated
}
