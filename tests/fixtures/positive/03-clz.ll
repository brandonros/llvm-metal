; Contract: count leading zero bits of a; zero returns 64. b is unused.
declare i64 @llvm.ctlz.i64(i64, i1 immarg)
define i64 @fixture(i64 %a, i64 %b) {
entry:
  %count = call i64 @llvm.ctlz.i64(i64 %a, i1 false)
  ret i64 %count
}
