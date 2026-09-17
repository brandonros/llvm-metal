; Contract: SHA-256 small sigma0 of low 32 bits of a; b is unused.
; sigma0(x) = ROTR7(x) xor ROTR18(x) xor SHR3(x).
declare i32 @llvm.fshr.i32(i32, i32, i32)
define i64 @fixture(i64 %a, i64 %b) {
entry:
  %x = trunc i64 %a to i32
  %r7 = call i32 @llvm.fshr.i32(i32 %x, i32 %x, i32 7)
  %r18 = call i32 @llvm.fshr.i32(i32 %x, i32 %x, i32 18)
  %s3 = lshr i32 %x, 3
  %mixed = xor i32 %r7, %r18
  %sigma = xor i32 %mixed, %s3
  %result = zext i32 %sigma to i64
  ret i64 %result
}
