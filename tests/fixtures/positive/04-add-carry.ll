; Contract: add low 32 bits; return sum in bits 0..31 and carry in bit 32.
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)
define i64 @fixture(i64 %a, i64 %b) {
entry:
  %x = trunc i64 %a to i32
  %y = trunc i64 %b to i32
  %pair = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %x, i32 %y)
  %sum = extractvalue { i32, i1 } %pair, 0
  %carry = extractvalue { i32, i1 } %pair, 1
  %lo = zext i32 %sum to i64
  %carry64 = zext i1 %carry to i64
  %hi = shl i64 %carry64, 32
  %packed = or i64 %hi, %lo
  ret i64 %packed
}
