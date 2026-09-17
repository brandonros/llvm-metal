; Contract: unsigned absolute difference, covering both paths and equality.
define i64 @fixture(i64 %a, i64 %b) {
entry:
  %greater = icmp ugt i64 %a, %b
  br i1 %greater, label %left, label %right
left:
  %ab = sub i64 %a, %b
  br label %merge
right:
  %ba = sub i64 %b, %a
  br label %merge
merge:
  %result = phi i64 [ %ab, %left ], [ %ba, %right ]
  ret i64 %result
}
