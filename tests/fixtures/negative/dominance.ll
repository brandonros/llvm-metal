; Parses but violates SSA dominance: %value is unavailable on the right path.
define i64 @fixture(i1 %condition) {
entry:
  br i1 %condition, label %left, label %right
left:
  %value = add i64 1, 2
  br label %merge
right:
  br label %merge
merge:
  ret i64 %value
}
