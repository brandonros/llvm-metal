; Contract: b + sum(0..a), wrapping u64 arithmetic. Tests bound a <= 256.
define i64 @fixture(i64 %a, i64 %b) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %acc = phi i64 [ %b, %entry ], [ %sum, %body ]
  %more = icmp ult i64 %i, %a
  br i1 %more, label %body, label %done
body:
  %sum = add i64 %acc, %i
  %next = add i64 %i, 1
  br label %loop
done:
  ret i64 %acc
}
