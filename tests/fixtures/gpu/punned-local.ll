; A local byte array written a byte at a time and read back as one i64: what
; rustc leaves of `u64::from_be_bytes(slice.try_into())` when the copy loop is
; neither unrolled nor promoted by SROA.
target datalayout = "e-p:64:64-i64:64-i128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @punned(ptr noalias readonly %input, ptr noalias writeonly %output) {
entry:
  %local = alloca [8 x i8], align 8
  store i64 0, ptr %local, align 8
  br label %head

head:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %done = icmp eq i64 %i, 8
  br i1 %done, label %exit, label %body

body:
  %from = getelementptr inbounds i8, ptr %input, i64 %i
  %byte = load i8, ptr %from, align 1
  %to = getelementptr inbounds i8, ptr %local, i64 %i
  store i8 %byte, ptr %to, align 1
  %next = add nuw nsw i64 %i, 1
  br label %head

exit:
  %word = load i64, ptr %local, align 8
  store i64 %word, ptr %output, align 1
  ret void
}
