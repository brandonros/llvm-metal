; Contract: write value to buffer[index], leaving every other element intact.
; Precondition: buffer contains at least index + 1 aligned i32 elements.
define internal void @write(ptr %base, i64 %index, i32 %value) noinline {
entry:
  %element = getelementptr i32, ptr %base, i64 %index
  store i32 %value, ptr %element, align 4
  ret void
}
define void @fixture(ptr %buffer, i64 %index, i32 %value) {
entry:
  call void @write(ptr %buffer, i64 %index, i32 %value)
  ret void
}
