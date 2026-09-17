; Input-only fixture for the proposed device-operation boundary.
; Future execution contract: write each valid invocation's linear index to out.
; This declaration must be lowered before execution; never JIT this input.
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm_metal.linear_thread_index()
define void @fixture(ptr %out, i32 %count) {
entry:
  %index = call i32 @llvm_metal.linear_thread_index()
  %valid = icmp ult i32 %index, %count
  br i1 %valid, label %write, label %done
write:
  %wide = zext i32 %index to i64
  %element = getelementptr i32, ptr %out, i64 %wide
  store i32 %index, ptr %element, align 4
  br label %done
done:
  ret void
}
!fixture.note = !{!0}
!0 = !{!"input verification only; not executable before lowering"}
