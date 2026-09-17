; Independent AIR output-path reference. Single invocation, two 4-byte buffers.
; Metadata follows the documented-in-source GPUCompiler.jl Metal backend.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx13.0.0"
define void @add42(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %x = load i32, ptr addrspace(1) %input, align 4
  %y = add i32 %x, 42
  store i32 %y, ptr addrspace(1) %output, align 4
  ret void
}
!air.version = !{!0}
!air.language_version = !{!1}
!air.kernel = !{!2}
!0 = !{i32 2, i32 4, i32 0}
!1 = !{!"Metal", i32 3, i32 0, i32 0}
!2 = !{ptr @add42, !3, !4}
!3 = !{}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
