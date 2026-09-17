; Contract: copy exactly count bytes; source/destination do not overlap.
; Pointers reference valid byte arrays, including when count = 0.
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
define void @fixture(ptr %destination, ptr %source, i64 %count) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %destination, ptr align 1 %source, i64 %count, i1 false)
  ret void
}
