; Input-only fixture: a device-space atomic increment returning the old value.
; LLVM accepts named scopes; this alone does not establish Metal semantics.
target triple = "nvptx64-nvidia-cuda"
define i32 @fixture(ptr addrspace(1) %counter, i32 %increment) {
entry:
  %old = atomicrmw add ptr addrspace(1) %counter, i32 %increment syncscope("device") monotonic, align 4
  ret i32 %old
}
