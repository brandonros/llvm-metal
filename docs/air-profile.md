# Initial AIR profile

Input is verified LLVM 21 IR/bitcode with the stock
`nvptx64-nvidia-cuda` triple and explicit data layout. This triple selects the
Rust producer's platform model; no NVIDIA instructions or PTX are emitted.
The interface describes one C entry, one to 31 buffers, byte sizes, alignments
and read/write access. Entries return void and take generic pointers only.
Buffers bind in argument order to Metal indices 0 onward. `aliasing` documents
the producer contract; it is not a memory-safety proof or a generated noalias fact.

Supported input includes 1/8/16/32/64-bit integer operations, branches, PHIs,
fixed vectors/arrays/structs with matching source/AIR layouts, stack allocations,
loads/stores, pointer helpers and constant integer/array globals. Defined C/fastcc
helpers must inline completely. Surviving pointer address-space conversions are
rejected. A type/opcode passing validation does not establish support for every
possible combination; library/pipeline creation remains a separate check.

The external-operation whitelist includes byte swaps, funnel shifts, unsigned
three-way comparison, nonvolatile memcpy/memset, and removable lifetime/alias
hints. Unknown externals, inline assembly, indirect calls, mutable globals,
pointer/integer casts, native LLVM atomics, volatile memory, floating point,
i128 and NVVM metadata are rejected. No allocator, panic handler, libdevice,
barrier, threadgroup memory or general GPU runtime is supplied.

Two explicit compiler operations are currently defined:

| Declaration | Contract |
|---|---|
| `i32 @llvm_metal.linear_thread_index()` | Linear 1D grid index, appended AIR builtin argument |
| `i32 @llvm_metal.atomic_add_device_u32(ptr, i32)` | Aligned device-buffer u32 fetch-add; returns old value; wrapping arithmetic; relaxed ordering, device scope |

`dispatch: "single"` requires `invocations: 1`. `dispatch: "grid1d"` requires
`invocations: null` and use of the index operation. Bounds checks belong to the
kernel. The atomic pointer must refer to suitably aligned device storage, never
a stack allocation. Source Rust wrappers declare these operations explicitly;
stock Rust atomics/NVPTX intrinsics are not automatically substituted.

Legalization maps buffer pointers to AIR address space 1 and constants to 2,
using LLVM's address-space inference through a small C++ bridge. It removes
source PIC/PIE code-generation flags and scoped alias/lifetime hints. Other
language semantics are preserved; no fixture name triggers a special lowering.
The compiler emits AIR 2.4/Metal 3.0 metadata and resource limits. The pinned
LLVM-21-compatible llvm-downgrade writes bitcode version 14; native LLVM verifies
that result before the single-function macOS metallib container is constructed.
Apple's runtime is the compatibility oracle, not LLVM verification alone.

The runtime loads a library, creates a pipeline, allocates shared buffers and
dispatches synchronously through objc2-metal. It checks declared minimum sizes,
offset alignment and dispatch limits. Its dispatch API is unsafe: callers must
provide a trusted kernel and satisfy all dynamic bounds, alias and race rules.
The test harness uses separate guarded allocations and reads outputs only after
command completion. Winner publication is checked at this host synchronization
point; this does not demonstrate release/acquire publication to concurrent GPU
consumers. Runtime reflection, asynchronous queues, reusable GPU allocations,
multi-kernel libraries and application integration remain future work.
