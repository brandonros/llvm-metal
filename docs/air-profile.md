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
three-way comparison, scalar integer absolute value, leading/trailing-zero counts and signed/unsigned min/max,
nonvolatile memcpy/memset,
and removable assume/lifetime/alias hints. Unknown externals, inline assembly, indirect calls, mutable globals,
pointer/integer casts, native LLVM atomics, volatile device memory, floating point,
and NVVM metadata are rejected. Volatile accesses rooted in private allocas
are preserved (including internal helper arguments when every direct caller
provably supplies private storage), supporting subtle's local optimization barrier and the consumer's
zeroize stores. Constant-size volatile memcpy up to 256 bytes between private
allocas expands to volatile byte loads/stores. This supplies no
device synchronization or constant-time execution guarantee. No allocator, panic handler, libdevice,
barrier, threadgroup memory or general GPU runtime is supplied.

`llvm.abs` is lowered for scalar i8/i16/i32/i64 using signed comparison,
negation and selection. A false `is_int_min_poison` flag preserves wrapping
INT_MIN; true uses NSW negation to preserve its poison case, following
[LLVM's intrinsic contract](https://llvm.org/docs/LangRef.html#llvm-abs-intrinsic).
Vector and wide absolute values remain unsupported. GPU tests cover every byte
value and wider signed boundaries; poison cases are checked structurally.

Scalar i8/i16/i32/i64 `smin`, `smax`, `umin` and `umax` lower to comparisons
with the corresponding signedness and selection. Vector/wide inputs are rejected.
`freeze` is preserved through LLVM 14 bitcode serialization and accepted by the
tested Apple pipeline; it is not replaced by its possibly poison operand.
`llvm.assume` is discarded as an optimization hint after legalization.
The generic InstCombine passes allow up to four iterations, retaining fixpoint
verification: expanded dalek scalar products do not converge within one iteration.

Scalar i8/i16/i32/i64 `ctlz` and `cttz` use a logarithmic sequence of
integer masks, shifts and selects. Zero returns the input width when defined;
the intrinsic's zero-is-poison flag is preserved. Vectors and i128 counts
are rejected. GPU tests cover zero, boundaries and every input bit.

A narrow scalar i128 pass lowers zero/sign extension from at most 64 bits,
addition/subtraction, multiplication, bitwise AND/OR/XOR, byte swap, comparisons, selection,
constant left/logical-right/arithmetic-right shifts (0..127), truncation to at
most 64 bits, loop/join PHIs, and nonvolatile/non-atomic stores into
pairs of i64 values. Products use 32-bit partial products and explicit carries.
The pass preserves defined wrapping results and drops optional no-wrap flags.
Wide loads, division, dynamic shifts, vectors, and wide
function interfaces remain unsupported. Unsupported producers/consumers are
rejected; input modules are never mutated. Generic i128 support is not claimed.

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
source PIC/PIE code-generation flags and scoped alias/lifetime hints. Function
and call-site noinline hints are removed to inline defined helpers. Other
language semantics are preserved; no fixture name triggers a special lowering.
Nullable pointer PHIs are explicitly typed only when all concrete underlying
objects prove the same device or constant address space. Null/undef inputs are
retained in that space; mixed private/device or device/constant merges remain
unsupported. The AIR profile uses zero-valued null pointers in its supported
spaces, allowing the constant null casts introduced by inference to fold.
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
consumers. PreparedKernel reuses fixed-size shared allocations across synchronous launches and
returns upload/download, wall and optional GPU command-buffer timings. Library
and pipeline creation have separate timings. Runtime reflection, asynchronous
queues and multi-kernel libraries remain future work.
