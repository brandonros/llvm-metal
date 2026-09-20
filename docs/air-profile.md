# Initial AIR profile

Input is verified LLVM 21 IR/bitcode with the stock
`nvptx64-nvidia-cuda` triple and explicit data layout. This triple selects the
Rust producer's platform model; no NVIDIA instructions or PTX are emitted.
The interface describes one C entry, one to 31 buffers, byte sizes, alignments
and read/write access. Entries return void and take generic pointers only.
Buffers bind in argument order to Metal indices 0 onward. `aliasing` documents
the producer contract; it is not a memory-safety proof or a generated noalias fact.

## Typed Rust descriptors

`llvm-metal-kernel` is an allocation-free `no_std` producer crate. `record!`
declares a `repr(C)` record and derives its target size, alignment, field names,
offsets and recursive field types. Padding is rejected. Supported transferable
types are fixed-width signed/unsigned 8/16/32/64-bit integers, nonempty arrays,
and records of those types. Pointers, references, `usize`, bool and Rust enums
are not transferable records. Manual `DeviceLayout` implementations are unsafe.

`kernel!` generates a C pointer entry, a named `Arguments<T>` host container,
and a descriptor from one argument declaration. Arguments declare `Read`,
`Write` or `ReadWrite` access and `Fixed` record or `Slice` element shape.
`DESCRIPTOR` evaluates independently in host and device compilations. On NVPTX,
the descriptor is exported as `__llvm_metal_descriptor_<entry>` and rooted in
LLVM's compiler-used list. These annotations are a retention mechanism, not a
promise that arbitrary subsequent optimizer invocations preserve metadata.

Extract immediately after bitcode linking, before internalization or DCE:

```sh
llvm-metalc extract linked.bc --output extracted
# Select an entry's descriptor from extracted/descriptors.json. Optimize
# extracted/stripped.bc, preserving the selected entry, into kernel.bc.
llvm-metalc compile kernel.bc --descriptor selected.json --output bundle
```

Extraction verifies every descriptor's symbol/entry association and C pointer
signature. It removes descriptor globals and their retention roots while
preserving unrelated roots, and rejects executable references to descriptors.
The producer must keep the extracted descriptor associated with that exact
module and record input/output hashes in its artifact provenance. Small inputs
can use `compile input.bc --entry name --output bundle` directly. Existing
`--interface` JSON remains supported for explicit LLVM fixtures and other users.

The descriptor encoding is canonical, at most 16 KiB: little-endian u32 magic
`0x00444d4c`, version 1, length-prefixed UTF-8 entry, u8 dispatch (single=0,
grid1d=1), u8 endianness (little=0), u32 argument count, then argument records.
Each contains a name, u8 access (read=0, write=1, read/write=2), u8 shape
(fixed=0, slice=1), and a recursive layout. A layout is u32 size, u32 alignment,
u8 kind (unsigned=0, signed=1, array=2, record=3). Arrays add a u32 count and
element layout; records add a u32 field count and fields containing a name,
u32 offset and layout. All strings use a u32 byte length; trailing bytes,
unknown versions/tags, duplicate names, padding and inconsistent layouts fail.

Generated Metal bindings retain the full descriptor. Call
`validate_host_descriptor` against the independently compiled host declaration
before pipeline creation. It compares field identities/layouts as well as the
Metal buffer mapping, so equal total byte sizes do not hide reordered fields.
`validate_lengths` checks fixed records and dynamic element counts with overflow
protection. Empty slices require backed storage for one element. The launcher
must still validate where those counts come from, dispatch bounds, disjoint
storage, initialization and ownership. Descriptors do not prove that arbitrary
kernel bodies obey their declared accesses.

## LLVM operations

Supported input includes 1/8/16/32/64-bit integer operations, branches, PHIs,
and existing unreachable terminators (source undefined behavior),
fixed vectors/arrays/structs with matching source/AIR layouts, stack allocations,
loads/stores, pointer helpers and constant integer/array/struct globals. Defined C/fastcc
helpers use selective inlining by default, as described below. Surviving pointer address-space conversions are rejected. A type/opcode passing validation does not establish support for every
possible combination; library/pipeline creation remains a separate check.

### Retained helpers and inlining

`compile --inlining selective` retains internal, nonrecursive helpers with void
or i8/i16/i32/i64 returns and scalar or pointer parameters. It retains eligible
functions with at least 32 LLVM instructions or an explicit `noinline` request;
smaller wrappers, constant-state initializers, and unsupported interfaces still inline. `retain-scalar` is a
narrow diagnostic policy without pointer parameters. `selective` is the default in both library APIs and the CLI; `--inlining all`
opts into full inlining. Internal C/fastcc definitions and all direct call sites are
normalized together to C for AIR. Scalar-i128 interfaces and helpers depending
on the entry's thread index continue to require inlining. Pointer/aggregate
returns are not yet retained. Preparation also inlines helpers that transitively
use unsupported runtime operations, indirect calls, or pointer-containing memory
interfaces, allowing LLVM to eliminate
unreachable paths using caller facts. Escaping callbacks may exist in producer
bitcode but must disappear before final legalization; live unsupported paths
still fail. Helpers combining loop-carried pointers with null-sentinel operations
also require inlining during final legalization: retaining this iterator shape caused a P-256 matcher
mismatch on Apple M5. Straight-line nullable pointer joins remain supported.
Copy/stack ABI parameters (`byval`, `byref`,
`inalloca`, `preallocated`, and nest/Swift context parameters) also require
inlining; ordinary pointer parameters and tested `sret` parameters can remain.

Consumers must preserve boundaries before their own optimization pipeline:
internalize to the selected entry and remove dead functions, then run
`llvm-metalc prepare selected.bc --entry name --output prepared.bc --inlining selective`.
Optimize `prepared.bc` without globally forcing `alwaysinline`, and pass the
same policy to `compile`. Producer fallback recursion can be eliminated by
ordinary LLVM inlining/optimization; any recursion surviving into legalization
is rejected. Function addresses must not escape, calls must be direct with
matching conventions, and operand bundles and `musttail` calls are rejected.

Pointer arguments specialize helpers by private/device/constant address-space
signature, including nested calls and constant-table offsets. Private joins
require proven private roots; loaded/unknown pointer flows fail. Specialization
is bounded to 4096 instances. Legalization and pointer/metadata checks run in
all surviving definitions. Private volatile accesses and zeroization are
preserved. LLVM 21 parameter facts missing from the legacy writer's encoding
are dropped explicitly; ABI attributes such as `sret` remain intact.

The existing LLVM writer and metallib container carry the helper definitions.
Retained LLVM calls do not promise how Apple's backend will optimize them.
Measure AIR size, first/repeated library and pipeline creation, GPU execution,
and correctness independently when tuning the policy or choosing full inlining.

The external-operation whitelist includes byte swaps, funnel shifts, unsigned
three-way comparison, scalar integer absolute value, leading/trailing-zero counts and signed/unsigned min/max,
nonvolatile memcpy/memset and scalar C `memcmp`/`bcmp`,
and removable assume/lifetime/alias hints. Unknown externals, inline assembly, indirect calls, mutable globals,
pointer/integer casts, native LLVM atomics, volatile device memory, floating point,
and NVVM metadata are rejected. Volatile accesses rooted in private allocas
are preserved (including internal helper arguments when every direct caller
provably supplies private storage), supporting subtle's local optimization barrier and the consumer's
zeroize stores. Constant-size volatile memcpy up to 4096 bytes into private
allocas from private or defined immutable constant storage expands to volatile
byte loads/stores. Volatile constant reads retain their constant address space. This supplies no
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
Scalar i8/i16/i32/i64 `uadd.sat` and `usub.sat`, which stock Rust emits for
saturating arithmetic and iterator adapters such as `take`, lower to a compare
and select. Signed saturating forms and vector/wide inputs are rejected.
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
constant and variable left/logical-right/arithmetic-right shifts (0..127), truncation to at
most 64 bits, loop/join PHIs, and nonvolatile/non-atomic stores into
pairs of i64 values. Products use 32-bit partial products and explicit carries.
The pass preserves defined wrapping results and drops optional no-wrap flags.
Private/defined-constant wide loads and private volatile stores split into
i64 accesses; device wide loads and atomic wide accesses remain unsupported. Private
scalar i128 allocas with constant element counts use `[2 x i64]` storage, retaining
the explicit alignment, count and 16-byte GEP stride. Private snapshots of
one-dimensional `[1..256 x i128]` arrays support scalar extracts, preserving
the complete load at its original position, including unused volatile elements.
Their storage retains array stride and alignment. Escaping pointers, unknown
helper calls, dynamic counts and other aggregate wide operations remain unsupported.
Variable counts at least 128 preserve LLVM poison semantics. Local scalar-i128
helper interfaces are inlined before splitting; escaping, recursive and external
wide interfaces are refused. Wide division and vectors remain unsupported. Unsupported producers/consumers are
rejected; input modules are never mutated. Generic i128 support is not claimed.

LLVM optimization can introduce i24/i40/i48/i56 scalar values. A separate pass
promotes these to i32/i64 operations, masking results and sign-extending only for
signed operations. Memory accesses use exactly 3/5/6/7 bytes, including unaligned
spans; they never read or overwrite the next byte. The pass runs before input
validation and after optimization. Odd-width storage types, globals, function
ABIs, vectors, atomics and volatile operations remain outside this profile.
CPU LLVM execution and guarded GPU tests cover arithmetic, comparisons, shifts,
PHIs and conversions. Apple does not natively support the tested i24 operation.
Register-only i2 through i7 values also promote to i32 with the same masking and
signed-operation rules. Loads, stores, aggregate storage and function ABIs using
these sub-byte types are rejected. Exhaustive CPU comparisons against the original
LLVM and Rust cover every operand pair at each width, including the i6 SEC1 tag
bitset. Guarded GPU tests cover 1,004 operand pairs across these widths.

Dynamic nonvolatile `memcpy` and `memset` lower after optimization to two
stride-two byte loops and an optional final byte. Zero-length operations access
no memory. Narrow unsigned lengths widen to i64 before index arithmetic, avoiding
one-bit shifts and signed GEP extension. Constant-size intrinsics retain LLVM's
normal lowering; dynamic volatile copies and fills remain unsupported. The loop
form survives LLVM O3 and loop-idiom recognition without relying on the additional
`no-builtin-memcpy`/`no-builtin-memset` attributes. Guarded CPU and GPU tests cover
private/device destinations, private/device/constant copy sources, unaligned spans,
zero lengths and exact copy/fill bounds. This avoids the observed runtime-zero
`memcpy` stall; a corresponding `memset` driver failure has not been established.

`memcmp` and `bcmp` use unsigned byte comparisons and stop at the first mismatch;
zero-length comparisons read no memory. Unsupported declarations are refused.

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
using LLVM's address-space inference. It resets source PIC/PIE code-generation
levels to zero and removes scoped alias/lifetime hints. Function
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
A final normalization puts constant-expression PHI operands on incoming edges:
the pinned legacy writer otherwise materializes some of these before the PHI,
violating SSA grouping. Both instruction and constant-expression address-space
casts must be resolved before serialization. Apple's runtime remains the
compatibility oracle; LLVM verification alone does not establish compatibility.

The runtime loads a library, creates a pipeline, allocates shared buffers and
dispatches synchronously through objc2-metal. It checks declared minimum sizes,
offset alignment and dispatch limits. Its dispatch API is unsafe: callers must
provide a trusted kernel and satisfy all dynamic bounds, alias and race rules.
The test harness uses separate guarded allocations and reads outputs only after
command completion. Winner publication is checked at this host synchronization
point; this does not demonstrate release/acquire publication to concurrent GPU
consumers. PreparedKernel reuses fixed-size shared allocations across synchronous launches and
returns upload/download, wall and optional GPU command-buffer timings.
`reconfigure` replaces storage only when buffer lengths/offsets change and keeps
the compiled pipeline. Failed validation/allocation preserves the old configuration.
`clear` erases retained shared storage after synchronous use; resources are also
erased before replacement or release. Callers own cleanup of host mirrors. Library
and pipeline creation have separate timings. Runtime reflection, asynchronous
queues and multi-kernel libraries remain future work.
