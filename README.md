# llvm-metal

Compile Rust GPU kernels to Metal libraries for Apple GPUs.

This is a GPU dialect of Rust, not a porting tool: kernels are written for the
GPU, and the compiler's job is to make that code correct and to say precisely,
at build time, what is not allowed. The same kernel source also runs on the
host, and that run is the reference every GPU test is compared against.

```rust
use llvm_metal_kernel::{In, Out, Thread, kernel};

kernel! {
    pub fn double(thread: Thread, input: In<[u32], 0>, output: Out<[u32], 1>) {
        let index = thread.index() as usize;
        let Some(value) = input.get(index) else { return };
        output.write(index, value * 2);
    }
}
```

## Layout

- `kernel/`: what a kernel author imports (`Thread`, `In`, `Out`, `Plain`,
  `kernel!`), and the host reference runner.
- `compiler/`: cargo, link, `select`, `verify`, `lower`, `emit`. `verify`'s
  `Rule` doc comments are the specification. No optimization pass runs here.
- `metallib/`: the Metal library container.
- `runtime/`: load a pipeline, own typed buffers, launch.
- `examples/`: kernels written for the GPU, each with a runner, a differential
  GPU test and a benchmark: `shallenge` (SHA-256 search), `rsa` (RSA-2048
  signing), `p256` (ECDSA P-256 signing). The two signers are toys: read the
  warnings at the top of their kernels.

## Use

Everything runs in the development shell, on a Mac with an Apple GPU:

```
nix develop --command cargo test                     # host tests
nix develop --command cargo test -- --ignored --test-threads=1   # GPU tests
nix develop --command cargo run --release -p shallenge
```

Read `AGENTS.md` before changing anything. The first compiler, a different
design, is kept at the tag `v1`.
