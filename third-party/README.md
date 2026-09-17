# Upstream references

The metallib writer adapts the single-function container layout from
[Metal.jl `src/compiler/library.jl`](https://github.com/JuliaGPU/Metal.jl/blob/main/src/compiler/library.jl).
The AIR argument metadata and data layout follow
[GPUCompiler.jl `src/metal.jl`](https://github.com/JuliaGPU/GPUCompiler.jl/blob/main/src/metal.jl).
Their MIT license notices are retained here.

`flake.lock` pins JuliaLLVM/llvm-downgrade's LLVM 21-compatible revision.
It is built as an external executable with its original license; no legacy
bitcode serializer is copied into this repository.
