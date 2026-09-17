{
  description = "LLVM 21 development environment for llvm-metal";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay/753568957a87312ed599cba5699e67126eded6c0";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  inputs.llvm-downgrade = {
    url = "github:JuliaLLVM/llvm-downgrade/49459235028889ea741f2aee3e7f346d53f68f5d";
    flake = false;
  };

  outputs = { nixpkgs, rust-overlay, llvm-downgrade, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forEachSystem = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forEachSystem (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          llvm = pkgs.llvmPackages_21.llvm;
          downgrade = pkgs.stdenv.mkDerivation {
            pname = "llvm-downgrade";
            version = "4945923";
            src = llvm-downgrade;
            nativeBuildInputs = [ pkgs.cmake pkgs.ninja ];
            buildInputs = [ llvm pkgs.libffi ];
            cmakeFlags = [
              "-DLLVM_DIR=${llvm.dev}/lib/cmake/llvm"
              "-DLLVMDG_LINK_DYLIB=ON"
              "-DLLVMDG_BUILD_TESTS=OFF"
            ];
          };
          # Official stable distribution with NVPTX libraries and LLVM 21.1.8.
          # Keep this producer separate from the newer compiler development Rust.
          fixtureRust = pkgs.rust-bin.stable."1.93.0".minimal.override {
            targets = [ "nvptx64-nvidia-cuda" ];
          };
        in {
          default = pkgs.mkShell {
            packages = [ pkgs.cargo pkgs.rustc pkgs.rustfmt pkgs.clippy llvm downgrade ];
            buildInputs = [ pkgs.libffi ];
            LLVM_SYS_211_PREFIX = "${llvm.dev}";
          };
          rust-fixtures = pkgs.mkShell {
            packages = [ fixtureRust llvm downgrade pkgs.python3 ];
            buildInputs = [ pkgs.libffi ];
            LLVM_SYS_211_PREFIX = "${llvm.dev}";
          };
        });
    };
}
