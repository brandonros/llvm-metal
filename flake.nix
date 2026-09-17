{
  description = "LLVM 21 development environment for llvm-metal";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay/753568957a87312ed599cba5699e67126eded6c0";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";

  outputs = { nixpkgs, rust-overlay, ... }:
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
          # Official stable distribution with NVPTX libraries and LLVM 21.1.8.
          # Keep this producer separate from the newer compiler development Rust.
          fixtureRust = pkgs.rust-bin.stable."1.93.0".minimal.override {
            targets = [ "nvptx64-nvidia-cuda" ];
          };
        in {
          default = pkgs.mkShell {
            packages = [ pkgs.cargo pkgs.rustc pkgs.rustfmt pkgs.clippy llvm ];
            buildInputs = [ pkgs.libffi ];
            LLVM_SYS_211_PREFIX = "${llvm.dev}";
          };
          rust-fixtures = pkgs.mkShell {
            packages = [ fixtureRust llvm pkgs.python3 ];
            buildInputs = [ pkgs.libffi ];
            LLVM_SYS_211_PREFIX = "${llvm.dev}";
          };
        });
    };
}
