{
  description = "LLVM 22 development environment for llvm-metal";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay/1fb104a12a8667045559b2575d6d448ae2fbd99b";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  inputs.llvm-downgrade = {
    url = "github:JuliaLLVM/llvm-downgrade/09c2e50d4526b08f55010f6c091ace35332a78a8";
    flake = false;
  };

  outputs = { nixpkgs, rust-overlay, llvm-downgrade, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
    in {
      devShells = nixpkgs.lib.genAttrs systems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          llvm = pkgs.llvmPackages_22.llvm;
          # Apple's compiler accepts only LLVM 14-encoded bitcode.
          downgrade = pkgs.stdenv.mkDerivation {
            pname = "llvm-downgrade";
            version = "09c2e50";
            src = llvm-downgrade;
            nativeBuildInputs = [ pkgs.cmake pkgs.ninja ];
            buildInputs = [ llvm pkgs.libffi ];
            cmakeFlags = [
              "-DLLVM_DIR=${llvm.dev}/lib/cmake/llvm"
              "-DLLVMDG_LINK_DYLIB=ON"
              "-DLLVMDG_BUILD_TESTS=OFF"
            ];
          };
          # Official stable distribution: its LLVM must match `llvm` above.
          toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        in {
          default = pkgs.mkShell {
            packages = [ toolchain llvm downgrade ];
            buildInputs = [ pkgs.libffi ];
            LLVM_SYS_221_PREFIX = "${llvm.dev}";
          };
        });
    };
}
