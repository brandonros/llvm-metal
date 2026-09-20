{
  description = "LLVM 22 development environment for llvm-metal";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay/1fb104a12a8667045559b2575d6d448ae2fbd99b";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  inputs.llvm-downgrade = {
    url = "github:JuliaLLVM/llvm-downgrade/09c2e50d4526b08f55010f6c091ace35332a78a8";
    flake = false;
  };

  outputs = { self, nixpkgs, rust-overlay, llvm-downgrade, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forEachSystem = nixpkgs.lib.genAttrs systems;
      # Everything that depends on the system, shared by the shells and the package.
      perSystem = forEachSystem (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          llvm = pkgs.llvmPackages_22.llvm;
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
          # Official stable distribution: its LLVM must match `llvm` above, and
          # its NVPTX target produces the test kernels' bitcode.
          toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          # What a consumer needs: the compiler with its LLVM and llvm-downgrade.
          # The consumer supplies only the Rust toolchain that produces bitcode.
          llvm-metalc = pkgs.rustPlatform.buildRustPackage {
            pname = "llvm-metalc";
            version = "0.1.0";
            src = self;
            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [ "-p" "llvm-metal-compiler" "--bin" "llvm-metalc" ];
            doCheck = false;
            nativeBuildInputs = [ pkgs.makeWrapper ];
            buildInputs = [ llvm pkgs.libffi ];
            LLVM_SYS_221_PREFIX = "${llvm.dev}";
            postInstall = "wrapProgram $out/bin/llvm-metalc --prefix PATH : ${downgrade}/bin";
          };
          shell = pkgs.mkShell {
            packages = [ toolchain llvm downgrade ];
            buildInputs = [ pkgs.libffi ];
            LLVM_SYS_221_PREFIX = "${llvm.dev}";
          };
        in {
          packages = { inherit llvm-metalc; default = llvm-metalc; };
          devShells.default = shell;
        });
    in {
      packages = builtins.mapAttrs (_: outputs: outputs.packages) perSystem;
      devShells = builtins.mapAttrs (_: outputs: outputs.devShells) perSystem;
    };
}
