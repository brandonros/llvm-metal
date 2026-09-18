//! AIR lowering, the pinned external bitcode writer, and metallib packaging.
use crate::{air, parse_bitcode};
use inkwell::module::Module;
use llvm_metal_abi::{KernelInterface, MetalBindings};
use std::{fs, process::Command};

pub struct CompiledKernel {
    pub air_ir: String,
    pub air_bitcode: Vec<u8>,
    pub metallib: Vec<u8>,
    pub bindings: MetalBindings,
}

pub fn compile(module: &Module<'_>, interface: &KernelInterface) -> Result<CompiledKernel, String> {
    compile_with_policy(module, interface, air::InliningPolicy::All)
}

pub fn compile_with_policy(
    module: &Module<'_>,
    interface: &KernelInterface,
    policy: air::InliningPolicy,
) -> Result<CompiledKernel, String> {
    let (module, bindings) = air::legalize_with_policy(module, interface, policy)?;
    let temporary = tempfile::tempdir().map_err(|e| e.to_string())?;
    let modern = temporary.path().join("modern.bc");
    let legacy = temporary.path().join("air.bc");
    fs::write(&modern, module.write_bitcode_to_memory().as_slice()).map_err(|e| e.to_string())?;
    let result = Command::new("llvm-downgrade")
        .arg(&modern)
        .arg("--bitcode-version=14.0")
        .arg("-o")
        .arg(&legacy)
        .output()
        .map_err(|e| format!("llvm-downgrade: {e}; use the pinned Nix shell"))?;
    if !result.status.success() {
        return Err(format!(
            "AIR serialization: {}",
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    let bitcode = fs::read(legacy).map_err(|e| e.to_string())?;
    let context = inkwell::context::Context::create();
    parse_bitcode(&context, &bitcode, "downgraded AIR").map_err(|e| e.to_string())?;
    Ok(CompiledKernel {
        air_ir: module.print_to_string().to_string(),
        metallib: llvm_metal_metallib::package(&bindings.entry, &bitcode)?,
        air_bitcode: bitcode,
        bindings,
    })
}
