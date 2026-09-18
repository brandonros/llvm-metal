//! Parse and structurally verify LLVM input. This does not establish Metal legality.

use inkwell::{context::Context, memory_buffer::MemoryBuffer, module::Module};
use std::fmt;

#[derive(Debug)]
pub enum InputError {
    Parse(String),
    Verify(String),
    Entry(String),
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(message) => write!(f, "LLVM parse error: {message}"),
            Self::Verify(message) => write!(f, "LLVM verification error: {message}"),
            Self::Entry(message) => write!(f, "entry error: {message}"),
        }
    }
}

impl std::error::Error for InputError {}

/// Parse textual LLVM IR using LLVM's reader, then verify SSA and type invariants.
pub fn parse_ir<'ctx>(
    context: &'ctx Context,
    source: &[u8],
    name: &str,
) -> Result<Module<'ctx>, InputError> {
    let buffer = input_buffer(source, name)?;
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|error| InputError::Parse(error.to_string()))?;
    verify(module)
}

/// Parse LLVM bitcode with the linked LLVM version, then run its verifier.
/// Newer producer versions are not guaranteed to be readable by LLVM 21.
pub fn parse_bitcode<'ctx>(
    context: &'ctx Context,
    bytes: &[u8],
    name: &str,
) -> Result<Module<'ctx>, InputError> {
    let buffer = input_buffer(bytes, name)?;
    let module = Module::parse_bitcode_from_buffer(&buffer, context)
        .map_err(|error| InputError::Parse(error.to_string()))?;
    verify(module)
}

/// Require a defined function. Kernel ABI and dependency-closure checks are separate work.
pub fn require_entry(module: &Module<'_>, name: &str) -> Result<(), InputError> {
    if name.contains('\0') {
        return Err(InputError::Entry("entry name contains a NUL byte".into()));
    }
    let function = module
        .get_function(name)
        .ok_or_else(|| InputError::Entry(format!("function {name:?} is missing")))?;
    if function.count_basic_blocks() == 0 {
        return Err(InputError::Entry(format!(
            "function {name:?} is only a declaration"
        )));
    }
    Ok(())
}

fn input_buffer(bytes: &[u8], name: &str) -> Result<MemoryBuffer<'static>, InputError> {
    if name.contains('\0') {
        return Err(InputError::Parse("input name contains a NUL byte".into()));
    }
    Ok(MemoryBuffer::create_from_memory_range_copy(bytes, name))
}

fn verify(module: Module<'_>) -> Result<Module<'_>, InputError> {
    module
        .verify()
        .map_err(|error| InputError::Verify(error.to_string()))?;
    Ok(module)
}
pub mod air;
pub mod compile;
mod wide;

mod libcalls;
