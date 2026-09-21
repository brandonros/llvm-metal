//! Compile a Rust kernel crate to a Metal library.
//!
//! The stages, in order, each run once:
//! 1. `cargo`: build the crate, collect its bitcode.
//! 2. link the crates into one module.
//! 3. `select`: give panics their device meaning; keep what the entry reaches.
//! 4. `verify`: refuse what a GPU kernel may not contain. Nothing later refuses.
//!
//! No LLVM optimization pass runs here. rustc optimized the code as the crate's
//! profile asked, and Apple's compiler optimizes it again; whether a kernel
//! builds cannot depend on either.
pub mod cargo;
pub mod emit;
mod ir;
pub mod select;
pub mod verify;

use inkwell::{context::Context, llvm_sys::core::*, memory_buffer::MemoryBuffer, module::Module};
use std::ffi::CString;
pub use verify::{Rule, Violation};

#[derive(Debug)]
pub enum Error {
    Input(String),
    Refused(Vec<Violation>),
}

/// Link `bitcode` and reduce it to the verified program of the kernel `name`.
pub fn program<'ctx>(
    context: &'ctx Context,
    bitcode: &[Vec<u8>],
    name: &str,
) -> Result<Module<'ctx>, Error> {
    let module = context.create_module(name);
    for bytes in bitcode {
        let buffer = MemoryBuffer::create_from_memory_range_copy(bytes, "crate");
        let member = Module::parse_bitcode_from_buffer(&buffer, context)
            .map_err(|error| Error::Input(error.to_string()))?;
        module
            .link_in_module(member)
            .map_err(|error| Error::Input(error.to_string()))?;
    }
    let symbol = CString::new(format!("kernel.{name}")).map_err(|e| Error::Input(e.to_string()))?;
    // SAFETY: `module` is live and exclusively ours; each stage collects the
    // values it will change before changing any.
    let violations = unsafe {
        let raw = module.as_mut_ptr();
        let entry = LLVMGetNamedFunction(raw, symbol.as_ptr());
        if entry.is_null() {
            return Err(Error::Input(format!("no kernel named {name}")));
        }
        select::rewrite_panics(raw);
        select::keep_reachable(raw, entry);
        verify::verify(raw)
    };
    module
        .verify()
        .map_err(|error| Error::Input(error.to_string()))?;
    if violations.is_empty() {
        Ok(module)
    } else {
        Err(Error::Refused(violations))
    }
}
