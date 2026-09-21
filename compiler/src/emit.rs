//! Turn a module that already follows Metal's rules into a Metal library: name
//! the target, describe the entry point to Metal, encode, package.
//!
//! The entry's signature is the description. Each `ptr addrspace(1)` parameter
//! is the device buffer at that index; a final `i32` is the thread's position.
use inkwell::{
    AddressSpace,
    module::Module,
    targets::{TargetData, TargetTriple},
    values::BasicMetadataValueEnum,
};
use std::{fs, path::Path, process::Command};

const TRIPLE: &str = "air64-apple-macosx13.0.0";
const LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// Write `kernel.metallib` for `entry` into `directory`, beside the bitcode it
/// was made from, and return its path.
pub fn library(
    module: &Module<'_>,
    entry: &str,
    directory: &Path,
) -> Result<std::path::PathBuf, String> {
    module.set_triple(&TargetTriple::create(TRIPLE));
    module.set_data_layout(&TargetData::create(LAYOUT).get_data_layout());
    clear_position_independence(module);
    describe(module, entry)?;
    module.verify().map_err(|error| error.to_string())?;
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let (modern, legacy) = (directory.join("kernel.bc"), directory.join("kernel.air.bc"));
    if !module.write_bitcode_to_path(&modern) {
        return Err(format!("could not write {}", modern.display()));
    }
    // Apple's compiler reads only the LLVM 14 encoding of bitcode.
    let output = Command::new("llvm-downgrade")
        .arg(&modern)
        .args(["--bitcode-version=14.0", "-o"])
        .arg(&legacy)
        .output()
        .map_err(|error| format!("llvm-downgrade: {error}; use the Nix shell"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let air = fs::read(&legacy).map_err(|error| error.to_string())?;
    let path = directory.join("kernel.metallib");
    fs::write(&path, llvm_metal_metallib::package(entry, &air)?).map_err(|e| e.to_string())?;
    Ok(path)
}

/// The metadata Metal reads to bind arguments. Layout after GPUCompiler.jl.
fn describe(module: &Module<'_>, entry: &str) -> Result<(), String> {
    let context = module.get_context();
    let function = module
        .get_function(entry)
        .ok_or(format!("no function named {entry}"))?;
    let number = |value: u64| -> BasicMetadataValueEnum {
        context.i32_type().const_int(value, false).into()
    };
    let text = |value: &str| -> BasicMetadataValueEnum { context.metadata_string(value).into() };
    let mut arguments = Vec::new();
    for (index, parameter) in function.get_param_iter().enumerate() {
        let index = index as u64;
        let device = parameter.is_pointer_value()
            && parameter
                .into_pointer_value()
                .get_type()
                .get_address_space()
                == AddressSpace::from(1u16);
        let last = index as u32 + 1 == function.count_params();
        arguments.push(
            if device {
                context.metadata_node(&[
                    number(index),
                    text("air.buffer"),
                    text("air.location_index"),
                    number(index),
                    number(1),
                    text("air.read_write"),
                    text("air.address_space"),
                    number(1),
                    text("air.arg_type_size"),
                    number(1),
                    text("air.arg_type_align_size"),
                    number(1),
                    text("air.arg_type_name"),
                    text("uchar"),
                    text("air.arg_name"),
                    text(&format!("buffer{index}")),
                ])
            } else if last
                && parameter.is_int_value()
                && parameter.into_int_value().get_type().get_bit_width() == 32
            {
                context.metadata_node(&[
                    number(index),
                    text("air.thread_position_in_grid"),
                    text("air.arg_type_name"),
                    text("uint"),
                    text("air.arg_name"),
                    text("thread"),
                ])
            } else {
                return Err(format!(
                    "entry parameter {index} is neither a device buffer nor the thread position"
                ));
            }
            .into(),
        );
    }
    let kernel = context.metadata_node(&[
        function.as_global_value().as_pointer_value().into(),
        context.metadata_node(&[]).into(),
        context.metadata_node(&arguments).into(),
    ]);
    let add = |name: &str, node| {
        module
            .add_global_metadata(name, &node)
            .map_err(|e| e.to_string())
    };
    add("air.kernel", kernel)?;
    add(
        "air.version",
        context.metadata_node(&[number(2), number(4), number(0)]),
    )?;
    add(
        "air.language_version",
        context.metadata_node(&[text("Metal"), number(3), number(0), number(0)]),
    )?;
    Ok(())
}

/// rustc marks its modules position-independent, which means nothing on a GPU
/// and can kill Apple's compiler (`tests/facts/pic_relocation.ll`). The C API
/// cannot remove a module flag, so set the levels to zero: not independent.
fn clear_position_independence(module: &Module<'_>) {
    use inkwell::llvm_sys::core::*;
    let flags = c"llvm.module.flags";
    // SAFETY: the module is live and ours, and the verifier has checked that
    // each flag is a node of three operands whose second is a string.
    unsafe {
        let raw = module.as_mut_ptr();
        let count = LLVMGetNamedMetadataNumOperands(raw, flags.as_ptr());
        let mut nodes = vec![std::ptr::null_mut(); count as usize];
        LLVMGetNamedMetadataOperands(raw, flags.as_ptr(), nodes.as_mut_ptr());
        for node in nodes {
            let mut operands = [std::ptr::null_mut(); 3];
            LLVMGetMDNodeOperands(node, operands.as_mut_ptr());
            let mut length = 0;
            let key = LLVMGetMDString(operands[1], &mut length);
            let key = std::slice::from_raw_parts(key.cast::<u8>(), length as usize);
            if key == b"PIC Level" || key == b"PIE Level" {
                let zero = LLVMConstInt(LLVMInt32TypeInContext(LLVMGetModuleContext(raw)), 0, 0);
                LLVMReplaceMDNodeOperandWith(node, 2, LLVMValueAsMetadata(zero));
            }
        }
    }
}
