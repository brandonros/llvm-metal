//! Extract the target's constant bytes before any internalization or DCE.
use inkwell::{module::Module, values::BasicValueEnum};
use llvm_metal_abi::descriptor::Descriptor;
use std::collections::BTreeMap;

pub fn validate_entry(module: &Module<'_>, d: &Descriptor) -> Result<(), String> {
    d.interface()?;
    let entry = module
        .get_function(&d.entry)
        .ok_or("descriptor entry is missing")?;
    let ty = entry.get_type();
    if entry.count_basic_blocks() == 0
        || entry.get_call_conventions() != 0
        || ty.is_var_arg()
        || ty.get_return_type().is_some()
        || ty.count_param_types() as usize != d.arguments.len()
        || ty.get_param_types().iter().any(|p| {
            !p.is_pointer_type()
                || p.into_pointer_type().get_address_space() != inkwell::AddressSpace::default()
        })
    {
        return Err(
            "descriptor entry must define C void(buffer pointers...) with matching arguments"
                .into(),
        );
    }
    Ok(())
}

/// Returns a metadata-free clone and the descriptors extracted from that exact
/// module. Unrelated llvm.used/compiler.used entries are preserved.
pub fn extract<'ctx>(
    input: &Module<'ctx>,
) -> Result<(Module<'ctx>, BTreeMap<String, Descriptor>), String> {
    input.verify().map_err(|e| e.to_string())?;
    let module = input.clone();
    let globals: Vec<_> = module
        .get_globals()
        .filter(|g| {
            g.get_name()
                .to_bytes()
                .starts_with(llvm_metal_kernel::PREFIX.as_bytes())
        })
        .collect();
    if globals.is_empty() {
        return Err("no embedded kernel descriptors".into());
    }
    let mut descriptors = BTreeMap::new();
    for global in &globals {
        if !global.is_constant() {
            return Err("descriptor must be constant".into());
        }
        let mut value = global
            .get_initializer()
            .ok_or("descriptor has no initializer")?;
        // rustc may wrap a byte array in a single-field aggregate.
        while let BasicValueEnum::StructValue(s) = value {
            if s.count_fields() != 1 {
                return Err("descriptor must contain only byte data".into());
            }
            value = s
                .get_field_at_index(0)
                .ok_or("empty descriptor aggregate")?;
        }
        let BasicValueEnum::ArrayValue(array) = value else {
            return Err("descriptor must be a byte array".into());
        };
        if !array.is_const_string() {
            return Err("descriptor must be constant bytes".into());
        }
        let d = Descriptor::decode(
            array
                .as_const_string()
                .ok_or("cannot read descriptor bytes")?,
        )?;
        if global.get_name().to_bytes()
            != format!("{}{}", llvm_metal_kernel::PREFIX, d.entry).as_bytes()
        {
            return Err("descriptor symbol/entry mismatch".into());
        }
        validate_entry(&module, &d)?;
        if descriptors.insert(d.entry.clone(), d).is_some() {
            return Err("duplicate descriptor entry".into());
        }
    }
    unsafe extern "C" {
        fn LLVMMetalRemoveDescriptors(module: inkwell::llvm_sys::prelude::LLVMModuleRef) -> bool;
    }
    // SAFETY: verified private clone; the helper rejects executable uses and
    // preserves unrelated metadata roots before removing descriptor globals.
    if !unsafe { LLVMMetalRemoveDescriptors(module.as_mut_ptr()) } {
        return Err("descriptor has non-metadata uses".into());
    }
    module.verify().map_err(|e| e.to_string())?;
    Ok((module, descriptors))
}
