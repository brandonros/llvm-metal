//! Extract the target's constant bytes before any internalization or DCE.
use inkwell::{
    llvm_sys::{LLVMLinkage, LLVMOpcode, core::*, prelude::*},
    module::Module,
    values::BasicValueEnum,
};
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
    remove(&module)?;
    module.verify().map_err(|e| e.to_string())?;
    Ok((module, descriptors))
}

// A constant is dead when nothing but other dead constants refers to it. The
// initializer of a deleted llvm.used list is the expected case.
unsafe fn dead_constant(value: LLVMValueRef) -> bool {
    unsafe {
        if LLVMIsAConstant(value).is_null() || !LLVMIsAGlobalValue(value).is_null() {
            return false;
        }
        let mut use_ = LLVMGetFirstUse(value);
        while !use_.is_null() {
            if !dead_constant(LLVMGetUser(use_)) {
                return false;
            }
            use_ = LLVMGetNextUse(use_);
        }
        true
    }
}

// Pointer casts and all-zero GEPs, as Value::stripPointerCasts. Not aliases.
unsafe fn strip_pointer_casts(mut value: LLVMValueRef) -> LLVMValueRef {
    unsafe {
        while !LLVMIsAConstantExpr(value).is_null() {
            let zero = move |i| {
                let index = LLVMGetOperand(value, i);
                !LLVMIsAConstantInt(index).is_null() && LLVMIsNull(index) != 0
            };
            match LLVMGetConstOpcode(value) {
                LLVMOpcode::LLVMBitCast | LLVMOpcode::LLVMAddrSpaceCast => {}
                LLVMOpcode::LLVMGetElementPtr
                    if (1..LLVMGetNumOperands(value) as u32).all(zero) => {}
                _ => break,
            }
            value = LLVMGetOperand(value, 0);
        }
        value
    }
}

unsafe fn is_descriptor(value: LLVMValueRef) -> bool {
    unsafe {
        let value = strip_pointer_casts(value);
        if LLVMIsAGlobalValue(value).is_null() {
            return false;
        }
        let mut length = 0;
        let name = LLVMGetValueName2(value, &mut length);
        std::slice::from_raw_parts(name.cast::<u8>(), length)
            .starts_with(llvm_metal_kernel::PREFIX.as_bytes())
    }
}

// Rebuild one used list without descriptor entries, keeping unrelated operands
// (including constant-expression casts), the section and the address space.
unsafe fn filter_used(module: LLVMModuleRef, name: &std::ffi::CStr) {
    unsafe {
        let list = LLVMGetNamedGlobal(module, name.as_ptr());
        if list.is_null() || LLVMGetInitializer(list).is_null() {
            return;
        }
        let initializer = LLVMGetInitializer(list);
        let mut kept: Vec<_> = (0..LLVMGetNumOperands(initializer) as u32)
            .map(|i| LLVMGetOperand(initializer, i))
            .filter(|&entry| !is_descriptor(entry))
            .collect();
        if kept.len() == LLVMGetNumOperands(initializer) as usize {
            return;
        }
        let element = LLVMGetElementType(LLVMGlobalGetValueType(list));
        let section = LLVMGetSection(list);
        let section = (!section.is_null()).then(|| std::ffi::CStr::from_ptr(section).to_owned());
        let space = LLVMGetPointerAddressSpace(LLVMTypeOf(list));
        LLVMDeleteGlobal(list);
        if kept.is_empty() {
            return;
        }
        let ty = LLVMArrayType2(element, kept.len() as u64);
        let new = LLVMAddGlobalInAddressSpace(module, ty, name.as_ptr(), space);
        LLVMSetLinkage(new, LLVMLinkage::LLVMAppendingLinkage);
        LLVMSetInitializer(
            new,
            LLVMConstArray2(element, kept.as_mut_ptr(), kept.len() as u64),
        );
        if let Some(section) = section {
            LLVMSetSection(new, section.as_ptr());
        }
    }
}

// Descriptors are metadata roots only. Refuse any executable or global use
// before erasing anything.
fn remove(module: &Module<'_>) -> Result<(), String> {
    // SAFETY: verified private clone with exclusive mutation. Descriptor globals
    // are collected before deletion and each is detached from dead constants.
    unsafe {
        let raw = module.as_mut_ptr();
        filter_used(raw, c"llvm.used");
        filter_used(raw, c"llvm.compiler.used");
        let mut descriptors = Vec::new();
        let mut global = LLVMGetFirstGlobal(raw);
        while !global.is_null() {
            if is_descriptor(global) {
                let mut use_ = LLVMGetFirstUse(global);
                while !use_.is_null() {
                    if !dead_constant(LLVMGetUser(use_)) {
                        return Err("descriptor has non-metadata uses".into());
                    }
                    use_ = LLVMGetNextUse(use_);
                }
                descriptors.push(global);
            }
            global = LLVMGetNextGlobal(global);
        }
        for global in descriptors {
            // The C API cannot destroy dead constants, so point them elsewhere.
            LLVMReplaceAllUsesWith(global, LLVMConstNull(LLVMTypeOf(global)));
            LLVMDeleteGlobal(global);
        }
    }
    Ok(())
}
