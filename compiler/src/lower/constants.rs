//! Constants live in thread memory. Metal wants globals in its constant address
//! space, and a pointer into that space is a different type from every other
//! pointer in the program; so the globals become one image in constant space,
//! each thread copies it once, and the program addresses the copy.
//!
//! Pre: every global is a constant whose addresses lead to other globals
//! (`Rule::ConstantData`). Post: no instruction refers to a global.
use crate::ir::{self, Value};
use inkwell::llvm_sys::{LLVMOpcode, LLVMTypeKind, core::*, prelude::*, target::*};
use std::collections::HashMap;

/// Where each global sits in the image, and the image itself.
pub struct Image {
    pub offsets: HashMap<Value, u64>,
    pub global: Value,
    pub size: u64,
    pub alignment: u32,
}

/// An address stored inside a constant: where it sits, and what it points to.
pub struct Relocation {
    pub offset: u64,
    pub target: Value,
}

pub unsafe fn image(module: LLVMModuleRef) -> (Image, Vec<Relocation>) {
    unsafe {
        let context = LLVMGetModuleContext(module);
        let layout = LLVMGetModuleDataLayout(module);
        let byte = LLVMInt8TypeInContext(context);
        let (mut parts, mut offsets, mut relocations) = (Vec::new(), HashMap::new(), Vec::new());
        let (mut size, mut alignment) = (0u64, 1u32);
        for global in ir::globals(module) {
            let initializer = LLVMGetInitializer(global);
            let align = LLVMGetAlignment(global).max(1);
            let padding = size.next_multiple_of(align.into()) - size;
            if padding != 0 {
                parts.push(LLVMConstNull(LLVMArrayType2(byte, padding)));
            }
            size += padding;
            offsets.insert(global, size);
            for (path, target) in addresses(initializer) {
                let offset = size + offset_of(layout, LLVMTypeOf(initializer), &path);
                relocations.push(Relocation { offset, target });
            }
            parts.push(without_addresses(context, initializer));
            size += LLVMABISizeOfType(layout, LLVMTypeOf(initializer));
            alignment = alignment.max(align);
        }
        let value = LLVMConstStructInContext(context, parts.as_mut_ptr(), parts.len() as u32, 1);
        let global = LLVMAddGlobalInAddressSpace(
            module,
            LLVMTypeOf(value),
            c"llvm_metal.constants".as_ptr(),
            2,
        );
        LLVMSetInitializer(global, value);
        LLVMSetGlobalConstant(global, 1);
        LLVMSetLinkage(global, inkwell::llvm_sys::LLVMLinkage::LLVMInternalLinkage);
        LLVMSetAlignment(global, alignment);
        (
            Image {
                offsets,
                global,
                size,
                alignment,
            },
            relocations,
        )
    }
}

/// Does this constant contain the address of a global?
pub unsafe fn refers(constant: Value) -> bool {
    unsafe {
        !LLVMIsAConstant(constant).is_null()
            && LLVMIsAFunction(constant).is_null()
            && (!LLVMIsAGlobalVariable(constant).is_null()
                || ir::operands(constant).into_iter().any(|part| refers(part)))
    }
}

/// The same value computed by instructions at `builder`, with every global's
/// address taken from the thread's copy at `copy`.
pub unsafe fn materialize(
    builder: LLVMBuilderRef,
    image: &Image,
    copy: Value,
    constant: Value,
) -> Value {
    unsafe {
        if !refers(constant) {
            return constant;
        }
        let context = LLVMGetTypeContext(LLVMTypeOf(constant));
        if !LLVMIsAGlobalVariable(constant).is_null() {
            let mut offset =
                LLVMConstInt(LLVMInt64TypeInContext(context), image.offsets[&constant], 0);
            let byte = LLVMInt8TypeInContext(context);
            return LLVMBuildGEP2(builder, byte, copy, &mut offset, 1, c"".as_ptr());
        }
        if !LLVMIsAConstantExpr(constant).is_null() {
            let base = materialize(builder, image, copy, LLVMGetOperand(constant, 0));
            return match LLVMGetConstOpcode(constant) {
                LLVMOpcode::LLVMGetElementPtr => {
                    let mut indices = ir::operands(constant).split_off(1);
                    let element = LLVMGetGEPSourceElementType(constant);
                    let count = indices.len() as u32;
                    LLVMBuildGEP2(
                        builder,
                        element,
                        base,
                        indices.as_mut_ptr(),
                        count,
                        c"".as_ptr(),
                    )
                }
                LLVMOpcode::LLVMPtrToInt => {
                    LLVMBuildPtrToInt(builder, base, LLVMTypeOf(constant), c"".as_ptr())
                }
                other => unreachable!("constant expression {other:?} holding an address"),
            };
        }
        // An aggregate: the plain part as a constant, the addresses inserted.
        let mut value = without_addresses(context, constant);
        for (path, target) in addresses(constant) {
            let address = materialize(builder, image, copy, target);
            value = insert(builder, value, &path, address);
        }
        value
    }
}

unsafe fn insert(builder: LLVMBuilderRef, aggregate: Value, path: &[u32], value: Value) -> Value {
    unsafe {
        let value = match path {
            [_] => value,
            [first, rest @ ..] => {
                let inner = LLVMBuildExtractValue(builder, aggregate, *first, c"".as_ptr());
                insert(builder, inner, rest, value)
            }
            [] => unreachable!("an address is inside the aggregate"),
        };
        LLVMBuildInsertValue(builder, aggregate, value, path[0], c"".as_ptr())
    }
}

/// The addresses inside a constant, each with the path of indices that leads
/// to it. A constant that is itself an address has the empty path.
unsafe fn addresses(constant: Value) -> Vec<(Vec<u32>, Value)> {
    unsafe {
        if !refers(constant) {
            return Vec::new();
        }
        if ir::is_pointer(LLVMTypeOf(constant)) {
            return vec![(Vec::new(), constant)];
        }
        let parts = ir::operands(constant).into_iter().enumerate();
        parts
            .flat_map(|(index, part)| {
                addresses(part).into_iter().map(move |(mut path, target)| {
                    path.insert(0, index as u32);
                    (path, target)
                })
            })
            .collect()
    }
}

/// The same constant with null in place of every address.
unsafe fn without_addresses(context: LLVMContextRef, constant: Value) -> Value {
    unsafe {
        if !refers(constant) {
            return constant;
        }
        let ty = LLVMTypeOf(constant);
        let mut parts: Vec<_> = ir::operands(constant)
            .into_iter()
            .map(|part| without_addresses(context, part))
            .collect();
        match LLVMGetTypeKind(ty) {
            LLVMTypeKind::LLVMPointerTypeKind => LLVMConstNull(ty),
            LLVMTypeKind::LLVMArrayTypeKind => LLVMConstArray2(
                LLVMGetElementType(ty),
                parts.as_mut_ptr(),
                parts.len() as u64,
            ),
            LLVMTypeKind::LLVMStructTypeKind if !LLVMGetStructName(ty).is_null() => {
                LLVMConstNamedStruct(ty, parts.as_mut_ptr(), parts.len() as u32)
            }
            LLVMTypeKind::LLVMStructTypeKind => LLVMConstStructInContext(
                context,
                parts.as_mut_ptr(),
                parts.len() as u32,
                LLVMIsPackedStruct(ty),
            ),
            kind => unreachable!("an address inside a constant of kind {kind:?}"),
        }
    }
}

unsafe fn offset_of(layout: LLVMTargetDataRef, ty: LLVMTypeRef, path: &[u32]) -> u64 {
    unsafe {
        let Some((&index, rest)) = path.split_first() else {
            return 0;
        };
        match LLVMGetTypeKind(ty) {
            LLVMTypeKind::LLVMStructTypeKind => {
                LLVMOffsetOfElement(layout, ty, index)
                    + offset_of(layout, LLVMStructGetTypeAtIndex(ty, index), rest)
            }
            _ => {
                let element = LLVMGetElementType(ty);
                u64::from(index) * LLVMABISizeOfType(layout, element)
                    + offset_of(layout, element, rest)
            }
        }
    }
}
