//! What a kernel may contain. `Rule`'s doc comments are the specification;
//! nothing else in the compiler refuses a kernel. A rule states a fact about the
//! program the author wrote, never about what an optimizer did to it.
use crate::ir::{self, Value};
use inkwell::llvm_sys::{LLVMOpcode, LLVMTypeKind, core::*, prelude::*};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// Every function is defined in the linked crates, or is a device operation
    /// (`llvm_metal.*`) or an LLVM intrinsic Apple's compiler is known to accept.
    /// libcore is not linked: what a kernel uses from it is generic or inline,
    /// and so already part of the kernel crate.
    KnownFunction,
    /// A buffer slot is a constant below `SLOTS`. Slots come from const
    /// generics, so this fails only for a handle type written outside `kernel`.
    ConstantSlot,
    /// Calls name their target. No function pointers, no `dyn`.
    DirectCall,
    /// No function reaches itself. The GPU has no call stack to grow.
    NoRecursion,
    /// Integers are at most 64 bits wide. There is no i128 on the GPU: use two
    /// words, or 32-bit limbs with 64-bit products.
    NativeInteger,
    /// Pointers and integers do not convert into each other. A pointer's
    /// address space is part of its type on the GPU and an integer has none.
    NoPointerCast,
    /// Globals are constants holding plain data: no pointers, no functions. A
    /// constant is copied to thread memory, where a stored address would be
    /// wrong. Mutable statics have no meaning across GPU threads.
    PlainConstant,
}

/// Buffer slots a kernel may name. Metal binds 31; two carry the launch.
pub const SLOTS: u64 = 29;

#[derive(Debug, PartialEq, Eq)]
pub struct Violation {
    pub rule: Rule,
    pub function: String,
    /// `file:line` when the kernel was built with line tables.
    pub location: Option<String>,
    pub detail: String,
}

const OPERATIONS: [&str; 5] = [
    "llvm_metal.thread_index",
    "llvm_metal.length",
    "llvm_metal.load",
    "llvm_metal.store",
    "llvm_metal.panic",
];

/// Intrinsics Apple's compiler accepted on `master` (M5, macOS 27), and the
/// hints that carry no meaning and that `lower` drops.
const INTRINSICS: [&str; 17] = [
    "llvm.experimental.noalias.scope.decl",
    "llvm.memcpy.",
    "llvm.memset.",
    "llvm.memmove.",
    "llvm.lifetime.",
    "llvm.assume",
    "llvm.bswap.",
    "llvm.ctlz.",
    "llvm.cttz.",
    "llvm.fshl.",
    "llvm.fshr.",
    "llvm.umin.",
    "llvm.umax.",
    "llvm.smin.",
    "llvm.smax.",
    "llvm.abs.",
    "llvm.dbg.",
];

pub unsafe fn verify(module: LLVMModuleRef) -> Vec<Violation> {
    let mut violations = Vec::new();
    unsafe {
        for global in ir::globals(module) {
            let initializer = LLVMGetInitializer(global);
            if LLVMIsGlobalConstant(global) == 0
                || initializer.is_null()
                || holds_pointer(LLVMTypeOf(initializer))
            {
                violations.push(Violation {
                    rule: Rule::PlainConstant,
                    function: String::new(),
                    location: None,
                    detail: ir::name(global),
                });
            }
        }
        for function in ir::functions(module) {
            let name = ir::name(function);
            if !ir::defined(function) {
                if !OPERATIONS.contains(&name.as_str())
                    && !INTRINSICS.iter().any(|prefix| name.starts_with(prefix))
                {
                    violations.push(Violation {
                        rule: Rule::KnownFunction,
                        function: name,
                        location: None,
                        detail: "undefined".into(),
                    });
                }
                continue;
            }
            if reaches_itself(function) {
                violations.push(Violation {
                    rule: Rule::NoRecursion,
                    function: name.clone(),
                    location: None,
                    detail: String::new(),
                });
            }
            for instruction in ir::instructions(function) {
                if let Some((rule, detail)) = broken(instruction) {
                    violations.push(Violation {
                        rule,
                        function: name.clone(),
                        location: ir::location(instruction),
                        detail,
                    });
                }
            }
        }
    }
    violations
}

/// The rule one instruction breaks, if any.
unsafe fn broken(instruction: Value) -> Option<(Rule, String)> {
    unsafe {
        let printed = || {
            let text = LLVMPrintValueToString(instruction);
            let owned = std::ffi::CStr::from_ptr(text)
                .to_string_lossy()
                .trim()
                .to_owned();
            LLVMDisposeMessage(text);
            owned
        };
        if matches!(
            ir::opcode(instruction),
            LLVMOpcode::LLVMPtrToInt | LLVMOpcode::LLVMIntToPtr
        ) {
            return Some((Rule::NoPointerCast, printed()));
        }
        let types = std::iter::once(LLVMTypeOf(instruction)).chain(
            ir::operands(instruction)
                .into_iter()
                .map(|operand| LLVMTypeOf(operand)),
        );
        if types.into_iter().any(|ty| wide(ty)) {
            return Some((Rule::NativeInteger, printed()));
        }
        if ir::is_call(instruction) {
            let Some(target) = ir::callee(instruction) else {
                return Some((Rule::DirectCall, printed()));
            };
            let name = ir::name(target);
            if ["llvm_metal.length", "llvm_metal.load", "llvm_metal.store"].contains(&name.as_str())
            {
                let slot = LLVMGetOperand(instruction, 0);
                if LLVMIsAConstantInt(slot).is_null() || LLVMConstIntGetZExtValue(slot) >= SLOTS {
                    return Some((Rule::ConstantSlot, printed()));
                }
            }
        }
        None
    }
}

unsafe fn wide(ty: LLVMTypeRef) -> bool {
    unsafe {
        match LLVMGetTypeKind(ty) {
            LLVMTypeKind::LLVMIntegerTypeKind => LLVMGetIntTypeWidth(ty) > 64,
            LLVMTypeKind::LLVMArrayTypeKind | LLVMTypeKind::LLVMVectorTypeKind => {
                wide(LLVMGetElementType(ty))
            }
            LLVMTypeKind::LLVMStructTypeKind => (0..LLVMCountStructElementTypes(ty))
                .any(|field| wide(LLVMStructGetTypeAtIndex(ty, field))),
            _ => false,
        }
    }
}

unsafe fn holds_pointer(ty: LLVMTypeRef) -> bool {
    unsafe {
        match LLVMGetTypeKind(ty) {
            LLVMTypeKind::LLVMPointerTypeKind => true,
            LLVMTypeKind::LLVMArrayTypeKind | LLVMTypeKind::LLVMVectorTypeKind => {
                holds_pointer(LLVMGetElementType(ty))
            }
            LLVMTypeKind::LLVMStructTypeKind => (0..LLVMCountStructElementTypes(ty))
                .any(|field| holds_pointer(LLVMStructGetTypeAtIndex(ty, field))),
            _ => false,
        }
    }
}

unsafe fn reaches_itself(function: Value) -> bool {
    unsafe {
        let mut seen = HashSet::new();
        let mut pending = vec![function];
        while let Some(caller) = pending.pop() {
            for instruction in ir::instructions(caller) {
                let Some(target) = ir::is_call(instruction)
                    .then(|| ir::callee(instruction))
                    .flatten()
                else {
                    continue;
                };
                if target == function {
                    return true;
                }
                if ir::defined(target) && seen.insert(target) {
                    pending.push(target);
                }
            }
        }
        false
    }
}
