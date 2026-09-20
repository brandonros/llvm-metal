//! Address-space inference and AIR pointer fixups, in their required order:
//! nullable PHIs before inference, then constant volatile-load typing and
//! null-cast folding afterward.
use crate::pointer_provenance::{constant_pointer, underlying_objects};
use inkwell::{
    llvm_sys::{LLVMOpcode, core::*, debuginfo::LLVMInstructionSetDebugLoc, prelude::*},
    module::Module,
};

unsafe fn space(value: LLVMValueRef) -> u32 {
    unsafe { LLVMGetPointerAddressSpace(LLVMTypeOf(value)) }
}
unsafe fn null_or_undef(value: LLVMValueRef) -> bool {
    unsafe { !LLVMIsAConstantPointerNull(value).is_null() || !LLVMIsAUndefValue(value).is_null() }
}
unsafe fn constant_cast(value: LLVMValueRef) -> bool {
    unsafe {
        !LLVMIsAConstantExpr(value).is_null()
            && LLVMGetConstOpcode(value) == LLVMOpcode::LLVMAddrSpaceCast
    }
}
unsafe fn instructions(function: LLVMValueRef) -> Vec<LLVMValueRef> {
    unsafe {
        let mut all = Vec::new();
        let mut block = LLVMGetFirstBasicBlock(function);
        while !block.is_null() {
            let mut instruction = LLVMGetFirstInstruction(block);
            while !instruction.is_null() {
                all.push(instruction);
                instruction = LLVMGetNextInstruction(instruction);
            }
            block = LLVMGetNextBasicBlock(block);
        }
        all
    }
}

// LLVM 21's inference handles nullable selects but joins a PHI's flat null/
// undef inputs with its device inputs as "flat". Give such PHIs an explicit
// address space only when every concrete underlying object proves the same
// device/constant space. Private or mixed-space pointers remain unsupported.
unsafe fn type_nullable_phis(function: LLVMValueRef) {
    unsafe {
        let context = LLVMGetTypeContext(LLVMTypeOf(function));
        let builder = LLVMCreateBuilderInContext(context);
        for phi in instructions(function) {
            if LLVMIsAPHINode(phi).is_null()
                || LLVMGetTypeKind(LLVMTypeOf(phi))
                    != inkwell::llvm_sys::LLVMTypeKind::LLVMPointerTypeKind
                || space(phi) != 0
            {
                continue;
            }
            let incoming = LLVMCountIncoming(phi);
            if !(0..incoming).any(|i| null_or_undef(LLVMGetIncomingValue(phi, i))) {
                continue;
            }
            let mut proven = 0;
            let mut valid = true;
            for object in underlying_objects(phi) {
                if null_or_undef(object) {
                    continue;
                }
                let candidate = space(object);
                if (candidate != 1 && candidate != 2) || (proven != 0 && proven != candidate) {
                    valid = false;
                    break;
                }
                proven = candidate;
            }
            if !valid || proven == 0 {
                continue;
            }
            let ty = LLVMPointerTypeInContext(context, proven);
            LLVMPositionBuilderBefore(builder, phi);
            let typed = LLVMBuildPhi(builder, ty, c"nullable".as_ptr());
            LLVMInstructionSetDebugLoc(typed, std::ptr::null_mut());
            for i in 0..incoming {
                let value = LLVMGetIncomingValue(phi, i);
                let mut block = LLVMGetIncomingBlock(phi, i);
                LLVMPositionBuilderBefore(builder, LLVMGetBasicBlockTerminator(block));
                let mut value = if !LLVMIsAConstantPointerNull(value).is_null() {
                    LLVMConstPointerNull(ty)
                } else if !LLVMIsAPoisonValue(value).is_null() {
                    LLVMGetPoison(ty)
                } else if !LLVMIsAUndefValue(value).is_null() {
                    LLVMGetUndef(ty)
                } else {
                    LLVMBuildAddrSpaceCast(builder, value, ty, c"".as_ptr())
                };
                LLVMAddIncoming(typed, &mut value, &mut block, 1);
            }
            let mut first = LLVMGetFirstInstruction(LLVMGetInstructionParent(phi));
            while !LLVMIsAPHINode(first).is_null() {
                first = LLVMGetNextInstruction(first);
            }
            LLVMPositionBuilderBefore(builder, first);
            let generic = LLVMBuildAddrSpaceCast(builder, typed, LLVMTypeOf(phi), c"".as_ptr());
            LLVMReplaceAllUsesWith(phi, generic);
            LLVMInstructionEraseFromParent(phi);
        }
        LLVMDisposeBuilder(builder);
    }
}

// AIR's supported pointer spaces use a zero null representation. LLVM has no
// AIR target model to fold the constant casts introduced by its inference pass.
unsafe fn fold_constant_casts(function: LLVMValueRef) {
    unsafe {
        for instruction in instructions(function) {
            for i in 0..LLVMGetNumOperands(instruction) as u32 {
                let cast = LLVMGetOperand(instruction, i);
                if !constant_cast(cast) {
                    continue;
                }
                let source = LLVMGetOperand(cast, 0);
                if !LLVMIsAConstantPointerNull(source).is_null()
                    && space(cast) <= 2
                    && space(source) <= 2
                {
                    LLVMSetOperand(instruction, i, LLVMConstPointerNull(LLVMTypeOf(cast)));
                }
            }
        }
    }
}

// LLVM's generic inference deliberately leaves volatile loads untouched. These
// loads already refer to a reviewed, defined immutable global in Metal constant
// storage; retain that pointer's address space instead of casting it to private.
unsafe fn type_constant_volatile_loads(function: LLVMValueRef) {
    unsafe {
        for load in instructions(function) {
            if LLVMIsALoadInst(load).is_null()
                || LLVMGetVolatile(load) == 0
                || LLVMGetOrdering(load)
                    != inkwell::llvm_sys::LLVMAtomicOrdering::LLVMAtomicOrderingNotAtomic
            {
                continue;
            }
            let cast = LLVMGetOperand(load, 0);
            if !constant_cast(cast) || space(cast) != 0 {
                continue;
            }
            let source = LLVMGetOperand(cast, 0);
            if space(source) == 2 && constant_pointer(source) {
                LLVMSetOperand(load, 0, source);
            }
        }
    }
}

// LLVM's pass takes its flat address space from the target, and AIR has no
// upstream TargetMachine. The textual pipeline accepts no parameter for it, but
// LLVM's own -assume-default-is-flat-addrspace option selects address space 0.
// It is process-wide, which is what this compiler wants for every module. An
// NVPTX TargetMachine is not a substitute: it also assumes allocas live in
// NVPTX's address space 5. If a future LLVM drops the option, option parsing
// reports it and exits instead of silently skipping inference.
unsafe fn run_inference(function: LLVMValueRef) {
    use inkwell::llvm_sys::{support::LLVMParseCommandLineOptions, transforms::pass_builder::*};
    static FLAT: std::sync::Once = std::sync::Once::new();
    unsafe {
        FLAT.call_once(|| {
            let arguments = [
                c"llvm-metal".as_ptr(),
                c"-assume-default-is-flat-addrspace".as_ptr(),
            ];
            LLVMParseCommandLineOptions(2, arguments.as_ptr(), std::ptr::null());
        });
        let options = LLVMCreatePassBuilderOptions();
        let failed = LLVMRunPassesOnFunction(
            function,
            c"infer-address-spaces".as_ptr(),
            std::ptr::null_mut(),
            options,
        );
        LLVMDisposePassBuilderOptions(options);
        assert!(
            failed.is_null(),
            "LLVM rejected the infer-address-spaces pipeline"
        );
    }
}

/// Infer one defined function. Call specialization uses this per caller.
pub(crate) unsafe fn infer_function(function: LLVMValueRef) {
    unsafe {
        if LLVMCountBasicBlocks(function) == 0 {
            return;
        }
        type_nullable_phis(function);
        run_inference(function);
        type_constant_volatile_loads(function);
        fold_constant_casts(function);
    }
}

pub(crate) fn infer(module: &Module<'_>) {
    // SAFETY: verified live module, exclusive mutation during the pass.
    unsafe {
        let mut function = LLVMGetFirstFunction(module.as_mut_ptr());
        while !function.is_null() {
            infer_function(function);
            function = LLVMGetNextFunction(function);
        }
    }
}
