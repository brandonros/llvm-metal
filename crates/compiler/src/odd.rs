//! Exact-span promotion of byte-sized scalar integers to i32/i64.
//! Call only on a disposable, verified module clone. No ABI changes are allowed.
use inkwell::module::Module;
use std::ffi::CStr;

pub(crate) fn lower(module: &Module<'_>) -> Result<(), String> {
    unsafe extern "C" {
        fn LLVMMetalLowerOddIntegers(
            module: inkwell::llvm_sys::prelude::LLVMModuleRef,
        ) -> *mut std::ffi::c_char;
    }
    // SAFETY: verified disposable module; release the owned native diagnostic.
    unsafe {
        let error = LLVMMetalLowerOddIntegers(module.as_mut_ptr());
        if !error.is_null() {
            let text = CStr::from_ptr(error).to_string_lossy().into_owned();
            inkwell::llvm_sys::core::LLVMDisposeMessage(error);
            return Err(text);
        }
    }
    module.verify().map_err(|error| error.to_string())
}
