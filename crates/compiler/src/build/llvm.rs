//! The LLVM operations the builder needs that `opt`, `llvm-link` and `llvm-nm`
//! used to supply: linking, entry selection, pass pipelines and symbol checks.
use inkwell::{
    context::Context,
    llvm_sys::{
        LLVMLinkage, LLVMVisibility, comdat::*, core::*, error::*, prelude::*,
        support::LLVMParseCommandLineOptions, target::*, target_machine::*,
        transforms::pass_builder::*,
    },
    module::Module,
};
use std::{
    collections::{HashMap, HashSet},
    ffi::{CStr, CString},
};

/// The version of the LLVM this binary links, which bounds the bitcode it reads.
pub fn linked_version() -> (u32, u32, u32) {
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    // SAFETY: the three out-parameters are valid for writes.
    unsafe { LLVMGetVersion(&mut major, &mut minor, &mut patch) };
    (major, minor, patch)
}

/// Link modules in order into an empty module, as `llvm-link` does.
pub fn link<'ctx>(
    context: &'ctx Context,
    members: Vec<Module<'ctx>>,
) -> Result<Module<'ctx>, String> {
    let linked = context.create_module("llvm-link");
    for member in members {
        linked.link_in_module(member).map_err(|e| e.to_string())?;
    }
    linked.verify().map_err(|e| e.to_string())?;
    Ok(linked)
}

unsafe fn name(value: LLVMValueRef) -> Vec<u8> {
    unsafe {
        let mut length = 0;
        let text = LLVMGetValueName2(value, &mut length);
        std::slice::from_raw_parts(text.cast::<u8>(), length).to_vec()
    }
}

unsafe fn global_objects(module: LLVMModuleRef) -> Vec<LLVMValueRef> {
    unsafe {
        let mut all = Vec::new();
        let mut function = LLVMGetFirstFunction(module);
        while !function.is_null() {
            all.push(function);
            function = LLVMGetNextFunction(function);
        }
        let mut global = LLVMGetFirstGlobal(module);
        while !global.is_null() {
            all.push(global);
            global = LLVMGetNextGlobal(global);
        }
        all
    }
}

/// Make every definition except `entry` internal. This is LLVM's `internalize`
/// pass with `-internalize-public-api-list=<entry>`; that option is a
/// process-wide list, so the pass itself cannot select a second entry.
pub fn internalize(module: &Module<'_>, entry: &str) -> Result<(), String> {
    // SAFETY: every value comes from `module`, which outlives this call.
    unsafe {
        let raw = module.as_mut_ptr();
        let mut preserved: HashSet<Vec<u8>> = [
            entry,
            "llvm.used",
            "llvm.compiler.used",
            "llvm.global_ctors",
            "llvm.global_dtors",
            "llvm.global.annotations",
            "__stack_chk_fail",
            "__stack_chk_guard",
            "__llvm_rpc_server",
        ]
        .iter()
        .map(|name| name.as_bytes().to_vec())
        .collect();
        // Members of llvm.used may be referenced where LLVM cannot see.
        let used = LLVMGetNamedGlobal(raw, c"llvm.used".as_ptr());
        if !used.is_null() {
            let members = LLVMGetInitializer(used);
            for i in 0..LLVMGetNumOperands(members) as u32 {
                let mut member = LLVMGetOperand(members, i);
                while !LLVMIsAConstantExpr(member).is_null() {
                    member = LLVMGetOperand(member, 0);
                }
                preserved.insert(name(member));
            }
        }
        let keep = |value: LLVMValueRef| {
            LLVMIsDeclaration(value) != 0
                || LLVMGetLinkage(value) == LLVMLinkage::LLVMAvailableExternallyLinkage
                || LLVMGetDLLStorageClass(value)
                    == inkwell::llvm_sys::LLVMDLLStorageClass::LLVMDLLExportStorageClass
                || (!LLVMIsAGlobalVariable(value).is_null()
                    && LLVMIsExternallyInitialized(value) != 0)
        };
        let local = |value: LLVMValueRef| {
            matches!(
                LLVMGetLinkage(value),
                LLVMLinkage::LLVMInternalLinkage | LLVMLinkage::LLVMPrivateLinkage
            )
        };
        let objects = global_objects(raw);
        // A comdat is kept whole: external if any member must stay visible.
        let mut comdats: HashMap<LLVMComdatRef, (usize, bool)> = HashMap::new();
        for &object in &objects {
            let comdat = LLVMGetComdat(object);
            if !comdat.is_null() {
                let info = comdats.entry(comdat).or_default();
                info.0 += 1;
                info.1 |= keep(object) || (!local(object) && preserved.contains(&name(object)));
            }
        }
        let mut aliases = Vec::new();
        let mut alias = LLVMGetFirstGlobalAlias(raw);
        while !alias.is_null() {
            aliases.push(alias);
            alias = LLVMGetNextGlobalAlias(alias);
        }
        for &object in &objects {
            let comdat = LLVMGetComdat(object);
            if !comdat.is_null() {
                match comdats[&comdat] {
                    (_, true) => continue,
                    (1, false) => LLVMSetComdat(object, std::ptr::null_mut()),
                    // LLVM renames such a comdat; the C API cannot.
                    _ => {
                        return Err(format!(
                            "{} shares an internalizable comdat, which is unsupported",
                            String::from_utf8_lossy(&name(object))
                        ));
                    }
                }
            } else if keep(object) || local(object) || preserved.contains(&name(object)) {
                continue;
            }
            LLVMSetVisibility(object, LLVMVisibility::LLVMDefaultVisibility);
            LLVMSetLinkage(object, LLVMLinkage::LLVMInternalLinkage);
        }
        for alias in aliases {
            if keep(alias) || local(alias) || preserved.contains(&name(alias)) {
                continue;
            }
            LLVMSetVisibility(alias, LLVMVisibility::LLVMDefaultVisibility);
            LLVMSetLinkage(alias, LLVMLinkage::LLVMInternalLinkage);
        }
    }
    Ok(())
}

/// Set LLVM command-line options for the rest of this process. They cannot be
/// unset, which is why each kernel is optimized in a process of its own and a
/// stage sets an option only after every stage that needs the default has run.
pub fn set_options(options: &[&str]) {
    let arguments: Vec<CString> = std::iter::once("llvm-metal")
        .chain(options.iter().copied())
        .map(|argument| CString::new(argument).unwrap())
        .collect();
    let pointers: Vec<_> = arguments.iter().map(|argument| argument.as_ptr()).collect();
    // SAFETY: the argument strings outlive the call. LLVM exits on an unknown
    // option instead of silently running a different pipeline.
    unsafe {
        LLVMParseCommandLineOptions(pointers.len() as i32, pointers.as_ptr(), std::ptr::null())
    };
}

/// Run a textual pass pipeline with a target machine for the module's triple,
/// so cost models match `opt` on the same module.
pub fn passes(module: &Module<'_>, pipeline: &str) -> Result<(), String> {
    static TARGETS: std::sync::Once = std::sync::Once::new();
    let pipeline = CString::new(pipeline).unwrap();
    // SAFETY: module, triple and pipeline stay live; LLVM-owned messages, the
    // target machine and the options are disposed.
    unsafe {
        TARGETS.call_once(|| {
            LLVM_InitializeAllTargetInfos();
            LLVM_InitializeAllTargets();
            LLVM_InitializeAllTargetMCs();
        });
        let raw = module.as_mut_ptr();
        let triple = LLVMGetTarget(raw);
        let mut target = std::ptr::null_mut();
        let mut message = std::ptr::null_mut();
        if LLVMGetTargetFromTriple(triple, &mut target, &mut message) != 0 {
            let text = CStr::from_ptr(message).to_string_lossy().into_owned();
            LLVMDisposeMessage(message);
            return Err(format!("no LLVM target for the module triple: {text}"));
        }
        let machine = LLVMCreateTargetMachine(
            target,
            triple,
            c"".as_ptr(),
            c"".as_ptr(),
            LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
            LLVMRelocMode::LLVMRelocDefault,
            LLVMCodeModel::LLVMCodeModelDefault,
        );
        let options = LLVMCreatePassBuilderOptions();
        let error = LLVMRunPasses(raw, pipeline.as_ptr(), machine, options);
        LLVMDisposePassBuilderOptions(options);
        LLVMDisposeTargetMachine(machine);
        if error.is_null() {
            return Ok(());
        }
        let message = LLVMGetErrorMessage(error);
        let text = CStr::from_ptr(message).to_string_lossy().into_owned();
        LLVMDisposeErrorMessage(message);
        Err(text)
    }
}

/// Declarations the module still needs from outside, as `llvm-nm
/// --undefined-only` lists them: LLVM's own `llvm.` names are not symbols.
pub fn undefined(module: &Module<'_>) -> Vec<String> {
    // SAFETY: every value comes from `module`.
    unsafe {
        global_objects(module.as_mut_ptr())
            .into_iter()
            .filter(|&object| LLVMIsDeclaration(object) != 0)
            .map(|object| String::from_utf8_lossy(&name(object)).into_owned())
            .filter(|name| !name.starts_with("llvm."))
            .collect()
    }
}
