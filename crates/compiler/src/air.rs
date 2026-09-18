//! Initial integer/buffer AIR profile. Unsupported target semantics fail closed.
use inkwell::{
    AddressSpace,
    attributes::{Attribute, AttributeLoc},
    module::{Linkage, Module},
    targets::{TargetData, TargetTriple},
    types::{AnyTypeEnum, BasicTypeEnum},
    values::{AsValueRef, InstructionOpcode},
};
use llvm_metal_abi::{Access, Dispatch, KernelInterface, MetalBindings};
use std::ffi::{CStr, CString};

pub const AIR_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn check_type(
    ty: BasicTypeEnum<'_>,
    source: &TargetData,
    destination: &TargetData,
) -> Result<(), String> {
    match ty {
        BasicTypeEnum::IntType(t) if matches!(t.get_bit_width(), 1 | 8 | 16 | 32 | 64) => (),
        BasicTypeEnum::PointerType(t) if t.get_address_space() == AddressSpace::default() => (),
        BasicTypeEnum::ArrayType(t) => check_type(t.get_element_type(), source, destination)?,
        BasicTypeEnum::VectorType(t) => check_type(t.get_element_type(), source, destination)?,
        BasicTypeEnum::StructType(t) if !t.is_opaque() => {
            for (i, field) in t.get_field_types().into_iter().enumerate() {
                check_type(field, source, destination)?;
                if source.offset_of_element(&t, i as u32)
                    != destination.offset_of_element(&t, i as u32)
                {
                    return Err("struct field layout differs on AIR".into());
                }
            }
        }
        _ => {
            return Err(format!(
                "unsupported type in integer/buffer profile: {ty:?}"
            ));
        }
    }
    if source.get_abi_size(&ty) != destination.get_abi_size(&ty)
        || source.get_abi_alignment(&ty) != destination.get_abi_alignment(&ty)
    {
        return Err(format!("NVPTX/AIR layout mismatch for {ty:?}"));
    }
    Ok(())
}

fn validate_input(module: &Module<'_>) -> Result<(), String> {
    if module.get_triple().as_str().to_bytes() != b"nvptx64-nvidia-cuda" {
        return Err("expected the stock nvptx64-nvidia-cuda producer".into());
    }
    if module.get_data_layout().as_str().to_bytes().is_empty() {
        return Err("input must specify its data layout".into());
    }
    if module.get_global_metadata_size("nvvm.annotations") != 0 {
        return Err("NVVM kernel metadata is not supported".into());
    }
    let source = TargetData::create(
        module
            .get_data_layout()
            .as_str()
            .to_str()
            .map_err(|e| e.to_string())?,
    );
    let destination = TargetData::create(AIR_LAYOUT);
    fn constant_data(ty: BasicTypeEnum<'_>) -> bool {
        match ty {
            BasicTypeEnum::IntType(_) => true,
            BasicTypeEnum::ArrayType(t) => constant_data(t.get_element_type()),
            _ => false,
        }
    }
    for global in module.get_globals() {
        let initializer = global
            .get_initializer()
            .ok_or("external global is unsupported")?;
        if !global.is_constant()
            || !constant_data(initializer.get_type())
            || global.as_pointer_value().get_type().get_address_space() != AddressSpace::default()
        {
            return Err(
                "only constant integer/array globals in address space 0 are supported".into(),
            );
        }
        check_type(initializer.get_type(), &source, &destination)?;
    }
    for function in module.get_functions() {
        let function_name = function.get_name().to_string_lossy();
        let context = module.get_context();
        let operation_type = match function_name.as_ref() {
            "llvm_metal.linear_thread_index" => Some(context.i32_type().fn_type(&[], false)),
            "llvm_metal.atomic_add_device_u32" => Some(context.i32_type().fn_type(
                &[
                    context.ptr_type(AddressSpace::default()).into(),
                    context.i32_type().into(),
                ],
                false,
            )),
            _ => None,
        };
        if let Some(expected) = operation_type {
            if function.get_type() != expected
                || function.count_basic_blocks() != 0
                || function.get_call_conventions() != 0
            {
                return Err(format!(
                    "invalid device operation declaration: {function_name}"
                ));
            }
            continue;
        }
        if function.get_call_conventions() != 0
            && (function.get_call_conventions() != 8 || function.count_basic_blocks() == 0)
        {
            return Err("unsupported helper calling convention".into());
        }
        if function.count_basic_blocks() == 0 {
            let name = function.get_name().to_str().map_err(|e| e.to_string())?;
            if function.get_intrinsic_id() == 0
                || ![
                    "llvm.bswap.",
                    "llvm.ctlz.",
                    "llvm.cttz.",
                    "llvm.fshl.",
                    "llvm.fshr.",
                    "llvm.ucmp.",
                    "llvm.abs.",
                    "llvm.assume",
                    "llvm.umin.",
                    "llvm.umax.",
                    "llvm.smin.",
                    "llvm.smax.",
                    "llvm.lifetime.start",
                    "llvm.lifetime.end",
                    "llvm.memcpy.",
                    "llvm.memset.",
                    "llvm.experimental.noalias.scope.decl",
                ]
                .iter()
                .any(|p| name.starts_with(p))
            {
                return Err(format!("unsupported external operation: {name}"));
            }
        }
        for block in function.get_basic_blocks() {
            for instruction in block.get_instructions() {
                use InstructionOpcode::*;
                if !matches!(
                    instruction.get_opcode(),
                    Add | Sub
                        | Mul
                        | UDiv
                        | URem
                        | SDiv
                        | SRem
                        | Shl
                        | LShr
                        | AShr
                        | And
                        | Or
                        | Xor
                        | ICmp
                        | Select
                        | Freeze
                        | Phi
                        | Br
                        | Switch
                        | Return
                        | Call
                        | Alloca
                        | Load
                        | Store
                        | GetElementPtr
                        | BitCast
                        | Trunc
                        | ZExt
                        | SExt
                        | ExtractElement
                        | InsertElement
                        | ShuffleVector
                        | ExtractValue
                        | InsertValue
                ) {
                    return Err(format!(
                        "unsupported instruction: {:?}",
                        instruction.get_opcode()
                    ));
                }
                // Preserve private barriers and zeroization accesses. Do not generalize to
                // device pointers, memory-mapped I/O, or synchronization.
                unsafe extern "C" {
                    fn LLVMMetalPrivateMemory(
                        value: inkwell::llvm_sys::prelude::LLVMValueRef,
                    ) -> bool;
                }
                // SAFETY: the native query only inspects this live instruction.
                let private_volatile_memory =
                    unsafe { LLVMMetalPrivateMemory(instruction.as_value_ref()) };
                if (instruction.get_opcode() == Load || instruction.get_opcode() == Store)
                    && ((instruction.get_volatile().unwrap_or(false) && !private_volatile_memory)
                        || instruction
                            .get_atomic_ordering()
                            .map(|x| x != inkwell::AtomicOrdering::NotAtomic)
                            .unwrap_or(false))
                {
                    return Err("volatile/atomic memory requires explicit legalization".into());
                }
                if instruction.get_opcode() == Call
                    && inkwell::values::CallSiteValue::try_from(instruction)
                        .map_err(|_| "invalid call instruction")?
                        .get_called_fn_value()
                        .is_none()
                {
                    return Err("indirect calls and inline assembly are unsupported".into());
                }
                if instruction.get_opcode() == Call {
                    let callee = inkwell::values::CallSiteValue::try_from(instruction)
                        .unwrap()
                        .get_called_fn_value()
                        .unwrap();
                    let name = callee.get_name().to_string_lossy();
                    if ["llvm.umin.", "llvm.umax.", "llvm.smin.", "llvm.smax."]
                        .iter()
                        .any(|prefix| name.starts_with(prefix))
                    {
                        let value = instruction.get_operand(0).and_then(|o| o.value()).unwrap();
                        if !value.is_int_value()
                            || !matches!(
                                value.into_int_value().get_type().get_bit_width(),
                                8 | 16 | 32 | 64
                            )
                        {
                            return Err("integer min/max requires a scalar i8/i16/i32/i64".into());
                        }
                    }
                    if name.starts_with("llvm.ctlz.") || name.starts_with("llvm.cttz.") {
                        let value = instruction.get_operand(0).and_then(|o| o.value()).unwrap();
                        if !value.is_int_value()
                            || !matches!(
                                value.into_int_value().get_type().get_bit_width(),
                                8 | 16 | 32 | 64
                            )
                        {
                            return Err("zero counting requires a scalar i8/i16/i32/i64".into());
                        }
                    }
                    if name.starts_with("llvm.abs.") {
                        let value = instruction.get_operand(0).and_then(|o| o.value()).unwrap();
                        if !value.is_int_value()
                            || !matches!(
                                value.into_int_value().get_type().get_bit_width(),
                                8 | 16 | 32 | 64
                            )
                        {
                            return Err("absolute value requires a scalar i8/i16/i32/i64".into());
                        }
                    }
                    if name.starts_with("llvm.memcpy.") || name.starts_with("llvm.memset.") {
                        let volatile = instruction
                            .get_operand(3)
                            .and_then(|o| o.value())
                            .and_then(|v| v.into_int_value().get_zero_extended_constant());
                        if volatile != Some(0) {
                            return Err(
                                "volatile memory intrinsics require explicit legalization".into()
                            );
                        }
                    }
                }
                if instruction.get_type() != AnyTypeEnum::VoidType(module.get_context().void_type())
                {
                    check_type(
                        BasicTypeEnum::try_from(instruction.get_type())
                            .map_err(|_| "non-basic instruction type")?,
                        &source,
                        &destination,
                    )?;
                }
                for index in 0..instruction.get_num_operands() {
                    use inkwell::{
                        llvm_sys::{LLVMTypeKind, core::*},
                        values::AsValueRef,
                    };
                    // SAFETY: valid instruction and in-range operand index. Metadata
                    // operands cannot be represented by Inkwell's BasicValueEnum.
                    let metadata = unsafe {
                        LLVMGetTypeKind(LLVMTypeOf(LLVMGetOperand(
                            instruction.as_value_ref(),
                            index,
                        ))) == LLVMTypeKind::LLVMMetadataTypeKind
                    };
                    if !metadata {
                        if let Some(operand) =
                            instruction.get_operand(index).and_then(|o| o.value())
                        {
                            check_type(operand.get_type(), &source, &destination)?;
                        }
                    }
                }
                if instruction.get_opcode() == GetElementPtr {
                    check_type(
                        instruction
                            .get_gep_source_element_type()
                            .map_err(|e| e.to_string())?,
                        &source,
                        &destination,
                    )?;
                }
                if instruction.get_opcode() == Alloca {
                    check_type(
                        instruction
                            .get_allocated_type()
                            .map_err(|e| e.to_string())?,
                        &source,
                        &destination,
                    )?;
                }
            }
        }
    }
    Ok(())
}

// AIR has no upstream TargetMachine. LLVM supports null for these generic passes.
fn passes(module: &Module<'_>, pipeline: &str) -> Result<(), String> {
    use inkwell::llvm_sys::{error::*, transforms::pass_builder::*};
    let pipeline = CString::new(pipeline).unwrap();
    // SAFETY: module/pipeline remain live; options and error messages are disposed.
    unsafe {
        let options = LLVMCreatePassBuilderOptions();
        let error = LLVMRunPasses(
            module.as_mut_ptr(),
            pipeline.as_ptr(),
            std::ptr::null_mut(),
            options,
        );
        LLVMDisposePassBuilderOptions(options);
        if error.is_null() {
            return Ok(());
        }
        let message = LLVMGetErrorMessage(error);
        let text = CStr::from_ptr(message).to_string_lossy().into_owned();
        LLVMDisposeErrorMessage(message);
        Err(text)
    }
}

pub fn legalize<'ctx>(
    input: &Module<'ctx>,
    interface: &KernelInterface,
) -> Result<(Module<'ctx>, MetalBindings), String> {
    input.verify().map_err(|e| e.to_string())?;
    let bindings = interface.validate()?;
    let module = input.clone();
    unsafe extern "C" {
        fn LLVMMetalExpandPrivateVolatileCopies(module: inkwell::llvm_sys::prelude::LLVMModuleRef);
    }
    // SAFETY: verified disposable clone, bounded private copies only.
    unsafe {
        LLVMMetalExpandPrivateVolatileCopies(module.as_mut_ptr());
    }
    crate::wide::lower(&module)?;
    validate_input(&module)?;
    let context = module.get_context();
    let implementation = module
        .get_function(&interface.entry)
        .ok_or("entry not found")?;
    if implementation.count_basic_blocks() == 0 {
        return Err("entry is only a declaration".into());
    }
    if implementation.get_call_conventions() != 0 {
        return Err("entry must have C calling convention".into());
    }
    let generic_pointer = context.ptr_type(AddressSpace::default());
    let expected = context.void_type().fn_type(
        &vec![generic_pointer.into(); interface.arguments.len()],
        false,
    );
    if implementation.get_type() != expected {
        return Err("entry must return void and take exactly the declared buffer pointers".into());
    }
    let internal_name = format!("{}.__llvm_metal_impl", interface.entry);
    if module.get_function(&internal_name).is_some() {
        return Err("reserved implementation name collision".into());
    }
    implementation.as_global_value().set_name(&internal_name);
    for function in module
        .get_functions()
        .filter(|f| f.count_basic_blocks() != 0)
    {
        function.set_linkage(Linkage::Internal);
        function.remove_string_attribute(AttributeLoc::Function, "target-cpu");
        function.remove_string_attribute(AttributeLoc::Function, "target-features");
        function.remove_enum_attribute(
            AttributeLoc::Function,
            Attribute::get_named_enum_kind_id("noinline"),
        );
        function.add_attribute(
            AttributeLoc::Function,
            context.create_enum_attribute(Attribute::get_named_enum_kind_id("alwaysinline"), 0),
        );
        // Rust can also attach noinline to individual calls (e.g. subtle's
        // volatile barrier). Inlining preserves the volatile access itself.
        for block in function.get_basic_blocks() {
            for instruction in block.get_instructions() {
                if let Ok(call) = inkwell::values::CallSiteValue::try_from(instruction) {
                    if call
                        .get_called_fn_value()
                        .is_some_and(|f| f.count_basic_blocks() != 0)
                    {
                        call.remove_enum_attribute(
                            AttributeLoc::Function,
                            Attribute::get_named_enum_kind_id("noinline"),
                        );
                    }
                }
            }
        }
    }
    let device_pointer = context.ptr_type(AddressSpace::from(1));
    let indexed = module
        .get_function("llvm_metal.linear_thread_index")
        .is_some();
    if indexed && interface.dispatch != Dispatch::Grid1d {
        return Err("thread index requires an explicit 1D grid contract".into());
    }
    if interface.dispatch == Dispatch::Grid1d && !indexed {
        return Err("grid kernels must consume the thread index".into());
    }
    let mut entry_types = vec![device_pointer.into(); interface.arguments.len()];
    if indexed {
        entry_types.push(context.i32_type().into());
    }
    let entry = module.add_function(
        &interface.entry,
        context.void_type().fn_type(&entry_types, false),
        None,
    );
    let builder = context.create_builder();
    builder.position_at_end(context.append_basic_block(entry, "entry"));
    let mut args = Vec::new();
    for (value, arg) in entry.get_param_iter().zip(&interface.arguments) {
        value.set_name(&arg.name);
        args.push(
            builder
                .build_address_space_cast(value.into_pointer_value(), generic_pointer, "generic")
                .map_err(|e| e.to_string())?
                .into(),
        );
    }
    builder
        .build_call(implementation, &args, "")
        .map_err(|e| e.to_string())?;
    builder.build_return(None).map_err(|e| e.to_string())?;
    // Move immutable data into Metal constant address space. Preserve all values
    // and alignment; the native inference pass propagates this through GEPs.
    let globals: Vec<_> = module.get_globals().collect();
    for old in globals {
        let initializer = old.get_initializer().unwrap();
        let name = old.get_name().to_string_lossy().into_owned();
        old.set_name(&format!("{name}.source"));
        let new = module.add_global(initializer.get_type(), Some(AddressSpace::from(2)), &name);
        new.set_initializer(&initializer);
        new.set_constant(true);
        new.set_linkage(Linkage::Private);
        new.set_alignment(old.get_alignment());
        new.set_unnamed_address(old.get_unnamed_address());
        old.as_pointer_value().replace_all_uses_with(
            new.as_pointer_value()
                .const_address_space_cast(generic_pointer),
        );
        // SAFETY: all uses were replaced, and no references to old remain.
        unsafe {
            old.delete();
        }
    }
    module.set_triple(&TargetTriple::create("air64-apple-macosx13.0.0"));
    module.set_data_layout(&TargetData::create(AIR_LAYOUT).get_data_layout());
    // Expanded dalek scalar products need multiple InstCombine iterations.
    // Keep fixpoint verification enabled rather than suppressing its assertion.
    passes(
        &module,
        "always-inline,function(sroa,instcombine<verify-fixpoint;max-iterations=4>),globaldce",
    )?;
    for block in entry.get_basic_blocks() {
        let instructions: Vec<_> = block.get_instructions().collect();
        for instruction in instructions {
            if instruction.get_opcode() != InstructionOpcode::Call {
                continue;
            }
            let Some(callee) = inkwell::values::CallSiteValue::try_from(instruction)
                .unwrap()
                .get_called_fn_value()
            else {
                continue;
            };
            let name = callee.get_name().to_string_lossy();
            let replacement = if name == "llvm_metal.linear_thread_index" {
                Some(
                    entry
                        .get_nth_param(interface.arguments.len() as u32)
                        .unwrap(),
                )
            } else if name == "llvm_metal.atomic_add_device_u32" {
                builder.position_before(&instruction);
                let pointer = instruction
                    .get_operand(0)
                    .and_then(|o| o.value())
                    .unwrap()
                    .into_pointer_value();
                let pointer = builder
                    .build_address_space_cast(pointer, device_pointer, "atomic_device")
                    .map_err(|e| e.to_string())?;
                let value = instruction.get_operand(1).and_then(|o| o.value()).unwrap();
                let i32 = context.i32_type();
                let ty = i32.fn_type(
                    &[
                        device_pointer.into(),
                        i32.into(),
                        i32.into(),
                        i32.into(),
                        context.bool_type().into(),
                    ],
                    false,
                );
                let atomic = module
                    .get_function("air.atomic.global.add.u.i32")
                    .unwrap_or_else(|| {
                        module.add_function("air.atomic.global.add.u.i32", ty, None)
                    });
                // AIR 2.4 / Metal 3.0: relaxed=0, device scope=2, volatile=true.
                let call = builder
                    .build_call(
                        atomic,
                        &[
                            pointer.into(),
                            value.into(),
                            i32.const_zero().into(),
                            i32.const_int(2, false).into(),
                            context.bool_type().const_int(1, false).into(),
                        ],
                        "old",
                    )
                    .map_err(|e| e.to_string())?;
                Some(
                    call.try_as_basic_value()
                        .basic()
                        .ok_or("atomic must return old value")?,
                )
            } else {
                None
            };
            if let Some(replacement) = replacement {
                use inkwell::values::AsValueRef;
                // SAFETY: known operation signatures guarantee equal result types.
                unsafe {
                    inkwell::llvm_sys::core::LLVMReplaceAllUsesWith(
                        instruction.as_value_ref(),
                        replacement.as_value_ref(),
                    );
                }
                instruction.erase_from_basic_block();
            }
        }
    }
    unsafe extern "C" {
        fn LLVMMetalInferAddressSpaces(module: inkwell::llvm_sys::prelude::LLVMModuleRef);
        fn LLVMMetalRemoveCodegenFlags(module: inkwell::llvm_sys::prelude::LLVMModuleRef);
    }
    // SAFETY: verified live module, exclusive mutation during the native pass.
    unsafe {
        LLVMMetalRemoveCodegenFlags(module.as_mut_ptr());
        LLVMMetalInferAddressSpaces(module.as_mut_ptr());
    }
    passes(
        &module,
        "function(instcombine<verify-fixpoint;max-iterations=4>,simplifycfg),globaldce",
    )?;
    // These are optimization/lifetime hints, not device operations. AIR does not
    // implement them. Dropping both dynamic scope declarations and alias facts
    // avoids carrying Rust's scoped alias model into a different backend.
    for block in entry.get_basic_blocks() {
        let instructions: Vec<_> = block.get_instructions().collect();
        for instruction in instructions {
            use inkwell::values::AsValueRef;
            for kind in ["alias.scope", "noalias"] {
                // SAFETY: live instruction; a null node removes optional metadata.
                unsafe {
                    inkwell::llvm_sys::core::LLVMSetMetadata(
                        instruction.as_value_ref(),
                        context.get_kind_id(kind),
                        std::ptr::null_mut(),
                    );
                }
            }
            if instruction.get_opcode() == InstructionOpcode::Call {
                if let Some(function) = inkwell::values::CallSiteValue::try_from(instruction)
                    .unwrap()
                    .get_called_fn_value()
                {
                    let name = function.get_name().to_string_lossy();
                    let minmax = [
                        ("llvm.umin.", inkwell::IntPredicate::ULT),
                        ("llvm.umax.", inkwell::IntPredicate::UGT),
                        ("llvm.smin.", inkwell::IntPredicate::SLT),
                        ("llvm.smax.", inkwell::IntPredicate::SGT),
                    ]
                    .into_iter()
                    .find(|(prefix, _)| name.starts_with(prefix));
                    if let Some((_, predicate)) = minmax {
                        let a = instruction.get_operand(0).and_then(|o| o.value()).unwrap();
                        let b = instruction.get_operand(1).and_then(|o| o.value()).unwrap();
                        if !a.is_int_value() || !b.is_int_value() {
                            return Err("vector integer min/max is unsupported".into());
                        }
                        builder.position_before(&instruction);
                        let condition = builder
                            .build_int_compare(
                                predicate,
                                a.into_int_value(),
                                b.into_int_value(),
                                "minmax.compare",
                            )
                            .map_err(|e| e.to_string())?;
                        let value = builder
                            .build_select(condition, a, b, "minmax.value")
                            .map_err(|e| e.to_string())?;
                        // SAFETY: scalar operands and replacement have the intrinsic's type.
                        unsafe {
                            inkwell::llvm_sys::core::LLVMReplaceAllUsesWith(
                                instruction.as_value_ref(),
                                value.as_value_ref(),
                            );
                        }
                        instruction.erase_from_basic_block();
                        continue;
                    }
                    if name.starts_with("llvm.ctlz.") || name.starts_with("llvm.cttz.") {
                        let original = instruction
                            .get_operand(0)
                            .and_then(|o| o.value())
                            .unwrap()
                            .into_int_value();
                        let ty = original.get_type();
                        let width = ty.get_bit_width();
                        let trailing = name.starts_with("llvm.cttz.");
                        let poison_zero = instruction
                            .get_operand(1)
                            .and_then(|o| o.value())
                            .and_then(|v| v.into_int_value().get_zero_extended_constant())
                            .ok_or("zero-count poison flag must be constant")?;
                        builder.position_before(&instruction);
                        let mut value = original;
                        let mut count = ty.const_zero();
                        let mut step = width / 2;
                        while step != 0 {
                            let mask = if trailing {
                                (1u64 << step) - 1
                            } else {
                                (u64::MAX >> (64 - width)) << (width - step)
                            };
                            let masked = builder
                                .build_and(value, ty.const_int(mask, false), "count.mask")
                                .map_err(|e| e.to_string())?;
                            let empty = builder
                                .build_int_compare(
                                    inkwell::IntPredicate::EQ,
                                    masked,
                                    ty.const_zero(),
                                    "count.empty",
                                )
                                .map_err(|e| e.to_string())?;
                            let distance = ty.const_int(step as u64, false);
                            let shifted = if trailing {
                                builder.build_right_shift(value, distance, false, "count.shift")
                            } else {
                                builder.build_left_shift(value, distance, "count.shift")
                            }
                            .map_err(|e| e.to_string())?;
                            value = builder
                                .build_select(empty, shifted, value, "count.next")
                                .map_err(|e| e.to_string())?
                                .into_int_value();
                            let add = builder
                                .build_select(empty, distance, ty.const_zero(), "count.distance")
                                .map_err(|e| e.to_string())?
                                .into_int_value();
                            count = builder
                                .build_int_add(count, add, "count.sum")
                                .map_err(|e| e.to_string())?;
                            step /= 2;
                        }
                        let zero = builder
                            .build_int_compare(
                                inkwell::IntPredicate::EQ,
                                original,
                                ty.const_zero(),
                                "count.zero",
                            )
                            .map_err(|e| e.to_string())?;
                        let on_zero = if poison_zero != 0 {
                            ty.get_poison()
                        } else {
                            ty.const_int(width as u64, false)
                        };
                        let result = builder
                            .build_select(zero, on_zero, count, "count.result")
                            .map_err(|e| e.to_string())?;
                        // SAFETY: validated scalar intrinsic and replacement have equal types.
                        unsafe {
                            inkwell::llvm_sys::core::LLVMReplaceAllUsesWith(
                                instruction.as_value_ref(),
                                result.as_value_ref(),
                            );
                        }
                        instruction.erase_from_basic_block();
                        continue;
                    }
                    if name.starts_with("llvm.abs.") {
                        let value = instruction.get_operand(0).and_then(|o| o.value()).unwrap();
                        if !value.is_int_value() {
                            return Err("vector absolute value is unsupported".into());
                        }
                        let value = value.into_int_value();
                        let poison_min = instruction
                            .get_operand(1)
                            .and_then(|o| o.value())
                            .and_then(|v| v.into_int_value().get_zero_extended_constant())
                            .ok_or("absolute-value poison flag must be constant")?;
                        builder.position_before(&instruction);
                        let negative = builder
                            .build_int_compare(
                                inkwell::IntPredicate::SLT,
                                value,
                                value.get_type().const_zero(),
                                "abs.negative",
                            )
                            .map_err(|e| e.to_string())?;
                        // llvm.abs(INT_MIN, false) wraps to INT_MIN; true makes
                        // that case poison. NSW negation preserves this distinction.
                        let negated = if poison_min != 0 {
                            builder.build_int_nsw_neg(value, "abs.negated")
                        } else {
                            builder.build_int_neg(value, "abs.negated")
                        }
                        .map_err(|e| e.to_string())?;
                        let result = builder
                            .build_select(negative, negated, value, "abs.value")
                            .map_err(|e| e.to_string())?;
                        // SAFETY: the scalar replacement has the intrinsic's type.
                        unsafe {
                            inkwell::llvm_sys::core::LLVMReplaceAllUsesWith(
                                instruction.as_value_ref(),
                                result.as_value_ref(),
                            );
                        }
                        instruction.erase_from_basic_block();
                        continue;
                    }
                    if name.starts_with("llvm.ucmp.") {
                        let a = instruction
                            .get_operand(0)
                            .and_then(|o| o.value())
                            .ok_or("missing ucmp argument")?;
                        let b = instruction
                            .get_operand(1)
                            .and_then(|o| o.value())
                            .ok_or("missing ucmp argument")?;
                        if !a.is_int_value()
                            || !b.is_int_value()
                            || !instruction.get_type().is_int_type()
                        {
                            return Err("vector three-way comparisons are unsupported".into());
                        }
                        let ty = instruction.get_type().into_int_type();
                        builder.position_before(&instruction);
                        let less = builder
                            .build_int_compare(
                                inkwell::IntPredicate::ULT,
                                a.into_int_value(),
                                b.into_int_value(),
                                "less",
                            )
                            .map_err(|e| e.to_string())?;
                        let greater = builder
                            .build_int_compare(
                                inkwell::IntPredicate::UGT,
                                a.into_int_value(),
                                b.into_int_value(),
                                "greater",
                            )
                            .map_err(|e| e.to_string())?;
                        let positive = builder
                            .build_select(
                                greater,
                                ty.const_int(1, false),
                                ty.const_zero(),
                                "positive",
                            )
                            .map_err(|e| e.to_string())?;
                        let result = builder
                            .build_select(
                                less,
                                ty.const_all_ones(),
                                positive.into_int_value(),
                                "comparison",
                            )
                            .map_err(|e| e.to_string())?;
                        // SAFETY: the replacement has the intrinsic's result type,
                        // including when LLVM folds it to a constant.
                        unsafe {
                            inkwell::llvm_sys::core::LLVMReplaceAllUsesWith(
                                instruction.as_value_ref(),
                                result.as_value_ref(),
                            );
                        }
                        instruction.erase_from_basic_block();
                        continue;
                    }
                    if name.starts_with("llvm.lifetime.")
                        || name == "llvm.experimental.noalias.scope.decl"
                        || name == "llvm.assume"
                    {
                        instruction.erase_from_basic_block();
                    }
                }
            }
        }
    }
    passes(&module, "strip-dead-prototypes")?;
    if module
        .get_functions()
        .any(|f| f.count_basic_blocks() != 0 && f != entry)
    {
        return Err("helper could not be inlined into the Metal entry".into());
    }
    for block in entry.get_basic_blocks() {
        if let Some(cast) = block
            .get_instructions()
            .find(|i| i.get_opcode() == InstructionOpcode::AddrSpaceCast)
        {
            use inkwell::values::AnyValue;
            let user = cast
                .get_first_use()
                .map(|u| u.get_user().print_to_string().to_string())
                .unwrap_or_default();
            return Err(format!(
                "device pointer escaped address-space inference: {}; user: {user}",
                cast.print_to_string()
            ));
        }
    }
    let n = |x| context.i32_type().const_int(x, false).into();
    let s = |x| context.metadata_string(x).into();
    let mut arguments = Vec::new();
    for (index, arg) in interface.arguments.iter().enumerate() {
        let access = match arg.access {
            Access::Read => "air.read",
            Access::Write | Access::ReadWrite => "air.read_write",
        };
        // Opaque byte buffers; the interface carries the stronger size/alignment contract.
        arguments.push(
            context
                .metadata_node(&[
                    n(index as u64),
                    s("air.buffer"),
                    s("air.location_index"),
                    n(index as u64),
                    n(1),
                    s(access),
                    s("air.address_space"),
                    n(1),
                    s("air.arg_type_size"),
                    n(1),
                    s("air.arg_type_align_size"),
                    n(1),
                    s("air.arg_type_name"),
                    s("uchar"),
                    s("air.arg_name"),
                    s(&arg.name),
                ])
                .into(),
        );
    }
    if indexed {
        arguments.push(
            context
                .metadata_node(&[
                    n(interface.arguments.len() as u64),
                    s("air.thread_position_in_grid"),
                    s("air.arg_type_name"),
                    s("uint"),
                    s("air.arg_name"),
                    s("thread_index"),
                ])
                .into(),
        );
    }
    let kernel = context.metadata_node(&[
        entry.as_global_value().as_pointer_value().into(),
        context.metadata_node(&[]).into(),
        context.metadata_node(&arguments).into(),
    ]);
    module
        .add_global_metadata("air.kernel", &kernel)
        .map_err(|e| e.to_string())?;
    module
        .add_global_metadata("air.version", &context.metadata_node(&[n(2), n(4), n(0)]))
        .map_err(|e| e.to_string())?;
    module
        .add_global_metadata(
            "air.language_version",
            &context.metadata_node(&[s("Metal"), n(3), n(0), n(0)]),
        )
        .map_err(|e| e.to_string())?;
    // Resource limits also tell Apple's compiler to allocate constant-buffer
    // resources for program-scope tables. Match GPUCompiler's AIR module ABI.
    for (name, limit) in [
        ("air.max_device_buffers", 31),
        ("air.max_constant_buffers", 31),
        ("air.max_threadgroup_buffers", 31),
        ("air.max_textures", 128),
        ("air.max_read_write_textures", 8),
        ("air.max_samplers", 16),
    ] {
        module
            .add_global_metadata(
                "llvm.module.flags",
                &context.metadata_node(&[n(7), s(name), n(limit)]),
            )
            .map_err(|e| e.to_string())?;
    }
    module.verify().map_err(|e| e.to_string())?;
    Ok((module, bindings))
}
