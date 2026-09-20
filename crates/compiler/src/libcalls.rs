//! Scalar, address-space-independent implementations of supported C operations.
use inkwell::{
    AddressSpace,
    module::{Linkage, Module},
};

pub(crate) fn lower(module: &Module<'_>) -> Result<(), String> {
    for name in ["memcmp", "bcmp"] {
        let Some(function) = module.get_function(name) else {
            continue;
        };
        if function.count_basic_blocks() != 0 {
            continue;
        }
        let context = module.get_context();
        let ptr = context.ptr_type(AddressSpace::default());
        let expected = context
            .i32_type()
            .fn_type(&[ptr.into(), ptr.into(), context.i64_type().into()], false);
        if function.get_type() != expected || function.get_call_conventions() != 0 {
            return Err(format!(
                "invalid {name} declaration: expected C i32(ptr, ptr, i64)"
            ));
        }
        // Read no byte when length is zero. Compare unsigned bytes in order;
        // the sign of the first difference implements both C contracts. The
        // bounded loop never reads beyond the supplied ranges and supports
        // overlapping/read-only storage. Address spaces are inferred later.
        let ir = format!(
            r#"
            define i32 @{name}(ptr %a, ptr %b, i64 %n) {{
            entry:
              %empty = icmp eq i64 %n, 0
              br i1 %empty, label %equal, label %loop
            loop:
              %i = phi i64 [0, %entry], [%next, %advance]
              %ap = getelementptr i8, ptr %a, i64 %i
              %bp = getelementptr i8, ptr %b, i64 %i
              %av = load i8, ptr %ap, align 1
              %bv = load i8, ptr %bp, align 1
              %same = icmp eq i8 %av, %bv
              br i1 %same, label %advance, label %different
            advance:
              %next = add i64 %i, 1
              %done = icmp eq i64 %next, %n
              br i1 %done, label %equal, label %loop
            different:
              %aw = zext i8 %av to i32
              %bw = zext i8 %bv to i32
              %difference = sub i32 %aw, %bw
              ret i32 %difference
            equal:
              ret i32 0
            }}
        "#
        );
        let implementation = context
            .create_module_from_ir(
                inkwell::memory_buffer::MemoryBuffer::create_from_memory_range_copy(
                    ir.as_bytes(),
                    name,
                ),
            )
            .map_err(|e| e.to_string())?;
        implementation.set_triple(&module.get_triple());
        implementation.set_data_layout(&module.get_data_layout());
        module
            .link_in_module(implementation)
            .map_err(|e| e.to_string())?;
        module
            .get_function(name)
            .unwrap()
            .set_linkage(Linkage::Internal);
    }
    Ok(())
}
