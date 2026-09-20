//! Temporary port check, enabled by LLVM_METAL_PARITY. Runs the C++ reference
//! and the Rust port on two clones and requires the same refusal or the same
//! printed IR. Deleted with the last native implementation.
use inkwell::module::Module;

pub(crate) fn enabled() -> bool {
    std::env::var_os("LLVM_METAL_PARITY").is_some()
}

pub(crate) fn check<'ctx>(
    pass: &str,
    module: &Module<'ctx>,
    reference: impl Fn(&Module<'ctx>) -> Result<(), String>,
    port: impl Fn(&Module<'ctx>) -> Result<(), String>,
) -> Result<(), String> {
    if !enabled() {
        return port(module);
    }
    // Cloning reorders use lists, so compare two clones, then lower the input.
    let (expected, actual) = (module.clone(), module.clone());
    let expected = reference(&expected).map(|()| expected.print_to_string().to_string());
    let actual = port(&actual).map(|()| actual.print_to_string().to_string());
    if expected != actual {
        panic!("{pass} parity mismatch\nC++: {expected:#?}\nRust: {actual:#?}");
    }
    port(module)
}
