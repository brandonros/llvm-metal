//! Optimize and compile one entry. LLVM's options are process-wide and cannot
//! be unset, so the builder runs this in a process per entry and each stage
//! sets an option only after every stage that needs LLVM's default has run.
use super::llvm;
use crate::{air::InliningPolicy, parse_bitcode};
use inkwell::{context::Context, module::Module};
use llvm_metal_abi::descriptor::Descriptor;
use std::{fs, path::Path, time::Instant};

/// Symbols a kernel may leave undefined: device operations the compiler lowers
/// and the two comparisons it implements. Anything else is a missing runtime.
pub const ALLOWED_UNDEFINED: [&str; 4] = [
    "llvm_metal.linear_thread_index",
    "llvm_metal.atomic_add_device_u32",
    "memcmp",
    "bcmp",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Select, prepare, inline, clean up and compile.
    Whole,
    /// `InliningPolicy::All` only: select and force-inline, writing `inlined.bc`.
    Inline,
    /// `InliningPolicy::All` only: clean up and compile `inlined.bc`. Forced
    /// inlining raises LLVM's inline threshold, which the cleanup must not see.
    Post,
}

pub struct Unit<'a> {
    pub input: &'a Path,
    pub descriptor: &'a Path,
    pub policy: InliningPolicy,
    pub stage: Stage,
    pub output: &'a Path,
}

const NO_VECTORIZE: [&str; 2] = ["-vectorize-slp=false", "-vectorize-loops=false"];

pub fn run(unit: &Unit<'_>) -> Result<(), String> {
    let descriptor: Descriptor =
        serde_json::from_slice(&fs::read(unit.descriptor).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let bytes = fs::read(unit.input).map_err(|e| e.to_string())?;
    // A module per stage: see `reread`.
    let contexts: [Context; 6] = std::array::from_fn(|_| Context::create());
    let module = parse_bitcode(&contexts[0], &bytes, "input").map_err(|e| e.to_string())?;
    let start = Instant::now();
    let mut inline_seconds = 0.0;
    let inlined = match (unit.policy, unit.stage) {
        (InliningPolicy::All, Stage::Inline) => {
            // Select and discard unrelated exported entries before forced inlining.
            llvm::internalize(&module, &descriptor.entry)?;
            llvm::set_options(&NO_VECTORIZE);
            llvm::set_options(&[
                "-force-remove-attribute=noinline",
                "-force-attribute=alwaysinline",
                "-inline-threshold=10000",
            ]);
            llvm::passes(
                &module,
                "globaldce,forceattrs,always-inline,default<O3>,globaldce,strip-dead-prototypes,verify",
            )?;
            if !module.write_bitcode_to_path(unit.output.join("inlined.bc")) {
                return Err("could not write inlined bitcode".into());
            }
            return timings(unit.output, start.elapsed().as_secs_f64(), 0.0, 0.0, 0.0);
        }
        (InliningPolicy::All, Stage::Post) => {
            llvm::set_options(&NO_VECTORIZE);
            module
        }
        (InliningPolicy::All, Stage::Whole) | (_, Stage::Inline | Stage::Post) => {
            return Err("forced inlining runs as two stages; other policies as one".into());
        }
        (policy, Stage::Whole) => {
            llvm::internalize(&module, &descriptor.entry)?;
            llvm::passes(
                &module,
                "globaldce,function(sroa,instcombine,simplifycfg,tailcallelim),globaldce,verify",
            )?;
            let module = reread(&contexts[1], &module)?;
            let prepared = match crate::calls::prepare(&module, &descriptor.entry, policy) {
                Ok(prepared) => prepared,
                Err(error) => {
                    reject(&module, unit.output, "rejected-prepare");
                    return Err(format!("preparation failed: {error}"));
                }
            };
            let module = reread(&contexts[2], &prepared)?;
            llvm::set_options(&NO_VECTORIZE);
            llvm::passes(
                &module,
                "always-inline,default<O3>,globaldce,strip-dead-prototypes,verify",
            )?;
            inline_seconds = start.elapsed().as_secs_f64();
            reread(&contexts[3], &module)?
        }
    };
    // kernel.bc is what gets compiled, not the module in memory.
    let bitcode = post_inline(inlined, &contexts[4])?.write_bitcode_to_memory();
    let module =
        parse_bitcode(&contexts[5], bitcode.as_slice(), "input").map_err(|e| e.to_string())?;
    let post_inline_seconds = start.elapsed().as_secs_f64() - inline_seconds;
    let undefined = llvm::undefined(&module);
    if undefined
        .iter()
        .any(|name| !ALLOWED_UNDEFINED.contains(&name.as_str()))
    {
        reject(&module, unit.output, "rejected");
        return Err(format!(
            "unresolved device runtime calls in {}: {}",
            descriptor.entry,
            undefined.join(", ")
        ));
    }
    fs::write(unit.output.join("kernel.bc"), bitcode.as_slice())
        .map_err(|e| format!("kernel.bc: {e}"))?;
    module
        .print_to_file(unit.output.join("kernel.ll"))
        .map_err(|e| e.to_string())?;
    let frontend_seconds = start.elapsed().as_secs_f64();
    let lowering = Instant::now();
    if let Err(error) = compile(&module, &descriptor, unit.policy, unit.output) {
        reject(&module, unit.output, "rejected");
        return Err(format!("compilation failed: {error}"));
    }
    timings(
        unit.output,
        inline_seconds,
        post_inline_seconds,
        frontend_seconds,
        lowering.elapsed().as_secs_f64(),
    )
}

/// Read a module back from its bitcode before the next stage, which is what
/// running each stage as a command did. LLVM's output depends on state a
/// module accumulates in memory: counters for value names, inferred attributes
/// on intrinsic declarations, use-list order. Bitcode resets them, so a stage's
/// result depends only on its input.
fn reread<'ctx>(context: &'ctx Context, module: &Module<'_>) -> Result<Module<'ctx>, String> {
    parse_bitcode(
        context,
        module.write_bitcode_to_memory().as_slice(),
        "input",
    )
    .map_err(|e| e.to_string())
}

fn post_inline<'ctx>(module: Module<'_>, context: &'ctx Context) -> Result<Module<'ctx>, String> {
    // Canonicalize counters and unroll before O3's loop pipeline. Running O3
    // directly on the partly unrolled Bech32 8-to-5-bit loops makes its
    // ScalarEvolution predicate analysis take minutes (see optimizer fixture).
    llvm::set_options(&["-unroll-threshold=1000"]);
    llvm::passes(
        &module,
        "function(loop-simplify,lcssa,loop(indvars),loop-unroll,sroa,\
         instcombine<verify-fixpoint;max-iterations=4>,simplifycfg),\
         default<O3>,globaldce,strip-dead-prototypes,verify",
    )?;
    // Large inlined hash blocks exceed GVN's default backward scan budget.
    // Bounded cleanup exposes stored SHA buffer lengths/domain tags before the
    // unresolved-runtime check. Never replace panic calls or assume their guards.
    let module = reread(context, &module)?;
    llvm::set_options(&["-memdep-block-scan-limit=10000"]);
    let cleanup =
        ["sroa,early-cse<memssa>,gvn,instcombine<verify-fixpoint;max-iterations=4>,simplifycfg"; 3]
            .join(",");
    llvm::passes(
        &module,
        &format!("function({cleanup}),globaldce,strip-dead-prototypes,verify"),
    )?;
    Ok(module)
}

fn compile(
    module: &Module<'_>,
    descriptor: &Descriptor,
    policy: InliningPolicy,
    output: &Path,
) -> Result<(), String> {
    crate::descriptor::validate_entry(module, descriptor)?;
    let mut result = crate::compile::compile_with_policy(module, &descriptor.interface()?, policy)?;
    result.bindings = descriptor.bindings()?;
    let write = |name: &str, bytes: &[u8]| {
        fs::write(output.join(name), bytes).map_err(|e| format!("{name}: {e}"))
    };
    write("kernel.air.ll", result.air_ir.as_bytes())?;
    write("kernel.air.bc", &result.air_bitcode)?;
    write(
        "kernel.bindings.json",
        &serde_json::to_vec_pretty(&result.bindings).map_err(|e| e.to_string())?,
    )?;
    write("kernel.metallib", &result.metallib)
}

// Best effort: the refused module is evidence, the error is what is reported.
fn reject(module: &Module<'_>, output: &Path, stem: &str) {
    module.write_bitcode_to_path(output.join(format!("{stem}.bc")));
    let _ = module.print_to_file(output.join(format!("{stem}.ll")));
}

fn timings(
    output: &Path,
    inline_seconds: f64,
    post_inline_seconds: f64,
    frontend_seconds: f64,
    lowering_seconds: f64,
) -> Result<(), String> {
    let report = serde_json::json!({
        "inline_seconds": inline_seconds,
        "post_inline_seconds": post_inline_seconds,
        "frontend_seconds": frontend_seconds,
        "lowering_seconds": lowering_seconds,
    });
    fs::write(output.join("unit.timings.json"), report.to_string()).map_err(|e| e.to_string())
}
