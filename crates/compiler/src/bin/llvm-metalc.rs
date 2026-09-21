use inkwell::context::Context;
use llvm_metal_compiler::{parse_bitcode, parse_ir, require_entry};
use std::{
    env,
    error::Error,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

const USAGE: &str = "Usage: llvm-metalc inspect <input.ll|input.bc> --entry <function>\n       llvm-metalc compile <input.ll|input.bc> <--interface json|--descriptor json|--entry name> --output <directory> [--inlining all|retain-scalar|selective|llvm]\n       llvm-metalc prepare <input.bc> --entry <function> --output <output.bc> --inlining <policy>\n       llvm-metalc extract <input.bc> --output <directory>
       llvm-metalc build <--crate directory [--target-dir directory]|--rlib archive...> --output <directory> [--cases json [--case name]|--entry name] [--inlining policy] [--panics refuse|unreachable] [--jobs n] [--keep-stage]";

fn main() -> ExitCode {
    let mut args: Vec<_> = env::args_os().skip(1).collect();
    let mut policy = llvm_metal_compiler::air::InliningPolicy::default();
    if let Some(index) = args.iter().position(|arg| arg == "--inlining") {
        if !args.first().is_some_and(|arg| {
            arg == "compile" || arg == "prepare" || arg == "build" || arg == "build-unit"
        }) {
            eprintln!("--inlining is supported only by build, compile and prepare");
            return ExitCode::from(2);
        }
        policy = match args.get(index + 1).and_then(|arg| arg.to_str()) {
            Some("all") => llvm_metal_compiler::air::InliningPolicy::All,
            Some("retain-scalar") => llvm_metal_compiler::air::InliningPolicy::RetainScalar,
            Some("selective") => llvm_metal_compiler::air::InliningPolicy::Selective,
            Some("llvm") => llvm_metal_compiler::air::InliningPolicy::Llvm,
            _ => {
                eprintln!("--inlining requires all, retain-scalar, selective, or llvm");
                return ExitCode::from(2);
            }
        };
        args.drain(index..index + 2);
    }
    if args
        .first()
        .is_some_and(|arg| arg == "build" || arg == "build-unit")
    {
        let result = if args[0] == "build" {
            build(&args[1..], policy)
        } else {
            build_unit(&args[1..], policy)
        };
        return match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("build failed: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 6 && args[0] == "prepare" && args[2] == "--entry" && args[4] == "--output" {
        let result = (|| -> Result<(), Box<dyn Error>> {
            let context = Context::create();
            let module = parse_bitcode(&context, &fs::read(&args[1])?, "prepare input")?;
            let entry = args[3].to_str().ok_or("entry must be UTF-8")?;
            let module = llvm_metal_compiler::calls::prepare(&module, entry, policy)?;
            if !module.write_bitcode_to_path(Path::new(&args[5])) {
                return Err("could not write prepared bitcode".into());
            }
            Ok(())
        })();
        return match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("preparation failed: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "{USAGE}\n\nCompile the supported integer/buffer profile to AIR and metallib. Selective inlining is the default; use --inlining all to force full inlining. Compilation does not execute GPU code."
        );
        return ExitCode::SUCCESS;
    }
    if args.len() == 6
        && args[0] == "compile"
        && (args[2] == "--interface" || args[2] == "--descriptor" || args[2] == "--entry")
        && args[4] == "--output"
    {
        return match compile(
            Path::new(&args[1]),
            Path::new(&args[3]),
            Path::new(&args[5]),
            args[2].to_str().unwrap(),
            policy,
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("compilation failed: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 4 && args[0] == "extract" && args[2] == "--output" {
        return match extract(Path::new(&args[1]), Path::new(&args[3])) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("descriptor extraction failed: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() != 4 || args[0] != "inspect" || args[2] != "--entry" {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let Some(entry) = args[3].to_str() else {
        eprintln!("entry name must be UTF-8");
        return ExitCode::from(2);
    };
    match inspect(Path::new(&args[1]), entry) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}: {error}", Path::new(&args[1]).display());
            ExitCode::FAILURE
        }
    }
}

/// Split `--flag value` arguments; a flag may repeat.
fn flags(args: &[OsString], known: &[&str]) -> Result<Vec<(String, PathBuf)>, String> {
    let mut pairs = Vec::new();
    for pair in args.chunks(2) {
        let flag = pair[0].to_str().unwrap_or_default();
        if !known.contains(&flag) || pair.len() != 2 {
            return Err(format!("unexpected argument {:?}\n{USAGE}", pair[0]));
        }
        pairs.push((flag.to_string(), PathBuf::from(&pair[1])));
    }
    Ok(pairs)
}

fn build(
    args: &[OsString],
    policy: llvm_metal_compiler::air::InliningPolicy,
) -> Result<(), String> {
    use llvm_metal_compiler::build::{Build, Input, run};
    let known = [
        "--crate",
        "--target-dir",
        "--rlib",
        "--output",
        "--cases",
        "--case",
        "--entry",
        "--jobs",
    ];
    let mut args = args.to_vec();
    let panics = panics(&mut args)?;
    let keep_stage = args
        .iter()
        .position(|arg| arg == "--keep-stage")
        .map(|i| args.remove(i));
    let pairs = flags(&args, &known)?;
    let all = |flag: &str| -> Vec<PathBuf> {
        let matching = pairs.iter().filter(|(name, _)| name == flag);
        matching.map(|(_, value)| value.clone()).collect()
    };
    // Cargo reports absolute paths, which the builder compares with these.
    let absolute = |path: PathBuf| std::path::absolute(&path).map_err(|e| e.to_string());
    let one = |flag: &str| -> Result<Option<PathBuf>, String> {
        let mut values = all(flag).into_iter();
        match (values.next(), values.next()) {
            (Some(value), None) if flag != "--case" && flag != "--entry" && flag != "--jobs" => {
                absolute(value).map(Some)
            }
            (value, None) => Ok(value),
            _ => Err(format!("{flag} given more than once")),
        }
    };
    let text = |flag: &str| -> Result<Option<String>, String> {
        one(flag)?
            .map(|value| {
                value
                    .into_os_string()
                    .into_string()
                    .map_err(|_| format!("{flag} must be UTF-8"))
            })
            .transpose()
    };
    let archives = all("--rlib");
    let input = match (one("--crate")?, archives.is_empty()) {
        (Some(directory), true) => {
            let directory = fs::canonicalize(&directory)
                .map_err(|e| format!("{}: {e}", directory.display()))?;
            Input::Crate {
                target_dir: one("--target-dir")?
                    .unwrap_or_else(|| directory.join("target/llvm-metal")),
                manifest: directory.join("Cargo.toml"),
            }
        }
        (None, false) => Input::Archives(archives),
        _ => return Err(format!("pass either --crate or --rlib\n{USAGE}")),
    };
    let jobs = match text("--jobs")? {
        Some(jobs) => jobs.parse().map_err(|_| "--jobs requires a number")?,
        None => std::thread::available_parallelism().map_or(1, usize::from),
    };
    let summary = run(&Build {
        input,
        output: one("--output")?.ok_or("--output is required")?,
        policy,
        panics,
        cases: one("--cases")?,
        case: text("--case")?,
        entry: text("--entry")?,
        jobs,
        keep_stage: keep_stage.is_some(),
        program: env::current_exe().map_err(|e| e.to_string())?,
    })?;
    println!("{summary}");
    Ok(())
}

/// Remove and parse `--panics refuse|unreachable`; refusing is the default.
fn panics(args: &mut Vec<OsString>) -> Result<llvm_metal_compiler::build::unit::Panics, String> {
    use llvm_metal_compiler::build::unit::Panics;
    let Some(index) = args.iter().position(|arg| arg == "--panics") else {
        return Ok(Panics::Refuse);
    };
    args.remove(index);
    let value = (index < args.len()).then(|| args.remove(index));
    match value.as_ref().and_then(|value| value.to_str()) {
        Some("refuse") => Ok(Panics::Refuse),
        Some("unreachable") => Ok(Panics::Unreachable),
        _ => Err("--panics requires refuse or unreachable".into()),
    }
}

/// One entry of a build, in a process of its own; see `build::unit`.
fn build_unit(
    args: &[OsString],
    policy: llvm_metal_compiler::air::InliningPolicy,
) -> Result<(), String> {
    use llvm_metal_compiler::build::unit::{Stage, Unit, run};
    let mut args = args.to_vec();
    let panics = panics(&mut args)?;
    let (input, rest) = args
        .split_first()
        .ok_or("build-unit is internal to build")?;
    let pairs = flags(
        rest,
        &["--descriptor", "--output", "--stage", "--force-inline"],
    )?;
    let force_inline: Vec<String> = pairs
        .iter()
        .filter(|(name, _)| name == "--force-inline")
        .map(|(_, value)| value.to_string_lossy().into_owned())
        .collect();
    let get = |flag: &str| {
        pairs
            .iter()
            .find(|(name, _)| name == flag)
            .map(|(_, value)| value.as_path())
            .ok_or(format!("build-unit requires {flag}"))
    };
    let stage = match get("--stage")?.to_str() {
        Some("whole") => Stage::Whole,
        Some("inline") => Stage::Inline,
        Some("post") => Stage::Post,
        _ => return Err("--stage requires whole, inline, or post".into()),
    };
    run(&Unit {
        input: Path::new(input),
        descriptor: get("--descriptor")?,
        policy,
        panics,
        force_inline: &force_inline,
        stage,
        output: get("--output")?,
    })
}

fn compile(
    path: &Path,
    interface_path: &Path,
    directory: &Path,
    mode: &str,
    policy: llvm_metal_compiler::air::InliningPolicy,
) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let context = Context::create();
    let module = match path.extension().and_then(|e| e.to_str()) {
        Some("ll") => parse_ir(&context, &bytes, "input")?,
        Some("bc") => parse_bitcode(&context, &bytes, "input")?,
        _ => return Err("expected a .ll or .bc input".into()),
    };
    let (module, descriptor) = if mode == "--entry" {
        let (module, mut descriptors) = llvm_metal_compiler::descriptor::extract(&module)?;
        let entry = interface_path.to_str().ok_or("entry must be UTF-8")?;
        let d = descriptors
            .remove(entry)
            .ok_or("no descriptor for selected entry")?;
        (module, Some(d))
    } else if mode == "--descriptor" {
        (
            module,
            Some(serde_json::from_slice::<
                llvm_metal_abi::descriptor::Descriptor,
            >(&fs::read(interface_path)?)?),
        )
    } else {
        (module, None)
    };
    let interface = if let Some(d) = &descriptor {
        llvm_metal_compiler::descriptor::validate_entry(&module, d)?;
        d.interface()?
    } else {
        serde_json::from_slice(&fs::read(interface_path)?)?
    };
    let mut result =
        llvm_metal_compiler::compile::compile_with_policy(&module, &interface, policy)?;
    if let Some(d) = descriptor {
        result.bindings = d.bindings()?;
    }
    fs::create_dir_all(directory)?;
    fs::write(directory.join("kernel.air.ll"), result.air_ir)?;
    fs::write(directory.join("kernel.air.bc"), result.air_bitcode)?;
    fs::write(
        directory.join("kernel.bindings.json"),
        serde_json::to_vec_pretty(&result.bindings)?,
    )?;
    fs::write(directory.join("kernel.metallib"), result.metallib)?;
    println!(
        "Compiled {} into {}; GPU execution has not been checked.",
        result.bindings.entry,
        directory.display()
    );
    Ok(())
}

fn inspect(path: &Path, entry: &str) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let context = Context::create();
    let name = path.to_string_lossy();
    let module = match path.extension().and_then(|extension| extension.to_str()) {
        Some("ll") => parse_ir(&context, &bytes, &name)?,
        Some("bc") => parse_bitcode(&context, &bytes, &name)?,
        _ => return Err("expected a .ll or .bc input".into()),
    };
    require_entry(&module, entry)?;
    let mut functions: Vec<_> = module
        .get_functions()
        .map(|function| {
            let kind = if function.count_basic_blocks() == 0 {
                "declaration"
            } else {
                "definition"
            };
            (function.get_name().to_string_lossy().into_owned(), kind)
        })
        .collect();
    functions.sort();
    println!("LLVM structure verified; entry: {entry}");
    for (name, kind) in functions {
        println!("{kind}: {name}");
    }
    println!("Metal legality and GPU execution have not been checked.");
    Ok(())
}

fn extract(path: &Path, directory: &Path) -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = parse_bitcode(&context, &fs::read(path)?, "descriptor input")?;
    let (module, descriptors) = llvm_metal_compiler::descriptor::extract(&module)?;
    fs::create_dir_all(directory)?;
    module.write_bitcode_to_path(&directory.join("stripped.bc"));
    fs::write(
        directory.join("descriptors.json"),
        serde_json::to_vec_pretty(&descriptors)?,
    )?;
    println!("Extracted {} device descriptors", descriptors.len());
    Ok(())
}
