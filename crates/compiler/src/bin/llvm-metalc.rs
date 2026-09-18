use inkwell::context::Context;
use llvm_metal_compiler::{parse_bitcode, parse_ir, require_entry};
use std::{env, error::Error, fs, path::Path, process::ExitCode};

const USAGE: &str = "Usage: llvm-metalc inspect <input.ll|input.bc> --entry <function>\n       llvm-metalc compile <input.ll|input.bc> <--interface json|--descriptor json|--entry name> --output <directory>\n       llvm-metalc extract <input.bc> --output <directory>";

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "{USAGE}\n\nCompile the supported integer/buffer profile to AIR and metallib. Compilation does not execute GPU code."
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

fn compile(
    path: &Path,
    interface_path: &Path,
    directory: &Path,
    mode: &str,
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
    let mut result = llvm_metal_compiler::compile::compile(&module, &interface)?;
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
