use inkwell::context::Context;
use llvm_metal_compiler::{parse_bitcode, parse_ir, require_entry};
use std::{env, error::Error, fs, path::Path, process::ExitCode};

const USAGE: &str = "Usage: llvm-metalc inspect <input.ll|input.bc> --entry <function>";

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "{USAGE}\n\nInspect and verify LLVM structure. AIR compilation is not implemented yet."
        );
        return ExitCode::SUCCESS;
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
