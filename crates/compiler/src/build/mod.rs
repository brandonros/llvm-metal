//! Build a kernel crate, or the archives Cargo produced for one, into a bundle
//! per entry: optimized bitcode, AIR, metallib, bindings and a build manifest.
pub mod llvm;
pub mod unit;

use crate::{air::InliningPolicy, parse_bitcode};
use inkwell::context::Context;
use llvm_metal_abi::descriptor::Descriptor;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

const TARGET: &str = "nvptx64-nvidia-cuda";
// External compiler overrides would make this a different producer.
const OVERRIDES: [&str; 6] = [
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "RUSTC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTC_BOOTSTRAP",
];
const ARTIFACTS: [&str; 7] = [
    "kernel.descriptor.json",
    "kernel.bc",
    "kernel.ll",
    "kernel.air.ll",
    "kernel.air.bc",
    "kernel.bindings.json",
    "kernel.metallib",
];

pub enum Input {
    /// A kernel crate; the builder compiles it for the device with Cargo.
    Crate {
        manifest: PathBuf,
        target_dir: PathBuf,
    },
    /// Rust archives whose members are device bitcode, in link order.
    Archives(Vec<PathBuf>),
}

pub struct Build {
    pub input: Input,
    pub output: PathBuf,
    pub policy: InliningPolicy,
    pub panics: unit::Panics,
    /// A group: `{"kernel": <entry>, "cases": [{"name", "entry", ...}]}`. Each
    /// case is built into `cases/<name>` and recorded in `kernel.group.json`.
    pub cases: Option<PathBuf>,
    /// Build one case of the group directly into `output`.
    pub case: Option<String>,
    /// The entry to build when there is no group and several are declared.
    pub entry: Option<String>,
    pub jobs: usize,
    /// Keep the linked module and every entry's intermediate files.
    pub keep_stage: bool,
    /// This program, which the builder runs again for each entry.
    pub program: PathBuf,
}

struct Case {
    name: Option<String>,
    entry: String,
    identity: Option<Value>,
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn digest_file(path: &Path) -> Result<String, String> {
    Ok(digest(
        &fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?,
    ))
}

fn capture(command: &mut Command) -> Result<String, String> {
    let output = command
        .stderr(std::process::Stdio::inherit())
        .output()
        .map_err(|e| format!("{command:?}: {e}"))?;
    if !output.status.success() {
        return Err(format!("{command:?} failed: {}", output.status));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

/// The producer's LLVM must not be newer than the linked LLVM, which reads its
/// bitcode, and the optimization pipeline is only checked within one major.
fn producer() -> Result<String, String> {
    for name in OVERRIDES {
        if std::env::var_os(name).is_some_and(|value| !value.is_empty()) {
            return Err(format!(
                "unset {name}: it would change the bitcode producer"
            ));
        }
    }
    let rustc = capture(Command::new("rustc").arg("-vV"))?;
    let version = rustc
        .lines()
        .find_map(|line| line.strip_prefix("LLVM version: "))
        .ok_or("rustc -vV reports no LLVM version")?;
    let mut parts = version.split('.').map(|part| part.parse::<u32>());
    let (Some(Ok(major)), Some(Ok(minor)), Some(Ok(patch))) =
        (parts.next(), parts.next(), parts.next())
    else {
        return Err(format!("unrecognized rustc LLVM version {version}"));
    };
    let linked = llvm::linked_version();
    if major != linked.0 || (minor, patch) > (linked.1, linked.2) {
        return Err(format!(
            "rustc produces LLVM {version} bitcode but this compiler links LLVM {}.{}.{}; \
             use a Rust release built on LLVM {} that is no newer",
            linked.0, linked.1, linked.2, linked.0
        ));
    }
    Ok(rustc)
}

/// Compile the crate for the device and return Cargo's current archives: never
/// a glob, which could pick up old builds.
fn device_archives(manifest: &Path, target_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let messages = capture(
        Command::new("cargo")
            .args([
                "rustc",
                "--locked",
                "--release",
                "--lib",
                "--target",
                TARGET,
            ])
            .arg("--manifest-path")
            .arg(manifest)
            .arg("--target-dir")
            .arg(target_dir)
            .arg("--config")
            .arg(format!(
                "target.{TARGET}.rustflags=[\"-Cno-vectorize-slp\", \"-Cno-vectorize-loops\"]"
            ))
            .args([
                "--message-format=json",
                "--",
                "--emit=llvm-bc",
                "-Cembed-bitcode=yes",
            ]),
    )?;
    // Cargo also reports host build dependencies; only the device's belong.
    let device = target_dir.join(TARGET);
    let mut archives = Vec::new();
    for line in messages.lines() {
        let message: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        if message["reason"] != "compiler-artifact" {
            continue;
        }
        for file in message["filenames"].as_array().into_iter().flatten() {
            let path = PathBuf::from(file.as_str().unwrap_or_default());
            if path.extension().is_some_and(|e| e == "rlib") && path.starts_with(&device) {
                archives.push(path);
            }
        }
    }
    if archives.is_empty() {
        return Err("Cargo reported no device archives".into());
    }
    // Cargo reports artifacts as jobs finish. Link order shapes the module, so
    // fix it: the same sources must give the same bundle.
    archives.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    archives.dedup();
    Ok(archives)
}

fn git(directory: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(arguments)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Every source file of the repository that rustc read for the archives, from
/// its dependency files, with the manifests and lockfile: what the bundle is
/// attributable to. Names are relative to the repository, so a bundle can be
/// checked in another checkout. The lockfile pins the sources outside it.
fn sources(manifest: &Path, archives: &[PathBuf]) -> Result<BTreeMap<String, PathBuf>, String> {
    let metadata: Value = serde_json::from_str(&capture(
        Command::new("cargo")
            .args(["metadata", "--locked", "--no-deps", "--format-version=1"])
            .arg("--manifest-path")
            .arg(manifest),
    )?)
    .map_err(|e| e.to_string())?;
    let root = PathBuf::from(
        metadata["workspace_root"]
            .as_str()
            .ok_or("no workspace root")?,
    );
    let mut files = vec![
        manifest.to_path_buf(),
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
    ];
    for archive in archives {
        // Cargo names a dependency's file after the crate and the final crate's
        // after its archive. Relative paths are relative to the workspace.
        let stem = archive
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let dependencies = [stem.trim_start_matches("lib"), stem]
            .map(|stem| archive.with_file_name(format!("{stem}.d")))
            .into_iter()
            .find(|path| path.is_file())
            .ok_or_else(|| format!("{}: no dependency file", archive.display()))?;
        let text = fs::read_to_string(&dependencies)
            .map_err(|e| format!("{}: {e}", dependencies.display()))?;
        let rule = text
            .lines()
            .find_map(|line| line.split_once(": "))
            .ok_or_else(|| format!("{}: no dependency rule", dependencies.display()))?;
        // Make escapes a space in a path with a backslash.
        let mut path = String::new();
        let mut characters = rule.1.chars().peekable();
        while let Some(character) = characters.next() {
            match character {
                '\\' if characters.peek() == Some(&' ') => path.push(characters.next().unwrap()),
                ' ' => files.push(root.join(std::mem::take(&mut path))),
                _ => path.push(character),
            }
        }
        files.push(root.join(path));
    }
    let repository = git(&root, &["rev-parse", "--show-toplevel"]).map_or(root, PathBuf::from);
    let repository = fs::canonicalize(&repository).map_err(|e| e.to_string())?;
    let mut named = BTreeMap::new();
    for file in files {
        let file = fs::canonicalize(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        if let Ok(name) = file.strip_prefix(&repository) {
            named.insert(name.display().to_string(), file.clone());
        }
    }
    Ok(named)
}

fn hashes(files: &BTreeMap<String, PathBuf>) -> Result<BTreeMap<String, String>, String> {
    files
        .iter()
        .map(|(name, path)| Ok((name.clone(), digest_file(path)?)))
        .collect()
}

/// Members of a Rust archive that are device bitcode, in archive order. The
/// other members are Rust metadata.
fn members(archive: &Path) -> Result<Vec<Vec<u8>>, String> {
    let file = fs::File::open(archive).map_err(|e| format!("{}: {e}", archive.display()))?;
    let mut reader = ar::Archive::new(file);
    let mut modules = Vec::new();
    while let Some(entry) = reader.next_entry() {
        let mut entry = entry.map_err(|e| format!("{}: {e}", archive.display()))?;
        let name = String::from_utf8_lossy(entry.header().identifier()).into_owned();
        if !name.ends_with(".o") {
            continue;
        }
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut data).map_err(|e| e.to_string())?;
        if !data.starts_with(b"BC\xc0\xde") {
            return Err(format!(
                "expected device bitcode in {}:{name}",
                archive.display()
            ));
        }
        modules.push(data);
    }
    Ok(modules)
}

fn cases(
    build: &Build,
    descriptors: &BTreeMap<String, Descriptor>,
) -> Result<(Option<String>, Vec<Case>), String> {
    let Some(path) = &build.cases else {
        if build.case.is_some() {
            return Err("--case requires --cases".into());
        }
        let entry = match &build.entry {
            Some(entry) => entry.clone(),
            None if descriptors.len() == 1 => descriptors.keys().next().unwrap().clone(),
            None => {
                return Err(format!(
                    "the module declares {} entries; pass --entry or --cases: {}",
                    descriptors.len(),
                    descriptors.keys().cloned().collect::<Vec<_>>().join(", ")
                ));
            }
        };
        return Ok((
            None,
            vec![Case {
                name: None,
                entry,
                identity: None,
            }],
        ));
    };
    let group: Value = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let kernel = group["kernel"]
        .as_str()
        .ok_or("cases: missing kernel")?
        .to_string();
    let mut all = Vec::new();
    for case in group["cases"].as_array().ok_or("cases: missing cases")? {
        let name = case["name"].as_str().ok_or("cases: a case has no name")?;
        let entry = case["entry"].as_str().ok_or("cases: a case has no entry")?;
        if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
            return Err(format!("cases: {name:?} cannot name a directory"));
        }
        if all
            .iter()
            .any(|other: &Case| other.name.as_deref() == Some(name))
        {
            return Err(format!("cases: duplicate case {name}"));
        }
        let mut identity = case.clone();
        identity["kernel"] = kernel.clone().into();
        all.push(Case {
            name: Some(name.into()),
            entry: entry.into(),
            identity: Some(identity),
        });
    }
    if all.is_empty() {
        return Err("cases: empty group".into());
    }
    // The device must declare exactly what the host inventory lists.
    let mut expected: Vec<&str> = all.iter().map(|case| case.entry.as_str()).collect();
    expected.push(&kernel);
    expected.sort();
    expected.dedup();
    if expected != descriptors.keys().map(String::as_str).collect::<Vec<_>>() {
        return Err(format!(
            "device descriptors differ from the group: declared {:?}, listed {expected:?}",
            descriptors.keys().collect::<Vec<_>>()
        ));
    }
    if let Some(selected) = &build.case {
        all.retain(|case| case.name.as_deref() == Some(selected));
        if all.is_empty() {
            return Err(format!("unknown case: {selected}"));
        }
        if build.output.join("kernel.group.json").exists() {
            return Err("--case output must not overwrite a group bundle".into());
        }
    }
    Ok((Some(kernel), all))
}

fn worker(build: &Build, stage: unit::Stage, input: &Path, directory: &Path) -> Result<(), String> {
    // A helper that cannot stay a call is known only once lowering has tried:
    // inline it and start the entry again. Each retry names a new helper.
    let mut forced: Vec<String> = Vec::new();
    loop {
        let error = match attempt(build, stage, input, directory, &forced) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        match error
            .lines()
            .find_map(|line| line.strip_prefix("force-inline: "))
        {
            Some(helper) if !forced.iter().any(|name| name == helper) && forced.len() < 64 => {
                forced.push(helper.to_owned());
            }
            _ => return Err(error),
        }
    }
}

fn attempt(
    build: &Build,
    stage: unit::Stage,
    input: &Path,
    directory: &Path,
    forced: &[String],
) -> Result<(), String> {
    let output = Command::new(&build.program)
        .arg("build-unit")
        .arg(input)
        .arg("--descriptor")
        .arg(directory.join("kernel.descriptor.json"))
        .arg("--output")
        .arg(directory)
        .args(["--inlining", policy_name(build.policy)])
        .args(
            forced
                .iter()
                .flat_map(|name| ["--force-inline", name.as_str()]),
        )
        .args([
            "--panics",
            match build.panics {
                unit::Panics::Refuse => "refuse",
                unit::Panics::Unreachable => "unreachable",
            },
        ])
        .args([
            "--stage",
            match stage {
                unit::Stage::Whole => "whole",
                unit::Stage::Inline => "inline",
                unit::Stage::Post => "post",
            },
        ])
        .output()
        .map_err(|e| format!("{}: {e}", build.program.display()))?;
    if output.status.success() {
        return Ok(());
    }
    Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
}

pub fn policy_name(policy: InliningPolicy) -> &'static str {
    match policy {
        InliningPolicy::All => "all",
        InliningPolicy::RetainScalar => "retain-scalar",
        InliningPolicy::Selective => "selective",
        InliningPolicy::Llvm => "llvm",
    }
}

fn read_timings(directory: &Path) -> Result<Value, String> {
    let path = directory.join("unit.timings.json");
    serde_json::from_slice(&fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| e.to_string())
}

/// Optimize and compile one entry in worker processes, leaving its artifacts
/// and timings in `directory`.
fn build_unit(build: &Build, stripped: &Path, directory: &Path) -> Result<Value, String> {
    if build.policy != InliningPolicy::All {
        worker(build, unit::Stage::Whole, stripped, directory)?;
        return read_timings(directory);
    }
    worker(build, unit::Stage::Inline, stripped, directory)?;
    let inline = read_timings(directory)?;
    worker(
        build,
        unit::Stage::Post,
        &directory.join("inlined.bc"),
        directory,
    )?;
    let mut timings = read_timings(directory)?;
    let inline = inline["inline_seconds"].as_f64().unwrap_or(0.0);
    timings["inline_seconds"] = inline.into();
    timings["frontend_seconds"] =
        (timings["frontend_seconds"].as_f64().unwrap_or(0.0) + inline).into();
    Ok(timings)
}

pub fn run(build: &Build) -> Result<Value, String> {
    let start = Instant::now();
    let rustc = producer()?;
    let (archives, files) = match &build.input {
        Input::Crate {
            manifest,
            target_dir,
        } => {
            let archives = device_archives(manifest, target_dir)?;
            let files = sources(manifest, &archives)?;
            (archives, files)
        }
        Input::Archives(archives) => {
            let named = archives
                .iter()
                .map(|path| (path.display().to_string(), path.clone()));
            (archives.clone(), named.collect())
        }
    };
    let source_hashes = hashes(&files)?;
    let source_revision = match &build.input {
        Input::Crate { manifest, .. } => git(
            manifest.parent().unwrap_or(Path::new(".")),
            &["rev-parse", "HEAD"],
        ),
        Input::Archives(_) => None,
    }
    .unwrap_or_else(|| "unknown".into());
    let rustc_seconds = start.elapsed().as_secs_f64();

    let link_start = Instant::now();
    let context = Context::create();
    let mut modules = Vec::new();
    for archive in &archives {
        for member in members(archive)? {
            modules.push(parse_bitcode(&context, &member, "member").map_err(|e| e.to_string())?);
        }
    }
    if modules.is_empty() {
        return Err("the archives contain no device bitcode".into());
    }
    let linked = llvm::link(&context, modules)?;
    let linked_hash = digest(linked.write_bitcode_to_memory().as_slice());
    let link_seconds = link_start.elapsed().as_secs_f64();
    let descriptor_start = Instant::now();
    let (stripped, descriptors) = crate::descriptor::extract(&linked)?;
    let (kernel, cases) = cases(build, &descriptors)?;
    let grouped = kernel.is_some() && build.case.is_none();

    fs::create_dir_all(&build.output).map_err(|e| e.to_string())?;
    if grouped {
        let pending = json!({"kernel": kernel, "source_sha256": source_hashes});
        fs::write(
            build.output.join("kernel.group.pending.json"),
            pending.to_string() + "\n",
        )
        .map_err(|e| e.to_string())?;
    }
    let temporary = tempfile::Builder::new()
        .prefix("stage-")
        .tempdir_in(&build.output)
        .map_err(|e| e.to_string())?;
    let (stage, _temporary) = if build.keep_stage {
        (temporary.keep(), None)
    } else {
        (temporary.path().to_path_buf(), Some(temporary))
    };
    if build.keep_stage && !linked.write_bitcode_to_path(stage.join("linked.bc")) {
        return Err("could not write the linked module".into());
    }
    let stripped_path = stage.join("stripped.bc");
    if !stripped.write_bitcode_to_path(&stripped_path) {
        return Err("could not write the stripped module".into());
    }
    let directories: Vec<PathBuf> = (0..cases.len())
        .map(|index| stage.join(format!("unit-{index}")))
        .collect();
    for (case, directory) in cases.iter().zip(&directories) {
        let descriptor = descriptors
            .get(&case.entry)
            .ok_or_else(|| format!("no descriptor for entry {}", case.entry))?;
        fs::create_dir(directory).map_err(|e| e.to_string())?;
        let text = serde_json::to_string_pretty(descriptor).map_err(|e| e.to_string())? + "\n";
        fs::write(directory.join("kernel.descriptor.json"), text).map_err(|e| e.to_string())?;
    }
    let descriptor_seconds = descriptor_start.elapsed().as_secs_f64();
    let shared_frontend_seconds = start.elapsed().as_secs_f64();

    // Entries are independent after linking; each is optimized in its own process.
    let next = AtomicUsize::new(0);
    let results: Vec<std::sync::Mutex<Option<Result<Value, String>>>> =
        cases.iter().map(|_| Default::default()).collect();
    std::thread::scope(|scope| {
        for _ in 0..build.jobs.clamp(1, cases.len()) {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::SeqCst);
                    if index >= cases.len() {
                        break;
                    }
                    let result = build_unit(build, &stripped_path, &directories[index]);
                    *results[index].lock().unwrap() = Some(result);
                }
            });
        }
    });

    let compiler_hash = digest_file(&build.program)?;
    let shared = json!({
        "rustc_seconds": rustc_seconds,
        "link_seconds": link_seconds,
        "descriptor_seconds": descriptor_seconds,
    });
    let (mut group_cases, mut frontend_total, mut lowering_total) =
        (Vec::new(), shared_frontend_seconds, 0.0);
    // Every entry is attempted, so one build reports every refusal.
    let mut refused = Vec::new();
    for ((case, directory), result) in cases.iter().zip(&directories).zip(results) {
        let destination = match (&case.name, grouped) {
            (Some(name), true) => build.output.join("cases").join(name),
            _ => build.output.clone(),
        };
        fs::create_dir_all(&destination).map_err(|e| e.to_string())?;
        let timings = match result.into_inner().unwrap() {
            Some(Ok(timings)) => timings,
            Some(Err(error)) => {
                // Keep what the worker refused next to where the bundle would be.
                for entry in fs::read_dir(directory)
                    .map_err(|e| e.to_string())?
                    .flatten()
                {
                    if entry.file_name().to_string_lossy().starts_with("rejected") {
                        let _ = fs::copy(entry.path(), destination.join(entry.file_name()));
                    }
                }
                refused.push(format!(
                    "{}: {error}",
                    case.name.as_deref().unwrap_or(&case.entry)
                ));
                continue;
            }
            None => return Err("a worker finished without a result".into()),
        };
        if source_hashes != hashes(&files)? {
            return Err("source changed during the build; rerun for an attributable bundle".into());
        }
        let frontend = timings["frontend_seconds"].as_f64().unwrap_or(0.0);
        let lowering = timings["lowering_seconds"].as_f64().unwrap_or(0.0);
        frontend_total += frontend;
        lowering_total += lowering;
        let mut artifacts = BTreeMap::new();
        for name in ARTIFACTS {
            artifacts.insert(name, digest_file(&directory.join(name))?);
        }
        let (major, minor, patch) = llvm::linked_version();
        let mut report = json!({
            "schema": 1,
            "rustc": rustc,
            "llvm": format!("{major}.{minor}.{patch}"),
            "source_revision": source_revision,
            "inlining": policy_name(build.policy),
            "compiler_sha256": compiler_hash,
            "linked_bitcode_sha256": linked_hash,
            "source_sha256": source_hashes,
            "artifacts": artifacts,
            "timings": {
                "rustc_seconds": rustc_seconds,
                "link_seconds": link_seconds,
                "descriptor_seconds": descriptor_seconds,
                "inline_seconds": timings["inline_seconds"],
                "post_inline_seconds": timings["post_inline_seconds"],
                "frontend_seconds": shared_frontend_seconds + frontend,
                "lowering_seconds": lowering,
            },
            "cache": "Cargo may reuse matching artifacts; linking is shared by a group; each entry is optimized and lowered separately",
            "metal_execution": "not run by builder",
        });
        if let Some(identity) = &case.identity {
            report["case"] = identity.clone();
            report["shared_frontend_seconds"] = shared_frontend_seconds.into();
        }
        let manifest = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())? + "\n";
        if let (Some(identity), true) = (&case.identity, grouped) {
            let mut listed = identity.clone();
            listed.as_object_mut().unwrap().remove("kernel");
            listed["directory"] = format!("cases/{}", case.name.as_deref().unwrap()).into();
            listed["manifest_sha256"] = digest(manifest.as_bytes()).into();
            group_cases.push(listed);
        }
        // Publish only a complete bundle; the manifest is written last.
        for name in ARTIFACTS {
            fs::rename(directory.join(name), destination.join(name)).map_err(|e| e.to_string())?;
        }
        fs::write(destination.join("kernel.build.json"), manifest).map_err(|e| e.to_string())?;
    }
    if !refused.is_empty() {
        return Err(format!(
            "{} of {} entries refused:\n{}",
            refused.len(),
            cases.len(),
            refused.join("\n")
        ));
    }
    if grouped {
        let mut timings = shared.clone();
        timings["frontend_seconds"] = frontend_total.into();
        timings["lowering_seconds"] = lowering_total.into();
        let group = json!({
            "schema": 1,
            "kind": "entry-group",
            "kernel": kernel,
            "cases": group_cases,
            "source_revision": source_revision,
            "source_sha256": source_hashes,
            "compiler_sha256": compiler_hash,
            "rustc": rustc,
            "timings": timings,
            "metal_execution": "not run by builder",
        });
        let text = serde_json::to_string_pretty(&group).map_err(|e| e.to_string())? + "\n";
        fs::write(build.output.join("kernel.group.json"), text).map_err(|e| e.to_string())?;
        fs::remove_file(build.output.join("kernel.group.pending.json"))
            .map_err(|e| e.to_string())?;
    }
    Ok(json!({
        "output": build.output,
        "frontend_seconds": frontend_total,
        "lowering_seconds": lowering_total,
        "wall_seconds": start.elapsed().as_secs_f64(),
        "cases": if kernel.is_some() { Some(cases.len()) } else { None },
    }))
}
