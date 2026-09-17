//! Native execution validates the fixture oracles, NOT AIR translation or GPU support.
//! Only these trusted, reviewed fixtures may be JIT-executed; the CLI never executes input.

use inkwell::{
    OptimizationLevel,
    context::Context,
    module::Module,
    targets::{InitializationConfig, Target},
};
use llvm_metal_compiler::{parse_bitcode, parse_ir};
use std::sync::OnceLock;

mod support;

fn native_target() {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| Target::initialize_native(&InitializationConfig::default()))
        .as_ref()
        .unwrap();
}

fn load<'ctx>(context: &'ctx Context, name: &str, round_trip: bool) -> Module<'ctx> {
    let module = parse_ir(context, &support::fixture(name), name).unwrap();
    if round_trip {
        let bytes = module.write_bitcode_to_memory();
        parse_bitcode(context, bytes.as_slice(), name).unwrap()
    } else {
        module
    }
}

fn scalar_cases(name: &str, cases: &[(u64, u64, u64)]) {
    native_target();
    for round_trip in [false, true] {
        for level in [OptimizationLevel::None, OptimizationLevel::Aggressive] {
            let context = Context::create();
            let module = load(&context, name, round_trip);
            let i64 = context.i64_type();
            assert_eq!(
                module.get_function("fixture").unwrap().get_type(),
                i64.fn_type(&[i64.into(), i64.into()], false)
            );
            let engine = module.create_jit_execution_engine(level).unwrap();
            // SAFETY: the exact signature is checked above. Only trusted scalar fixtures
            // with bounded loop inputs are executed, while their module/engine stay alive.
            let function = unsafe {
                engine
                    .get_function::<unsafe extern "C" fn(u64, u64) -> u64>("fixture")
                    .unwrap()
            };
            for &(a, b, expected) in cases {
                let actual = unsafe { function.call(a, b) };
                assert_eq!(
                    actual, expected,
                    "{name}: ({a:#x}, {b:#x}), bitcode={round_trip}, optimization={level:?}"
                );
            }
        }
    }
}

fn pairs() -> Vec<(u64, u64)> {
    let edges = [
        0,
        1,
        31,
        32,
        63,
        64,
        65,
        u32::MAX as u64,
        1 << 32,
        1 << 63,
        u64::MAX,
    ];
    let mut values: Vec<_> = edges
        .iter()
        .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
        .collect();
    // Deterministic corpus extension, never dependent on ambient randomness.
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    for _ in 0..128 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let a = state;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        values.push((a, state));
    }
    values
}

#[test]
fn scalar_fixture_oracles_cover_integer_edges_and_control_flow() {
    type Oracle = fn(u64, u64) -> u64;
    let fixtures: &[(&str, Oracle)] = &[
        ("01-wrapping-add.ll", u64::wrapping_add),
        ("02-rotate.ll", |a, b| a.rotate_left((b & 63) as u32)),
        ("03-clz.ll", |a, _| a.leading_zeros() as u64),
        ("04-add-carry.ll", |a, b| {
            (a as u32 as u64) + (b as u32 as u64)
        }),
        ("05-branch-phi.ll", u64::abs_diff),
        ("07-helper-call.ll", |a, b| {
            a.wrapping_mul(3) ^ b.wrapping_mul(3)
        }),
        ("10-sha256-sigma0.ll", |a, _| {
            let x = a as u32;
            (x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)) as u64
        }),
    ];
    for &(name, oracle) in fixtures {
        let cases: Vec<_> = pairs()
            .into_iter()
            .map(|(a, b)| (a, b, oracle(a, b)))
            .collect();
        scalar_cases(name, &cases);
    }
}

#[test]
fn loop_fixture_covers_zero_one_and_many_iterations_with_overflow() {
    let mut cases = Vec::new();
    for n in [0, 1, 2, 7, 32, 256] {
        for initial in [0u64, 1, u64::MAX] {
            let expected = (0..n).fold(initial, u64::wrapping_add);
            cases.push((n, initial, expected));
        }
    }
    scalar_cases("06-loop-phi.ll", &cases);
}

#[test]
fn sha256_sigma0_has_fixed_known_answers() {
    // Fixed answers in addition to differential testing; upper bits are discarded.
    scalar_cases(
        "10-sha256-sigma0.ll",
        &[
            (0, 0, 0),
            (1, 0, 0x0200_4000),
            (u32::MAX as u64, 0, 0x1fff_ffff),
            (0x0000_0001_0000_0000, 0, 0),
        ],
    );
}

#[test]
fn pointer_helper_writes_exactly_one_element() {
    native_target();
    for round_trip in [false, true] {
        let context = Context::create();
        let module = load(&context, "08-pointer-helper.ll", round_trip);
        let pointer = context.ptr_type(Default::default());
        assert_eq!(
            module.get_function("fixture").unwrap().get_type(),
            context.void_type().fn_type(
                &[
                    pointer.into(),
                    context.i64_type().into(),
                    context.i32_type().into()
                ],
                false
            )
        );
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        // SAFETY: signature checked above, and all pointers/indices below remain within
        // the allocated array. No references alias the buffer during the call.
        let function = unsafe {
            engine
                .get_function::<unsafe extern "C" fn(*mut u32, u64, u32)>("fixture")
                .unwrap()
        };
        for index in [0usize, 1, 7, 14] {
            let mut actual = [0xa5a5_a5a5; 17];
            let mut expected = actual;
            expected[index + 1] = 0x1234_5678;
            unsafe {
                function.call(actual.as_mut_ptr().add(1), index as u64, 0x1234_5678);
            }
            assert_eq!(actual, expected, "index={index}, bitcode={round_trip}");
        }
    }
}

#[test]
fn byte_copy_preserves_guard_bytes_and_source() {
    native_target();
    for round_trip in [false, true] {
        let context = Context::create();
        let module = load(&context, "09-byte-copy.ll", round_trip);
        let pointer = context.ptr_type(Default::default());
        assert_eq!(
            module.get_function("fixture").unwrap().get_type(),
            context.void_type().fn_type(
                &[pointer.into(), pointer.into(), context.i64_type().into()],
                false
            )
        );
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        // SAFETY: signature checked above. Source/destination are separate allocations;
        // count never exceeds their bounds, and both pointers are valid for count = 0.
        let function = unsafe {
            engine
                .get_function::<unsafe extern "C" fn(*mut u8, *const u8, u64)>("fixture")
                .unwrap()
        };
        for count in [0usize, 1, 3, 7, 16] {
            let source: [u8; 18] = std::array::from_fn(|i| i as u8);
            let original = source;
            let mut actual = [0xa5u8; 18];
            let mut expected = actual;
            expected[1..1 + count].copy_from_slice(&source[1..1 + count]);
            unsafe {
                function.call(
                    actual.as_mut_ptr().add(1),
                    source.as_ptr().add(1),
                    count as u64,
                );
            }
            assert_eq!(actual, expected, "count={count}, bitcode={round_trip}");
            assert_eq!(source, original);
        }
    }
}
