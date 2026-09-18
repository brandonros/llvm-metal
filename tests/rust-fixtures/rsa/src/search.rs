//! Small interfaces around the actual resumable miner, before its grid wrapper.
use vanity_logic::{
    modes::rsa_modulus::{self as rsa, Pair, SearchConfig, Task},
    search::{device_record::DeviceRecord, hex_pattern::HexPattern},
};
unsafe fn read<T: DeviceRecord>(p: *const u8) -> T {
    unsafe { p.cast::<T>().read_unaligned() }
}
unsafe fn write<T: DeviceRecord>(p: *mut u8, value: T) {
    unsafe { p.cast::<T>().write_unaligned(value) }
}
/// # Safety
/// Disjoint readable input[1080] and writable output[129].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_generate(input: *const u8, output: *mut u8) {
    unsafe {
        let config: SearchConfig = read(input);
        let id = input.add(1072).cast::<u64>().read_unaligned();
        let result = rsa::generate_p(&config, id);
        output.write(u8::from(result.is_some()));
        output
            .add(1)
            .cast::<[u8; 128]>()
            .write_unaligned(result.unwrap_or([0; 128]));
    }
}
/// # Safety
/// Disjoint readable input[1200] and writable output[257].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_progression(input: *const u8, output: *mut u8) {
    unsafe {
        let config: SearchConfig = read(input);
        let p = input.add(1072).cast::<[u8; 128]>().read_unaligned();
        let result = rsa::progression(&config, &p);
        output.write(u8::from(result.is_some()));
        let (first, count) = result.unwrap_or(([0; 128], [0; 128]));
        output.add(1).cast::<[u8; 128]>().write_unaligned(first);
        output.add(129).cast::<[u8; 128]>().write_unaligned(count);
    }
}
/// # Safety
/// Disjoint readable input[1984] and writable output[913].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_prepare(input: *const u8, output: *mut u8) {
    unsafe {
        let config: SearchConfig = read(input);
        let mut task: Task = read(input.add(1072));
        let status = match rsa::prepare_range(&config, &mut task) {
            Ok(false) => 0,
            Ok(true) => 1,
            Err(_) => 2,
        };
        output.write(status);
        write(output.add(1), task);
    }
}
/// # Safety
/// Disjoint readable input[1992] and writable output[1041].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_advance(input: *const u8, output: *mut u8) {
    unsafe {
        let config: SearchConfig = read(input);
        let mut task: Task = read(input.add(1072));
        let offset = input.add(1984).cast::<u32>().read_unaligned();
        let assigned = input.add(1988).cast::<u32>().read_unaligned();
        let q = rsa::q_at(&config, &task, offset);
        output.write(u8::from(q.is_some()));
        output
            .add(1)
            .cast::<[u8; 128]>()
            .write_unaligned(q.unwrap_or([0; 128]));
        rsa::finish_tile(&mut task, assigned);
        write(output.add(129), task);
    }
}
/// # Safety
/// Disjoint readable input[2516] and writable output[1209].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_mine(input: *const u8, output: *mut u8) {
    unsafe {
        let config: SearchConfig = read(input);
        let pattern: HexPattern = read(input.add(1072));
        let mut task: Task = read(input.add(1588));
        let start = input.add(2500).cast::<u64>().read_unaligned();
        let stride = input.add(2508).cast::<u32>().read_unaligned();
        let steps = input.add(2512).cast::<u32>().read_unaligned();
        let (counts, pair) = rsa::mine(&config, &pattern, &mut task, start, stride, steps);
        write(output, counts);
        write(output.add(32), task);
        output.add(944).write(u8::from(pair.is_some()));
        write(output.add(945), pair.unwrap_or(Pair::EMPTY));
    }
}
