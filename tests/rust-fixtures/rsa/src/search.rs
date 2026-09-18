//! Runtime-input interfaces around independent RSA candidates, before grid dispatch.
use vanity_logic::{
    modes::rsa_modulus::{self as rsa, Pair, SearchConfig},
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
/// Disjoint readable input[1208] and writable output[129].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_sample_q(input: *const u8, output: *mut u8) {
    unsafe {
        let config: SearchConfig = read(input);
        let p = input.add(1072).cast::<[u8; 128]>().read_unaligned();
        let id = input.add(1200).cast::<u64>().read_unaligned();
        let (status, q) = match rsa::generate_q(&config, &p, id) {
            Ok(Some(q)) => (1, q),
            Ok(None) => (0, [0; 128]),
            Err(_) => (2, [0; 128]),
        };
        output.write(status);
        output.add(1).cast::<[u8; 128]>().write_unaligned(q);
    }
}
/// # Safety
/// Disjoint readable input[772] and writable output[1].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_eligible_pair(input: *const u8, output: *mut u8) {
    unsafe {
        let p = input.cast::<[u8; 128]>().read_unaligned();
        let q = input.add(128).cast::<[u8; 128]>().read_unaligned();
        let pattern: HexPattern = read(input.add(256));
        output.write(u8::from(rsa::eligible_pair(&p, &q, &pattern)));
    }
}
/// # Safety
/// Disjoint readable input[1596] and writable output[260]. No state survives calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn consumer_rsa_candidate(input: *const u8, output: *mut u8) {
    unsafe {
        let config: SearchConfig = read(input);
        let pattern: HexPattern = read(input.add(1072));
        let id = input.add(1588).cast::<u64>().read_unaligned();
        write(output, rsa::rsa_modulus(&config, id, &pattern));
    }
}
