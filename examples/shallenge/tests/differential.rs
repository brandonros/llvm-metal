//! The GPU computes what the host computes, for every thread, however rustc
//! was asked to optimize the kernel.
use shallenge_kernel::{PREFIX, Request};

#[test]
#[ignore = "requires an Apple GPU"]
fn gpu_matches_host_at_every_optimization_level() {
    let request = Request {
        seed: 7,
        prefix_length: 5,
        prefix: [b'x'; PREFIX],
    };
    let threads = 1000;
    let expected = shallenge::on_host(&request, threads);
    for level in ["0", "s", "3"] {
        // SAFETY: this test binary has one test and no other threads.
        unsafe { std::env::set_var("CARGO_PROFILE_RELEASE_OPT_LEVEL", level) };
        let directory = shallenge::root()
            .join("../../target/examples/shallenge")
            .join(level);
        let records = shallenge::on_gpu(&request, threads, &directory)
            .unwrap_or_else(|error| panic!("opt-level {level}: {error}"));
        assert!(
            records == expected,
            "opt-level {level}: GPU and host differ"
        );
    }
}
