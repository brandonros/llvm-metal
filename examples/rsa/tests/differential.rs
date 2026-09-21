//! The GPU signs what the host signs, for every thread, however rustc was asked
//! to optimize the kernel; and the public key, which shares only `montmul` with
//! the signer, accepts every signature.
#[test]
#[ignore = "requires an Apple GPU"]
fn gpu_matches_host_at_every_optimization_level() {
    let key = rsa::key::sign_key();
    let public = rsa::key::public_key();
    let messages = rsa::messages(100);
    let expected = rsa::on_host(&key, &messages);
    for (signature, message) in expected.iter().zip(&messages) {
        assert!(
            public.verifies(signature, message),
            "host signature rejected"
        );
    }
    for level in ["0", "s", "3"] {
        // SAFETY: this test binary has one test and no other threads.
        unsafe { std::env::set_var("CARGO_PROFILE_RELEASE_OPT_LEVEL", level) };
        let directory = rsa::root().join("../../target/examples/rsa").join(level);
        let (signatures, _) = rsa::Gpu::compile(&directory)
            .and_then(|gpu| gpu.sign(&key, &messages))
            .unwrap_or_else(|error| panic!("opt-level {level}: {error}"));
        assert!(
            signatures == expected,
            "opt-level {level}: GPU and host differ"
        );
    }
}
