//! The curve code is right (a known answer from RFC 6979, and the group's
//! order), the host's signatures verify, and the GPU signs what the host signs
//! however rustc was asked to optimize the kernel.
use p256::curve::{self, number};
use p256_kernel::{Request, Signature};

// RFC 6979, A.2.5: P-256, SHA-256, message "sample".
const D: &str = "c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721";
const QX: &str = "60fed4ba255a9d31c961eb74c6356d68c049b8923b61fa6ce669622e60f29fb6";
const QY: &str = "7903fe1008b8bc99a41ae9e95628bc64f2f1b20c2d7e9f5177a3c294d4462299";
const Z: &str = "af2bdbe1aa9b6ec1e2ade1d694f41fc71a831d0268e9891562113d8a62add1bf";
const K: &str = "a6e3c57dd01abe90086538398355dd4c3b17aa873382b0f24d6129493d8aad60";
const R: &str = "efd48b2aacb6a8fd1140dd9cd45e81d69d2c877b56aaf991c34d0ea84eaf3716";
const S: &str = "f7cb1c942d657c41d436c7a1b6e29f65f3e900dbb9aff4064dc4ab2f843acda8";

#[test]
fn the_order_of_the_generator_is_n() {
    let key = curve::key(&number(D));
    let end = curve::multiply(&key.n.modulus, &curve::generator(&key), &key);
    assert_eq!(curve::affine(&end, &key), None);
}

#[test]
fn the_host_signs_the_rfc_6979_vector() {
    let key = curve::key(&number(D));
    let public = curve::multiply(&number(D), &curve::generator(&key), &key);
    assert_eq!(curve::affine(&public, &key), Some([number(QX), number(QY)]));
    let signatures = p256::on_host(
        &key,
        &curve::table(&key),
        &[Request {
            z: number(Z),
            k: number(K),
        }],
    );
    assert_eq!(
        signatures,
        [Signature {
            r: number(R),
            s: number(S)
        }]
    );
    assert!(curve::verifies(&signatures[0], &number(Z), &public, &key));
}

/// Inputs the single code path must absorb: k = 1 and k = 16 add the identity in
/// 63 of 64 windows, k = n - 1 gives -G, and a hash of 2^256 - 1 is above n.
fn edge_cases(key: &p256_kernel::Key) -> Vec<Request> {
    let small = |k| {
        let mut number = [0; 8];
        number[0] = k;
        number
    };
    let mut last = key.n.modulus;
    last[0] -= 1;
    let z = number(Z);
    vec![
        Request { z, k: small(1) },
        Request { z, k: small(16) },
        Request { z, k: last },
        Request {
            z: [u32::MAX; 8],
            k: number(K),
        },
    ]
}

#[test]
fn the_edge_cases_sign_and_verify() {
    let key = curve::key(&number(D));
    let public = curve::multiply(&number(D), &curve::generator(&key), &key);
    let requests = edge_cases(&key);
    let signatures = p256::on_host(&key, &curve::table(&key), &requests);
    for (signature, request) in signatures.iter().zip(&requests) {
        assert!(curve::verifies(signature, &request.z, &public, &key));
    }
    // G and -G share an x coordinate, and it is below n.
    assert_eq!(signatures[0].r, number(curve::GX));
    assert_eq!(signatures[2].r, number(curve::GX));
}

#[test]
#[ignore = "requires an Apple GPU"]
fn gpu_matches_host_at_every_optimization_level() {
    let key = curve::key(&number(D));
    let table = curve::table(&key);
    let public = curve::multiply(&number(D), &curve::generator(&key), &key);
    let mut requests = edge_cases(&key);
    requests.extend(p256::requests(100));
    let expected = p256::on_host(&key, &table, &requests);
    for (signature, Request { z, .. }) in expected.iter().zip(&requests) {
        assert!(
            curve::verifies(signature, z, &public, &key),
            "host signature rejected"
        );
    }
    for level in ["0", "s", "3"] {
        // SAFETY: the other tests in this binary do not read the environment.
        unsafe { std::env::set_var("CARGO_PROFILE_RELEASE_OPT_LEVEL", level) };
        let directory = p256::root().join("../../target/examples/p256").join(level);
        let matches = p256::Gpu::compile(&directory, &key, &table, requests.len())
            .and_then(|mut gpu| {
                gpu.requests.write(|slice| slice.copy_from_slice(&requests));
                gpu.sign(requests.len())?;
                Ok(gpu.signatures.read(|signatures| signatures == expected))
            })
            .unwrap_or_else(|error| panic!("opt-level {level}: {error}"));
        assert!(matches, "opt-level {level}: GPU and host differ");
    }
}
