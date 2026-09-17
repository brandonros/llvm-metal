#![no_std]
//! Runtime-input probes of the same k256 release used by vanity-miner.
use k256::{FieldElement, Scalar, elliptic_curve::PrimeField};

#[cfg(feature = "consumer")]
mod consumer;
#[cfg(feature = "consumer")]
pub use consumer::*;

unsafe fn read<const N: usize>(input: *const u8) -> [u8; N] {
    unsafe { input.cast::<[u8; N]>().read_unaligned() }
}

/// # Safety
/// Disjoint readable 32-byte input and writable 33-byte output, byte aligned.
/// Input/output scalar bytes are big endian. Output starts with validity (0/1).
/// Zero is a valid Scalar, though it is not a valid SecretKey.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k256_scalar_roundtrip(input: *const u8, output: *mut u8) {
    let scalar: Option<Scalar> = Scalar::from_repr(unsafe { read::<32>(input) }.into()).into();
    let mut result = [0u8; 33];
    if let Some(scalar) = scalar {
        result[0] = 1;
        result[1..].copy_from_slice(&scalar.to_bytes());
    }
    unsafe { output.cast::<[u8; 33]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable 16-byte input and writable 16-byte output, byte aligned.
/// Input is two LE u64s; output is their exact LE u128 product.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wide_mul(input: *const u8, output: *mut u8) {
    let a = u64::from_le_bytes(unsafe { read(input) });
    let b = u64::from_le_bytes(unsafe { read(input.add(8)) });
    let product = (a as u128) * (b as u128);
    unsafe {
        output
            .cast::<[u8; 16]>()
            .write_unaligned(product.to_le_bytes())
    };
}

/// # Safety
/// Disjoint readable 64-byte input and writable 33-byte output, byte aligned.
/// Input is two canonical BE field elements. Output: validity + normalized BE product.
/// Invalid inputs produce 33 zero bytes, without a panic path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k256_field_mul(input: *const u8, output: *mut u8) {
    let a: Option<FieldElement> =
        FieldElement::from_bytes(&unsafe { read::<32>(input) }.into()).into();
    let b: Option<FieldElement> =
        FieldElement::from_bytes(&unsafe { read::<32>(input.add(32)) }.into()).into();
    let mut result = [0u8; 33];
    if let (Some(a), Some(b)) = (a, b) {
        result[0] = 1;
        result[1..].copy_from_slice(&a.mul(&b).normalize().to_bytes());
    }
    unsafe { output.cast::<[u8; 33]>().write_unaligned(result) };
}

pub type Probe = unsafe extern "C" fn(*const u8, *mut u8);

unsafe fn field_unary<const INVERT: bool>(input: *const u8, output: *mut u8) {
    let a: Option<FieldElement> =
        FieldElement::from_bytes(&unsafe { read::<32>(input) }.into()).into();
    let mut result = [0u8; 33];
    if let Some(a) = a {
        let value: Option<FieldElement> = if INVERT {
            a.invert().into()
        } else {
            Some(a.square())
        };
        if let Some(value) = value {
            result[0] = 1;
            result[1..].copy_from_slice(&value.normalize().to_bytes());
        }
    }
    unsafe { output.cast::<[u8; 33]>().write_unaligned(result) };
}

/// # Safety
/// Disjoint readable BE field element[32] and writable validity + BE square[33].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k256_field_square(input: *const u8, output: *mut u8) {
    unsafe { field_unary::<false>(input, output) };
}

/// # Safety
/// Disjoint readable BE field element[32] and writable validity + BE inverse[33].
/// Zero and noncanonical inputs produce all-zero output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k256_field_invert(input: *const u8, output: *mut u8) {
    unsafe { field_unary::<true>(input, output) };
}

/// # Safety
/// Disjoint readable affine BE x[32],y[32] and writable validity + SEC1 point[66].
/// Invalid coordinates produce all-zero output. Inputs are runtime coordinates.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k256_point_double(input: *const u8, output: *mut u8) {
    use k256::{
        AffinePoint, EncodedPoint, ProjectivePoint,
        elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint},
    };
    let x = unsafe { read::<32>(input) };
    let y = unsafe { read::<32>(input.add(32)) };
    let encoded = EncodedPoint::from_affine_coordinates((&x).into(), (&y).into(), false);
    let point: Option<AffinePoint> = AffinePoint::from_encoded_point(&encoded).into();
    let mut result = [0u8; 66];
    if let Some(point) = point {
        let encoded = ProjectivePoint::from(point)
            .double()
            .to_affine()
            .to_encoded_point(false);
        if encoded.as_bytes().len() == 65 {
            result[0] = 1;
            result[1..].copy_from_slice(encoded.as_bytes());
        }
    }
    unsafe { output.cast::<[u8; 66]>().write_unaligned(result) };
}

pub fn probe(entry: &str) -> Option<(usize, usize, Probe)> {
    match entry {
        "k256_scalar_roundtrip" => Some((32, 33, k256_scalar_roundtrip)),
        "wide_mul" => Some((16, 16, wide_mul)),
        "k256_field_mul" => Some((64, 33, k256_field_mul)),
        "k256_field_square" => Some((32, 33, k256_field_square)),
        "k256_field_invert" => Some((32, 33, k256_field_invert)),
        "k256_point_double" => Some((64, 66, k256_point_double)),
        "k256_scalar_mul" => Some((32, 99, k256_scalar_mul)),
        #[cfg(feature = "consumer")]
        "consumer_public_keys" => Some((32, 99, consumer_public_keys)),
        #[cfg(feature = "consumer")]
        "consumer_keccak256" => Some((64, 32, consumer_keccak256)),
        #[cfg(feature = "consumer")]
        "consumer_ethereum_address" => Some((32, 85, consumer_ethereum_address)),
        _ => None,
    }
}

/// # Safety
/// Disjoint readable BE scalar[32] and writable validity + compressed[33] +
/// uncompressed[65] output. Zero or noncanonical scalars produce all zeroes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn k256_scalar_mul(input: *const u8, output: *mut u8) {
    use k256::{
        ProjectivePoint,
        elliptic_curve::{point::AffineCoordinates, sec1::ToEncodedPoint},
    };
    let scalar: Option<Scalar> = Scalar::from_repr(unsafe { read::<32>(input) }.into()).into();
    let mut result = [0u8; 99];
    if let Some(scalar) = scalar
        && scalar != Scalar::ZERO
    {
        let point = (ProjectivePoint::GENERATOR * scalar).to_affine();
        let uncompressed = point.to_encoded_point(false);
        if uncompressed.as_bytes().len() == 65 {
            result[0] = 1;
            result[1] = 2 | (point.y_is_odd().unwrap_u8() & 1);
            result[2..34].copy_from_slice(&point.x());
            result[34..].copy_from_slice(uncompressed.as_bytes());
        }
    }
    unsafe { output.cast::<[u8; 99]>().write_unaligned(result) };
}
