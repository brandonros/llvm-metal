//! Points of y^2 = x^3 - 3x + b in projective coordinates (X : Y : Z), every
//! coordinate in Montgomery form. The identity is (0 : 1 : 0).
use crate::field::{Field, Number};

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Point {
    pub x: Number,
    pub y: Number,
    pub z: Number,
}

pub fn identity(field: &Field) -> Point {
    Point {
        x: [0; 8],
        y: field.one,
        z: [0; 8],
    }
}

/// `p + q` by the complete formulas of Renes, Costello and Batina (2015), with
/// a = -3. One code path covers addition, doubling and the identity, where the
/// usual formulas branch on each. `b3` is 3b.
pub fn add(p: &Point, q: &Point, b3: &Number, f: &Field) -> Point {
    let (x1, y1, z1) = (&p.x, &p.y, &p.z);
    let (x2, y2, z2) = (&q.x, &q.y, &q.z);
    let (xx, yy, zz) = (f.mul(x1, x2), f.mul(y1, y2), f.mul(z1, z2));
    // The cross terms: (x1 + y1)(x2 + y2) - xx - yy = x1 y2 + x2 y1, and so on.
    let cross = |a1, b1, a2, b2, aa, bb| {
        let product = f.mul(&f.add(a1, b1), &f.add(a2, b2));
        f.sub(&product, &f.add(aa, bb))
    };
    let xy = cross(x1, y1, x2, y2, &xx, &yy);
    let xz = cross(x1, z1, x2, z2, &xx, &zz);
    let yz = cross(y1, z1, y2, z2, &yy, &zz);

    // u = 3b zz - 3 xz;  w = 3b xz - 3 xx - 9 zz;  t = 3 xx - 3 zz
    let u = f.sub(&f.mul(b3, &zz), &f.triple(&xz));
    let zz3 = f.triple(&zz);
    let w = f.sub(&f.mul(b3, &xz), &f.add(&f.triple(&xx), &f.triple(&zz3)));
    let t = f.sub(&f.triple(&xx), &zz3);
    let (low, high) = (f.sub(&yy, &u), f.add(&yy, &u));
    Point {
        x: f.sub(&f.mul(&xy, &low), &f.mul(&yz, &w)),
        y: f.add(&f.mul(&low, &high), &f.mul(&t, &w)),
        z: f.add(&f.mul(&yz, &high), &f.mul(&xy, &t)),
    }
}
