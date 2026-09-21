//! xoroshiro128**, seeded from one word through SplitMix64.
pub struct Xoroshiro(u64, u64);

impl Xoroshiro {
    pub fn seeded(seed: u64) -> Self {
        let mut state = seed;
        let mut word = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mixed = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            let mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            mixed ^ (mixed >> 31)
        };
        // An all-zero state never leaves zero.
        match (word(), word()) {
            (0, 0) => Self(1, 0),
            (first, second) => Self(first, second),
        }
    }

    pub fn next(&mut self) -> u64 {
        let result = self.0.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        self.1 ^= self.0;
        self.0 = self.0.rotate_left(24) ^ self.1 ^ (self.1 << 16);
        self.1 = self.1.rotate_left(37);
        result
    }
}
