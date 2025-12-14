// ============================================================================
// Shared Utility Functions for IEEE 754 Operations
// ============================================================================

// Helper to shift right with sticky bit preservation for u64
pub fn shift_right_sticky_u64(value: u64, shift: u64) -> u64 {
    if shift == 0 {
        value
    } else if shift >= 64 {
        if value != 0 {
            1
        } else {
            0
        }
    } else {
        let mask = (1 << shift) - 1;
        let shifted_out = value & mask;
        let result = value >> shift;
        if shifted_out != 0 {
            result | 1
        } else {
            result
        }
    }
}

// Helper for 128-bit multiplication result
pub struct U128 {
    pub high: u64,
    pub low: u64,
}

// Multiply two u64 values and return 128-bit result
pub fn mul_u64_to_u128(a: u64, b: u64) -> U128 {
    // Split into 32-bit parts
    let a_lo = a & 0xFFFFFFFF;
    let a_hi = a >> 32;
    let b_lo = b & 0xFFFFFFFF;
    let b_hi = b >> 32;

    // Partial products
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    // Sum partial products with carry propagation
    let mid = p1 + p2;
    let mid_carry: u64 = if mid < p1 { 1 } else { 0 };

    let low = p0 + (mid << 32);
    let low_carry: u64 = if low < p0 { 1 } else { 0 };

    let high = p3 + (mid >> 32) + (mid_carry << 32) + low_carry;

    U128 { high, low }
}

// Shift right a 128-bit value with sticky bit preservation
pub fn shift_right_sticky_u128(val: U128, shift: u64) -> u64 {
    if shift == 0 {
        val.low
    } else if shift >= 128 {
        if (val.high != 0) | (val.low != 0) {
            1
        } else {
            0
        }
    } else if shift >= 64 {
        let effective_shift = shift - 64;
        let sticky = if val.low != 0 { 1 } else { 0 };
        if effective_shift >= 64 {
            if (val.high != 0) | (sticky != 0) {
                1
            } else {
                0
            }
        } else {
            let mask = (1 << effective_shift) - 1;
            let shifted_out = val.high & mask;
            let result = val.high >> effective_shift;
            if (shifted_out != 0) | (sticky != 0) {
                result | 1
            } else {
                result
            }
        }
    } else {
        // shift < 64
        let mask = (1 << shift) - 1;
        let shifted_out = val.low & mask;
        let result_low = (val.low >> shift) | (val.high << (64 - shift));
        if shifted_out != 0 {
            result_low | 1
        } else {
            result_low
        }
    }
}

// Helper for 128-bit division: divides a 128-bit dividend by a 64-bit divisor
// Returns quotient (fits in 64 bits for our use case) and remainder
pub fn div_u128_by_u64(dividend: U128, divisor: u64) -> (u64, u64) {
    // For simplicity, we'll do this with a long division approach
    // This works because our dividend is at most 2^109 and divisor is at least 2^52
    // so the quotient fits in 57 bits (< 64 bits)

    if divisor == 0 {
        (0, 0)
    } else if dividend.high == 0 {
        (dividend.low / divisor, dividend.low % divisor)
    } else {
        // Long division bit by bit
        // Process 128 bits from most significant to least significant
        let mut quotient: u64 = 0;
        let mut remainder: u64 = 0;

        // Process high bits first (64 bits)
        for i in 0..64 {
            let bit_pos = 63 - i;
            remainder = remainder << 1;
            remainder = remainder | ((dividend.high >> (bit_pos as u64)) & 1);

            if remainder >= divisor {
                remainder = remainder - divisor;
                // This bit position in quotient would be 64 + bit_pos, which is >= 64
                // For our use case, we know the quotient fits in 64 bits, so these
                // high bits should all be 0. We can skip adding to quotient here.
            }
        }

        // Process low bits (64 bits)
        for i in 0..64 {
            let bit_pos = 63 - i;
            remainder = remainder << 1;
            remainder = remainder | ((dividend.low >> (bit_pos as u64)) & 1);

            quotient = quotient << 1;
            if remainder >= divisor {
                remainder = remainder - divisor;
                quotient = quotient | 1;
            }
        }

        (quotient, remainder)
    }
}
