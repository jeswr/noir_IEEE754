// ============================================================================
// IEEE 754 Float32 Multiplication
// ============================================================================

use crate::float32::helpers::{
    float32_infinity, float32_is_denormal, float32_is_infinity, float32_is_nan, float32_is_zero,
    float32_nan, float32_zero,
};
use crate::types::{FLOAT32_EXPONENT_MAX, FLOAT32_IMPLICIT_BIT, IEEE754Float32};
use crate::utils::shift_right_sticky_u64;

// IEEE 754 compliant multiplication for 32-bit floats
pub fn mul_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    // Determine special cases
    let a_is_nan = float32_is_nan(a);
    let b_is_nan = float32_is_nan(b);
    let a_is_inf = float32_is_infinity(a);
    let b_is_inf = float32_is_infinity(b);
    let a_is_zero = float32_is_zero(a);
    let b_is_zero = float32_is_zero(b);
    let a_is_denormal = float32_is_denormal(a);
    let b_is_denormal = float32_is_denormal(b);

    // Result sign is XOR of input signs
    let result_sign: u1 = a.sign ^ b.sign;

    // Get effective mantissas with implicit bit (24 bits each)
    let mant_a: u64 = if a_is_denormal {
        a.mantissa as u64
    } else if a_is_zero {
        0
    } else {
        (a.mantissa | FLOAT32_IMPLICIT_BIT) as u64
    };

    let mant_b: u64 = if b_is_denormal {
        b.mantissa as u64
    } else if b_is_zero {
        0
    } else {
        (b.mantissa | FLOAT32_IMPLICIT_BIT) as u64
    };

    // Get effective exponents (unbiased)
    let exp_a_eff: i64 = if a_is_denormal {
        1 - 127
    } else if a_is_zero {
        0
    } else {
        (a.exponent as i64) - 127
    };

    let exp_b_eff: i64 = if b_is_denormal {
        1 - 127
    } else if b_is_zero {
        0
    } else {
        (b.exponent as i64) - 127
    };

    // Multiply mantissas (24-bit * 24-bit = 48-bit max)
    // For two normalized numbers: 1.xxx * 1.yyy gives result in range [1, 4)
    // The product will have the implicit bit at position 46 or 47
    let product: u64 = mant_a * mant_b;

    // Special case: result is zero
    let result_is_zero = product == 0;

    // For multiplication of normalized numbers:
    // mant_a = 1.xxx (implicit bit at position 23)
    // mant_b = 1.yyy (implicit bit at position 23)
    // product = result in [2^46, 2^48) for normalized * normalized
    // If bit 47 is set, result >= 2.0, need to shift right by 1
    // If bit 46 is set but not 47, result in [1, 2), normal case

    // Result exponent starts at exp_a + exp_b
    let mut result_exp: i64 = exp_a_eff + exp_b_eff + 127;
    let mut result_mant: u64 = product;

    // Normalize the product: find leading bit and adjust
    // For normalized * normalized, leading bit is at 46 or 47
    // For denormal * normal, it can be lower
    // We need the leading bit at position 46 before the guard shift

    // Check if bit 47 is set (overflow case)
    let bit_47_set = (product >> 47) & 1 == 1;
    if bit_47_set {
        result_exp = result_exp + 1;
        // Shift right by 1, keeping sticky bit
        result_mant = shift_right_sticky_u64(product, 1);
    }

    // Now normalize if the leading bit is below position 46
    // Use binary search to count leading zeros
    if (result_mant != 0) & ((result_mant >> 46) == 0) {
        let target_bit: u64 = 46;
        let mut leading_zeros: u64 = 0;
        let mut v = result_mant;
        if v & 0xFFFFFFFF00000000 == 0 {
            leading_zeros += 32;
            v <<= 32;
        }
        if v & 0xFFFF000000000000 == 0 {
            leading_zeros += 16;
            v <<= 16;
        }
        if v & 0xFF00000000000000 == 0 {
            leading_zeros += 8;
            v <<= 8;
        }
        if v & 0xF000000000000000 == 0 {
            leading_zeros += 4;
            v <<= 4;
        }
        if v & 0xC000000000000000 == 0 {
            leading_zeros += 2;
            v <<= 2;
        }
        if v & 0x8000000000000000 == 0 {
            leading_zeros += 1;
        }

        // shift_needed = leading_zeros - (64 - target_bit - 1) = leading_zeros - 17
        let shift_needed = leading_zeros - (64 - target_bit - 1);
        result_mant = result_mant << shift_needed;
        result_exp = result_exp - (shift_needed as i64);
    }

    // Now the implicit bit should be at position 46
    // We need to shift down to position 23, but keep 3 guard bits
    // So shift from position 46 to position 26 (46 - 26 = 20)
    let guard_shift: u64 = 46 - 26;
    result_mant = shift_right_sticky_u64(result_mant, guard_shift);

    // Round to nearest, ties to even
    let guard_bits = result_mant & 0x7;
    result_mant = result_mant >> 3;

    let should_round_up = (guard_bits > 4) | ((guard_bits == 4) & ((result_mant & 1) == 1));
    if should_round_up {
        result_mant = result_mant + 1;
        // Check for overflow from rounding
        if result_mant >= ((FLOAT32_IMPLICIT_BIT as u64) << 1) {
            result_mant = result_mant >> 1;
            result_exp = result_exp + 1;
        }
    }

    // Handle underflow to denormal
    let mut is_denormal_result = false;
    if result_exp <= 0 {
        // Shift mantissa right to create denormal
        let denorm_shift = 1 - result_exp;
        if denorm_shift < 24 {
            result_mant = shift_right_sticky_u64(result_mant, denorm_shift as u64);
        } else {
            result_mant = 0;
        }
        result_exp = 0;
        is_denormal_result = true;
    }

    // Check for overflow to infinity
    let overflows_to_inf = result_exp >= (FLOAT32_EXPONENT_MAX as i64);

    // Remove implicit bit for normalized results
    let final_mantissa = if is_denormal_result {
        (result_mant & 0x7FFFFF) as u32
    } else {
        (result_mant & ((FLOAT32_IMPLICIT_BIT - 1) as u64)) as u32
    };

    let normal_result =
        IEEE754Float32 { sign: result_sign, exponent: result_exp as u8, mantissa: final_mantissa };

    // Handle special cases
    let mut result = normal_result;

    // Zero result
    if result_is_zero | (result_mant == 0) {
        result = float32_zero(result_sign);
    }

    // Overflow to infinity
    if overflows_to_inf {
        result = float32_infinity(result_sign);
    }

    // Handle zero inputs
    if a_is_zero | b_is_zero {
        // 0 * inf = NaN, 0 * finite = 0
        if a_is_inf | b_is_inf {
            result = float32_nan();
        } else {
            result = float32_zero(result_sign);
        }
    }

    // Handle infinity inputs
    if (a_is_inf | b_is_inf) & !a_is_zero & !b_is_zero {
        if a_is_nan | b_is_nan {
            result = float32_nan();
        } else {
            result = float32_infinity(result_sign);
        }
    }

    // Handle NaN inputs (highest priority)
    if a_is_nan | b_is_nan {
        result = float32_nan();
    }

    result
}
