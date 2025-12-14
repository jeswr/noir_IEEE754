// ============================================================================
// IEEE 754 Float64 Square Root
// ============================================================================
//
// IEEE 754 square root rules:
// - sqrt(+0) = +0
// - sqrt(-0) = -0
// - sqrt(+Inf) = +Inf
// - sqrt(negative) = NaN (except -0)
// - sqrt(NaN) = NaN
// ============================================================================

use crate::float64::helpers::{
    float64_infinity, float64_is_denormal, float64_is_infinity, float64_is_nan, float64_is_zero,
    float64_nan, float64_zero,
};
use crate::types::{FLOAT64_EXPONENT_MAX, FLOAT64_IMPLICIT_BIT, IEEE754Float64};
use crate::utils::{shift_right_sticky_u64, U128};

// IEEE 754 compliant square root for 64-bit floats
pub fn sqrt_float64(a: IEEE754Float64) -> IEEE754Float64 {
    // Handle special cases
    let a_is_nan = float64_is_nan(a);
    let a_is_inf = float64_is_infinity(a);
    let a_is_zero = float64_is_zero(a);
    let a_is_denormal = float64_is_denormal(a);
    let a_is_negative = a.sign == 1;

    // Normalize denormals and get effective exponent
    let a_mant_raw: u64 = a.mantissa;
    let mut a_lz: i64 = 0;
    if a_is_denormal & (a_mant_raw != 0) {
        let mut v = a_mant_raw;
        // Binary search for leading zeros relative to bit 51
        if (v & 0xFFFFFFF000000) == 0 {
            a_lz = a_lz + 28;
            v = v << 28;
        }
        if (v & 0xFFFC000000000) == 0 {
            a_lz = a_lz + 14;
            v = v << 14;
        }
        if (v & 0xFE00000000000) == 0 {
            a_lz = a_lz + 7;
            v = v << 7;
        }
        if (v & 0xF000000000000) == 0 {
            a_lz = a_lz + 4;
            v = v << 4;
        }
        if (v & 0xC000000000000) == 0 {
            a_lz = a_lz + 2;
            v = v << 2;
        }
        if (v & 0x8000000000000) == 0 {
            a_lz = a_lz + 1;
        }
        a_lz = a_lz + 1;
    }

    // Get mantissa with implicit bit (53 bits, implicit at position 52)
    let mant_a: u64 = if a_is_denormal {
        a_mant_raw << (a_lz as u64)
    } else {
        a.mantissa | FLOAT64_IMPLICIT_BIT
    };

    // Get effective exponent (unbiased)
    let exp_a_eff: i64 = if a_is_denormal {
        1 - 1023 - a_lz
    } else {
        (a.exponent as i64) - 1023
    };

    // For sqrt: result_exp = floor(exp_a_eff / 2)
    // If exp_a_eff is odd, we need to adjust the mantissa
    let exp_is_odd = (exp_a_eff & 1) != 0;
    let mut result_exp: i64 = if exp_is_odd {
        (exp_a_eff - 1) / 2 + 1023
    } else {
        exp_a_eff / 2 + 1023
    };

    // Adjust mantissa for odd exponent (multiply by 2)
    let adjusted_mant: u64 = if exp_is_odd { mant_a << 1 } else { mant_a };

    // Square root algorithm scaling:
    // - mant_a has implicit bit at position 52, representing value in [1, 2)
    // - For result with 55 bits (52 mantissa + 3 guard), we need 110-bit input
    // - result_mant / 2^55 = sqrt(adjusted_mant / 2^k) where k = 52 or 53
    // - => result_mant = sqrt(adjusted_mant * 2^(110 - k))
    // - For even exp: k=52, radicand = adjusted_mant << 58
    // - For odd exp: k=53, radicand = adjusted_mant << 57

    let shift_amount: u64 = if exp_is_odd { 57 } else { 58 };

    // Create 128-bit radicand
    let radicand_high: u64 = adjusted_mant >> (64 - shift_amount);
    let radicand_low: u64 = adjusted_mant << shift_amount;

    // Digit-by-digit square root algorithm for 110-bit input
    // Result will be ~55 bits
    let mut sqrt_result: u64 = 0;
    let mut remainder = U128 { high: 0, low: 0 };
    let mut bit: u64 = 1 << 54; // Start from bit position 54

    // Process 110 bits of input, 2 at a time (55 iterations)
    let mut input_pos: i64 = 108; // Start at bits 108-109

    for _i in 0..55 {
        // Shift remainder left by 2
        remainder = U128 {
            high: (remainder.high << 2) | (remainder.low >> 62),
            low: remainder.low << 2,
        };

        // Add next two bits from radicand
        if input_pos >= 0 {
            let two_bits = if input_pos >= 64 {
                (radicand_high >> ((input_pos - 64) as u64)) & 3
            } else {
                (radicand_low >> (input_pos as u64)) & 3
            };
            remainder.low = remainder.low | two_bits;
        }
        input_pos = input_pos - 2;

        // trial = (2 * result + bit) = (result << 1) | bit
        let trial: u64 = (sqrt_result << 1) | bit;

        // Compare remainder >= trial (where trial fits in 64 bits but remainder is 128-bit)
        let remainder_ge_trial =
            (remainder.high > 0) | ((remainder.high == 0) & (remainder.low >= trial));

        if remainder_ge_trial {
            // remainder = remainder - trial (trial fits in 64 bits)
            if remainder.low >= trial {
                remainder.low = remainder.low - trial;
            } else {
                // Borrow from high
                remainder.high = remainder.high - 1;
                remainder.low = remainder.low + (0 - trial); // wrapping subtraction
            }
            sqrt_result = sqrt_result | bit;
        }
        bit = bit >> 1;
    }

    // sqrt_result has ~55 bits with implicit bit at position 54
    // We need implicit bit at position 52, so we have 2 extra bits for guard
    // Actually let's be more precise: we want 3 guard bits

    // Shift result to have implicit at position 55 (so we have 3 guard bits at 2,1,0)
    // Current implicit is at 54, shift left by 1
    let mut result_mant = sqrt_result << 1;

    // Add sticky bit if remainder is non-zero
    if (remainder.high != 0) | (remainder.low != 0) {
        result_mant = result_mant | 1;
    }

    // result_mant format: implicit bit at position 55, we want at position 52
    // So we have 3 guard bits at positions 2, 1, 0

    // Handle underflow to denormal
    let mut is_denormal_result = false;

    if result_exp <= 0 {
        let denorm_shift = 1 - result_exp;
        if denorm_shift < 56 {
            result_mant = shift_right_sticky_u64(result_mant, denorm_shift as u64);
        } else {
            result_mant = 0;
        }
        result_exp = 0;
        is_denormal_result = true;
    }

    // Shift to remove guard bits (shift by 3)
    let guard_bits = result_mant & 0x7;
    result_mant = result_mant >> 3;

    // Round to nearest, ties to even
    let should_round_up = (guard_bits > 4) | ((guard_bits == 4) & ((result_mant & 1) == 1));
    if should_round_up {
        result_mant = result_mant + 1;
        // Check for overflow from rounding
        if !is_denormal_result & (result_mant >= (FLOAT64_IMPLICIT_BIT << 1)) {
            result_mant = result_mant >> 1;
            result_exp = result_exp + 1;
        }
        // For denormals, if rounding causes overflow into normal range
        if is_denormal_result & (result_mant >= FLOAT64_IMPLICIT_BIT) {
            result_exp = 1;
            is_denormal_result = false;
        }
    }

    // Check for overflow to infinity (shouldn't happen for sqrt)
    let overflows_to_inf = result_exp >= (FLOAT64_EXPONENT_MAX as i64);

    // Remove implicit bit for final mantissa
    let final_mantissa = if is_denormal_result {
        result_mant & 0xFFFFFFFFFFFFF
    } else {
        result_mant & (FLOAT64_IMPLICIT_BIT - 1)
    };

    let normal_result =
        IEEE754Float64 { sign: 0, exponent: result_exp as u16, mantissa: final_mantissa };

    // Handle special cases (highest priority last)
    let mut result = normal_result;

    // Overflow to infinity
    if overflows_to_inf {
        result = float64_infinity(0);
    }

    // sqrt(+0) = +0, sqrt(-0) = -0
    if a_is_zero {
        result = float64_zero(a.sign);
    }

    // sqrt(+Inf) = +Inf
    if a_is_inf & !a_is_negative {
        result = float64_infinity(0);
    }

    // sqrt of negative number (except -0) is NaN
    if a_is_negative & !a_is_zero {
        result = float64_nan();
    }

    // NaN input produces NaN
    if a_is_nan {
        result = float64_nan();
    }

    result
}
