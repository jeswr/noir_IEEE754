// ============================================================================
// IEEE 754 Float64 Division
// ============================================================================

use crate::float64::helpers::{
    float64_infinity, float64_is_denormal, float64_is_infinity, float64_is_nan, float64_is_zero,
    float64_nan, float64_zero,
};
use crate::types::{FLOAT64_EXPONENT_MAX, FLOAT64_IMPLICIT_BIT, IEEE754Float64};
use crate::utils::{div_u128_by_u64, shift_right_sticky_u64, U128};

// IEEE 754 compliant division for 64-bit floats
pub fn div_float64(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
    // Determine special cases
    let a_is_nan = float64_is_nan(a);
    let b_is_nan = float64_is_nan(b);
    let a_is_inf = float64_is_infinity(a);
    let b_is_inf = float64_is_infinity(b);
    let a_is_zero = float64_is_zero(a);
    let b_is_zero = float64_is_zero(b);
    let a_is_denormal = float64_is_denormal(a);
    let b_is_denormal = float64_is_denormal(b);

    // Result sign is XOR of input signs
    let result_sign: u1 = a.sign ^ b.sign;

    // Get effective mantissas with implicit bit (53 bits)
    let mant_a: u64 = if a_is_denormal {
        a.mantissa
    } else if a_is_zero {
        0
    } else {
        a.mantissa | FLOAT64_IMPLICIT_BIT
    };

    let mant_b: u64 = if b_is_denormal {
        b.mantissa
    } else if b_is_zero {
        0
    } else {
        b.mantissa | FLOAT64_IMPLICIT_BIT
    };

    // Get effective exponents (unbiased)
    let exp_a_eff: i64 = if a_is_denormal {
        1 - 1023
    } else if a_is_zero {
        0
    } else {
        (a.exponent as i64) - 1023
    };

    let exp_b_eff: i64 = if b_is_denormal {
        1 - 1023
    } else if b_is_zero {
        0
    } else {
        (b.exponent as i64) - 1023
    };

    // For division of normalized numbers:
    // Both mantissas have implicit bit at position 52
    // Quotient is in range [0.5, 2) since both inputs are in [1, 2)

    // Shift dividend left by 56 bits to get 56 quotient bits (53 + 3 guard)
    // mant_a << 56 creates a 109-bit value
    let shift_amount: u64 = 56;
    let shifted_dividend =
        U128 { high: mant_a >> (64 - shift_amount), low: mant_a << shift_amount };

    // Perform division
    let (quotient, remainder) = if mant_b != 0 {
        div_u128_by_u64(shifted_dividend, mant_b)
    } else {
        (0, 0)
    };

    // Result exponent
    let mut result_exp: i64 = exp_a_eff - exp_b_eff + 1023;

    // The quotient structure:
    // - If mant_a >= mant_b (quotient >= 1.0): quotient in [2^56, 2^57), bit 56 is implicit bit
    // - If mant_a < mant_b (quotient < 1.0): quotient in [2^55, 2^56), bit 55 is highest

    let mut result_mant: u64 = quotient;

    // Check if quotient >= 2^57 (result >= 2.0), need to shift right
    let bit_57_set = (quotient >> 57) & 1 == 1;
    if bit_57_set {
        result_mant = shift_right_sticky_u64(quotient, 1);
        result_exp = result_exp + 1;
    }

    // Check if quotient < 2^56 (result < 1.0), need to shift left and decrease exponent
    let bit_56_clear = (quotient >> 56) & 1 == 0;
    if bit_56_clear & !bit_57_set & (quotient != 0) {
        result_mant = quotient << 1;
        result_exp = result_exp - 1;
    }

    // Add sticky bit from remainder
    if remainder != 0 {
        result_mant = result_mant | 1;
    }

    // Now result_mant has implicit bit at position 56
    // We need to shift down to position 52, keeping guard bits
    // Shift from 56 to 55 (leaving 3 guard bits at positions 2, 1, 0)
    // Wait, we need shift of 56 - 52 = 4, then use 4 bits as guard
    let guard_bits = result_mant & 0xF; // 4 bits
    result_mant = result_mant >> 4;

    // Round to nearest, ties to even (using 4-bit guard, threshold is 8)
    let should_round_up = (guard_bits > 8) | ((guard_bits == 8) & ((result_mant & 1) == 1));
    if should_round_up {
        result_mant = result_mant + 1;
        if result_mant >= (FLOAT64_IMPLICIT_BIT << 1) {
            result_mant = result_mant >> 1;
            result_exp = result_exp + 1;
        }
    }

    // Handle underflow to denormal
    let mut is_denormal_result = false;
    if result_exp <= 0 {
        let denorm_shift = 1 - result_exp;
        if denorm_shift < 53 {
            result_mant = shift_right_sticky_u64(result_mant, denorm_shift as u64);
        } else {
            result_mant = 0;
        }
        result_exp = 0;
        is_denormal_result = true;
    }

    // Check for overflow to infinity
    let overflows_to_inf = result_exp >= (FLOAT64_EXPONENT_MAX as i64);

    // Remove implicit bit for final mantissa
    let final_mantissa = if is_denormal_result {
        result_mant & 0xFFFFFFFFFFFFF
    } else {
        result_mant & (FLOAT64_IMPLICIT_BIT - 1)
    };

    let normal_result =
        IEEE754Float64 { sign: result_sign, exponent: result_exp as u16, mantissa: final_mantissa };

    // Handle special cases
    let mut result = normal_result;

    // Zero dividend
    if a_is_zero {
        if b_is_zero | b_is_nan {
            result = float64_nan();
        } else {
            result = float64_zero(result_sign);
        }
    }

    // Zero divisor (division by zero)
    if b_is_zero & !a_is_zero & !a_is_nan {
        result = float64_infinity(result_sign);
    }

    // Overflow to infinity
    if overflows_to_inf & !a_is_zero & !b_is_inf {
        result = float64_infinity(result_sign);
    }

    // Handle infinity dividend
    if a_is_inf & !b_is_nan {
        if b_is_inf {
            result = float64_nan();
        } else {
            result = float64_infinity(result_sign);
        }
    }

    // Handle infinity divisor
    if b_is_inf & !a_is_inf & !a_is_nan {
        result = float64_zero(result_sign);
    }

    // Handle NaN inputs (highest priority)
    if a_is_nan | b_is_nan {
        result = float64_nan();
    }

    result
}
