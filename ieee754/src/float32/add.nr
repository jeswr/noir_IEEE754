// ============================================================================
// IEEE 754 Float32 Addition
// ============================================================================

use crate::float32::helpers::{
    float32_infinity, float32_is_denormal, float32_is_infinity, float32_is_nan, float32_is_zero,
    float32_nan, float32_zero,
};
use crate::types::{FLOAT32_EXPONENT_MAX, FLOAT32_IMPLICIT_BIT, IEEE754Float32};
use crate::unconstrained_ops::{count_leading_zeros_u64_verified, shift_right_sticky_u64_verified};

// IEEE 754 compliant addition for 32-bit floats
pub fn add_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    let a_is_nan = float32_is_nan(a);
    let b_is_nan = float32_is_nan(b);
    let a_is_inf = float32_is_infinity(a);
    let b_is_inf = float32_is_infinity(b);
    let a_is_zero = float32_is_zero(a);
    let b_is_zero = float32_is_zero(b);
    let a_is_denormal = float32_is_denormal(a);
    let b_is_denormal = float32_is_denormal(b);

    let mant_a_raw: u64 = if a_is_denormal {
        a.mantissa as u64
    } else {
        (a.mantissa | FLOAT32_IMPLICIT_BIT) as u64
    };
    let mant_b_raw: u64 = if b_is_denormal {
        b.mantissa as u64
    } else {
        (b.mantissa | FLOAT32_IMPLICIT_BIT) as u64
    };
    let exp_a_eff: u64 = if a_is_denormal | a_is_zero {
        1
    } else {
        a.exponent as u64
    };
    let exp_b_eff: u64 = if b_is_denormal | b_is_zero {
        1
    } else {
        b.exponent as u64
    };

    let mut mant_a: u64 = mant_a_raw << 3;
    let mut mant_b: u64 = mant_b_raw << 3;
    let mut result_exp: u64 = exp_a_eff;

    if exp_a_eff > exp_b_eff {
        let shift = exp_a_eff - exp_b_eff;
        mant_b = shift_right_sticky_u64_verified(mant_b, shift);
        result_exp = exp_a_eff;
    } else if exp_b_eff > exp_a_eff {
        let shift = exp_b_eff - exp_a_eff;
        mant_a = shift_right_sticky_u64_verified(mant_a, shift);
        result_exp = exp_b_eff;
    }

    let mut result_mant: u64 = 0;
    let mut result_sign: u1 = 0;
    if a.sign == b.sign {
        result_mant = mant_a + mant_b;
        result_sign = a.sign;
    } else {
        if mant_a >= mant_b {
            result_mant = mant_a - mant_b;
            result_sign = a.sign;
        } else {
            result_mant = mant_b - mant_a;
            result_sign = b.sign;
        }
    }

    // Overflow normalization
    let overflow_threshold: u64 = (FLOAT32_IMPLICIT_BIT as u64) << 4;
    if result_mant >= overflow_threshold {
        result_mant = shift_right_sticky_u64_verified(result_mant, 1);
        result_exp = result_exp + 1;
    }

    // Underflow normalization using optimized leading zero count
    let normal_position: u64 = (FLOAT32_IMPLICIT_BIT as u64) << 3;
    if (result_mant != 0) & (result_mant < normal_position) {
        let target_bit: u64 = 26;
        let leading_zeros = count_leading_zeros_u64_verified(result_mant);

        let shift_needed = leading_zeros - (64 - target_bit - 1);
        let max_shift = if result_exp > 1 { result_exp - 1 } else { 0 };
        let actual_shift = if shift_needed <= max_shift {
            shift_needed
        } else {
            max_shift
        };

        result_mant = result_mant << (actual_shift as u8);
        result_exp = result_exp - actual_shift;
    }

    // Denormal result
    let mut is_denormal_result = false;
    if (result_exp == 0) | ((result_exp == 1) & (result_mant < normal_position)) {
        result_exp = 0;
        is_denormal_result = true;
    }

    // Rounding
    let guard_bits = result_mant & 0x7;
    result_mant = result_mant >> 3;
    let should_round_up = (guard_bits > 4) | ((guard_bits == 4) & ((result_mant & 1) == 1));
    if should_round_up {
        result_mant = result_mant + 1;
        if result_mant >= ((FLOAT32_IMPLICIT_BIT as u64) << 1) {
            result_mant = result_mant >> 1;
            result_exp = result_exp + 1;
        } else if is_denormal_result & (result_mant >= (FLOAT32_IMPLICIT_BIT as u64)) {
            result_exp = 1;
        }
    }

    // Build result
    let overflows_to_inf = result_exp >= (FLOAT32_EXPONENT_MAX as u64);
    let result_is_zero = result_mant == 0;
    let final_mantissa = if is_denormal_result & (result_exp == 0) {
        (result_mant & 0x7FFFFF) as u32
    } else {
        (result_mant & ((FLOAT32_IMPLICIT_BIT - 1) as u64)) as u32
    };
    let normal_result =
        IEEE754Float32 { sign: result_sign, exponent: result_exp as u8, mantissa: final_mantissa };

    let mut result = normal_result;
    if result_is_zero {
        result = float32_zero(0);
    }
    if overflows_to_inf {
        result = float32_infinity(result_sign);
    }
    if a_is_zero & b_is_zero {
        result = float32_zero(a.sign & b.sign);
    } else if a_is_zero {
        result = b;
    } else if b_is_zero {
        result = a;
    }
    if a_is_inf & !b_is_inf {
        result = a;
    } else if b_is_inf & !a_is_inf {
        result = b;
    } else if a_is_inf & b_is_inf {
        if a.sign == b.sign {
            result = a;
        } else {
            result = float32_nan();
        }
    }
    if a_is_nan | b_is_nan {
        result = float32_nan();
    }

    result
}
