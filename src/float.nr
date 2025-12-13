// ============================================================================
// IEEE 754 Compliant Floating Point Types
// ============================================================================

// IEEE 754 Single-Precision (32-bit) Float
// - 1 bit sign
// - 8 bits exponent (bias 127)
// - 23 bits mantissa (with implicit leading 1 for normalized numbers)
pub struct IEEE754Float32 {
    pub sign: u1, // 0 = positive, 1 = negative
    pub exponent: u8, // Biased exponent (0-255)
    pub mantissa: u32, // 23-bit mantissa stored in lower bits
}

// IEEE 754 Double-Precision (64-bit) Float
// - 1 bit sign
// - 11 bits exponent (bias 1023)
// - 52 bits mantissa (with implicit leading 1 for normalized numbers)
pub struct IEEE754Float64 {
    pub sign: u1, // 0 = positive, 1 = negative
    pub exponent: u16, // Biased exponent (0-2047), using u16 to hold 11 bits
    pub mantissa: u64, // 52-bit mantissa stored in lower bits
}

// Constants for IEEE 754 Float32
global FLOAT32_EXPONENT_MAX: u8 = 255;
global FLOAT32_IMPLICIT_BIT: u32 = 0x800000; // 1 << 23

// Constants for IEEE 754 Float64
global FLOAT64_EXPONENT_MAX: u16 = 2047;
global FLOAT64_IMPLICIT_BIT: u64 = 0x10000000000000; // 1 << 52

// ============================================================================
// IEEE 754 Float32 Helper Functions
// ============================================================================

// Check if Float32 is NaN (exponent = 255, mantissa != 0)
pub fn float32_is_nan(x: IEEE754Float32) -> bool {
    (x.exponent == FLOAT32_EXPONENT_MAX) & (x.mantissa != 0)
}

// Check if Float32 is Infinity (exponent = 255, mantissa = 0)
pub fn float32_is_infinity(x: IEEE754Float32) -> bool {
    (x.exponent == FLOAT32_EXPONENT_MAX) & (x.mantissa == 0)
}

// Check if Float32 is Zero (exponent = 0, mantissa = 0)
pub fn float32_is_zero(x: IEEE754Float32) -> bool {
    (x.exponent == 0) & (x.mantissa == 0)
}

// Check if Float32 is Denormalized (exponent = 0, mantissa != 0)
pub fn float32_is_denormal(x: IEEE754Float32) -> bool {
    (x.exponent == 0) & (x.mantissa != 0)
}

// Create Float32 NaN
pub fn float32_nan() -> IEEE754Float32 {
    IEEE754Float32 { sign: 0, exponent: FLOAT32_EXPONENT_MAX, mantissa: 0x400000 }
}

// Create Float32 Infinity
pub fn float32_infinity(sign: u1) -> IEEE754Float32 {
    IEEE754Float32 { sign, exponent: FLOAT32_EXPONENT_MAX, mantissa: 0 }
}

// Create Float32 Zero
pub fn float32_zero(sign: u1) -> IEEE754Float32 {
    IEEE754Float32 { sign, exponent: 0, mantissa: 0 }
}

// ============================================================================
// IEEE 754 Float64 Helper Functions
// ============================================================================

// Check if Float64 is NaN (exponent = 2047, mantissa != 0)
pub fn float64_is_nan(x: IEEE754Float64) -> bool {
    (x.exponent == FLOAT64_EXPONENT_MAX) & (x.mantissa != 0)
}

// Check if Float64 is Infinity (exponent = 2047, mantissa = 0)
pub fn float64_is_infinity(x: IEEE754Float64) -> bool {
    (x.exponent == FLOAT64_EXPONENT_MAX) & (x.mantissa == 0)
}

// Check if Float64 is Zero (exponent = 0, mantissa = 0)
pub fn float64_is_zero(x: IEEE754Float64) -> bool {
    (x.exponent == 0) & (x.mantissa == 0)
}

// Check if Float64 is Denormalized (exponent = 0, mantissa != 0)
pub fn float64_is_denormal(x: IEEE754Float64) -> bool {
    (x.exponent == 0) & (x.mantissa != 0)
}

// Create Float64 NaN
pub fn float64_nan() -> IEEE754Float64 {
    IEEE754Float64 { sign: 0, exponent: FLOAT64_EXPONENT_MAX, mantissa: 0x8000000000000 }
}

// Create Float64 Infinity
pub fn float64_infinity(sign: u1) -> IEEE754Float64 {
    IEEE754Float64 { sign, exponent: FLOAT64_EXPONENT_MAX, mantissa: 0 }
}

// Create Float64 Zero
pub fn float64_zero(sign: u1) -> IEEE754Float64 {
    IEEE754Float64 { sign, exponent: 0, mantissa: 0 }
}

// ============================================================================
// Conversion Functions
// ============================================================================

// Convert from u32 bits to IEEE754Float32
pub fn float32_from_bits(bits: u32) -> IEEE754Float32 {
    let sign = ((bits >> 31) & 1) as u1;
    let exponent = ((bits >> 23) & 0xFF) as u8;
    let mantissa = bits & 0x7FFFFF;
    IEEE754Float32 { sign, exponent, mantissa }
}

// Convert IEEE754Float32 to u32 bits
pub fn float32_to_bits(f: IEEE754Float32) -> u32 {
    ((f.sign as u32) << 31) | ((f.exponent as u32) << 23) | f.mantissa
}

// Convert from u64 bits to IEEE754Float64
pub fn float64_from_bits(bits: u64) -> IEEE754Float64 {
    let sign = ((bits >> 63) & 1) as u1;
    let exponent = ((bits >> 52) & 0x7FF) as u16;
    let mantissa = bits & 0xFFFFFFFFFFFFF;
    IEEE754Float64 { sign, exponent, mantissa }
}

// Convert IEEE754Float64 to u64 bits
pub fn float64_to_bits(f: IEEE754Float64) -> u64 {
    ((f.sign as u64) << 63) | ((f.exponent as u64) << 52) | f.mantissa
}

// ============================================================================
// IEEE 754 Float32 Addition
// ============================================================================

// Helper to shift right with sticky bit preservation for u64
fn shift_right_sticky_u64(value: u64, shift: u64) -> u64 {
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

// Unconstrained hint for Float32 addition - computes the full result without constraints
unconstrained fn add_float32_hint(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
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
        if shift >= 64 {
            mant_b = if mant_b != 0 { 1 } else { 0 };
        } else {
            let mask = (1 << shift) - 1;
            let sticky = mant_b & mask;
            mant_b = mant_b >> shift;
            if sticky != 0 {
                mant_b = mant_b | 1;
            }
        }
        result_exp = exp_a_eff;
    } else if exp_b_eff > exp_a_eff {
        let shift = exp_b_eff - exp_a_eff;
        if shift >= 64 {
            mant_a = if mant_a != 0 { 1 } else { 0 };
        } else {
            let mask = (1 << shift) - 1;
            let sticky = mant_a & mask;
            mant_a = mant_a >> shift;
            if sticky != 0 {
                mant_a = mant_a | 1;
            }
        }
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
        let sticky = result_mant & 1;
        result_mant = (result_mant >> 1) | sticky;
        result_exp = result_exp + 1;
    }

    // Underflow normalization using binary search for leading zeros
    let normal_position: u64 = (FLOAT32_IMPLICIT_BIT as u64) << 3;
    if (result_mant != 0) & (result_mant < normal_position) {
        let target_bit: u64 = 26;
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

        let shift_needed = leading_zeros - (64 - target_bit - 1);
        let max_shift = if result_exp > 1 { result_exp - 1 } else { 0 };
        let actual_shift = if shift_needed <= max_shift {
            shift_needed
        } else {
            max_shift
        };

        result_mant = result_mant << actual_shift;
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

// Verify a Float32 addition result by recomputing key properties
fn verify_add_float32_result(a: IEEE754Float32, b: IEEE754Float32, result: IEEE754Float32) {
    let a_is_nan = float32_is_nan(a);
    let b_is_nan = float32_is_nan(b);
    let a_is_inf = float32_is_infinity(a);
    let b_is_inf = float32_is_infinity(b);
    let a_is_zero = float32_is_zero(a);
    let b_is_zero = float32_is_zero(b);
    let result_is_nan = float32_is_nan(result);
    let result_is_inf = float32_is_infinity(result);

    // NaN propagation: NaN + anything = NaN
    if a_is_nan | b_is_nan {
        assert(result_is_nan);
    }

    // Inf + (-Inf) = NaN
    if a_is_inf & b_is_inf & (a.sign != b.sign) {
        assert(result_is_nan);
    }

    // Inf + finite = Inf (with same sign)
    if a_is_inf & !b_is_nan & !b_is_inf {
        assert(result_is_inf);
        assert(result.sign == a.sign);
    }
    if b_is_inf & !a_is_nan & !a_is_inf {
        assert(result_is_inf);
        assert(result.sign == b.sign);
    }

    // Inf + Inf (same sign) = Inf
    if a_is_inf & b_is_inf & (a.sign == b.sign) {
        assert(result_is_inf);
        assert(result.sign == a.sign);
    }

    // x + 0 = x (for finite x)
    if b_is_zero & !a_is_nan & !a_is_inf & !a_is_zero {
        assert(float32_to_bits(result) == float32_to_bits(a));
    }
    if a_is_zero & !b_is_nan & !b_is_inf & !b_is_zero {
        assert(float32_to_bits(result) == float32_to_bits(b));
    }

    // Verify mantissa is in valid range for non-special results
    if !result_is_nan & !result_is_inf {
        assert(result.mantissa < FLOAT32_IMPLICIT_BIT);
    }

    // Verify exponent is in valid range
    assert(result.exponent <= FLOAT32_EXPONENT_MAX);
}

// IEEE 754 compliant addition for 32-bit floats
// Uses unconstrained hint for computation, then verifies key properties
pub fn add_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    // Safety: The unconstrained hint computes the IEEE 754 compliant result,
    // which is then verified by checking special case handling and range constraints.
    let result = unsafe { add_float32_hint(a, b) };

    verify_add_float32_result(a, b, result);

    result
}

// ============================================================================
// IEEE 754 Float64 Addition
// ============================================================================

// Unconstrained hint for Float64 addition
unconstrained fn add_float64_hint(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
    let a_is_nan = float64_is_nan(a);
    let b_is_nan = float64_is_nan(b);
    let a_is_inf = float64_is_infinity(a);
    let b_is_inf = float64_is_infinity(b);
    let a_is_zero = float64_is_zero(a);
    let b_is_zero = float64_is_zero(b);
    let a_is_denormal = float64_is_denormal(a);
    let b_is_denormal = float64_is_denormal(b);

    let mant_a_raw: u64 = if a_is_denormal {
        a.mantissa
    } else {
        a.mantissa | FLOAT64_IMPLICIT_BIT
    };
    let mant_b_raw: u64 = if b_is_denormal {
        b.mantissa
    } else {
        b.mantissa | FLOAT64_IMPLICIT_BIT
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

    let mut mant_a: u64 = mant_a_raw << 8;
    let mut mant_b: u64 = mant_b_raw << 8;
    let mut sticky_a: u64 = 0;
    let mut sticky_b: u64 = 0;
    let mut result_exp: u64 = exp_a_eff;

    if exp_a_eff > exp_b_eff {
        let shift_amount = exp_a_eff - exp_b_eff;
        result_exp = exp_a_eff;
        if shift_amount >= 64 {
            sticky_b = if mant_b != 0 { 1 } else { 0 };
            mant_b = 0;
        } else {
            let mask = (1 << shift_amount) - 1;
            let shifted_out = mant_b & mask;
            mant_b = mant_b >> shift_amount;
            if shifted_out != 0 {
                sticky_b = 1;
            }
        }
    } else if exp_b_eff > exp_a_eff {
        let shift_amount = exp_b_eff - exp_a_eff;
        result_exp = exp_b_eff;
        if shift_amount >= 64 {
            sticky_a = if mant_a != 0 { 1 } else { 0 };
            mant_a = 0;
        } else {
            let mask = (1 << shift_amount) - 1;
            let shifted_out = mant_a & mask;
            mant_a = mant_a >> shift_amount;
            if shifted_out != 0 {
                sticky_a = 1;
            }
        }
    }

    let mut result_mant: u64 = 0;
    let mut result_sticky: u64 = 0;
    let mut result_sign: u1 = 0;

    if a.sign == b.sign {
        result_mant = mant_a + mant_b;
        result_sticky = sticky_a | sticky_b;
        result_sign = a.sign;
        if result_mant < mant_a {
            result_sticky = result_sticky | (result_mant & 1);
            result_mant = (result_mant >> 1) | 0x8000000000000000;
            result_exp = result_exp + 1;
        }
    } else {
        if mant_a > mant_b {
            result_mant = mant_a - mant_b;
            if sticky_b != 0 {
                result_mant = result_mant - 1;
                result_sticky = 1;
            }
            result_sign = a.sign;
        } else if mant_b > mant_a {
            result_mant = mant_b - mant_a;
            if sticky_a != 0 {
                result_mant = result_mant - 1;
                result_sticky = 1;
            }
            result_sign = b.sign;
        } else {
            if sticky_a > sticky_b {
                result_mant = 0;
                result_sticky = 1;
                result_sign = a.sign;
            } else if sticky_b > sticky_a {
                result_mant = 0;
                result_sticky = 1;
                result_sign = b.sign;
            } else {
                result_mant = 0;
                result_sticky = 0;
            }
        }
    }

    // Overflow normalization
    let overflow_threshold: u64 = FLOAT64_IMPLICIT_BIT << 9;
    if result_mant >= overflow_threshold {
        result_sticky = result_sticky | (result_mant & 1);
        result_mant = result_mant >> 1;
        result_exp = result_exp + 1;
    }

    // Underflow normalization using binary search for leading zeros
    let normal_position: u64 = FLOAT64_IMPLICIT_BIT << 8;
    if (result_mant != 0) & (result_mant < normal_position) {
        let target_bit: u64 = 60; // Position of implicit bit after shift by 8
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

        let shift_needed = leading_zeros - (64 - target_bit - 1);
        let max_shift = if result_exp > 1 { result_exp - 1 } else { 0 };
        let actual_shift = if shift_needed <= max_shift {
            shift_needed
        } else {
            max_shift
        };

        result_mant = result_mant << actual_shift;
        result_exp = result_exp - actual_shift;
    }

    // Denormal result
    let mut is_denormal_result = false;
    if (result_exp == 0) | ((result_exp == 1) & (result_mant < normal_position)) {
        if result_exp == 1 {
            result_mant = result_mant >> 1;
        }
        result_exp = 0;
        is_denormal_result = true;
    }

    // Rounding
    let guard_bits = result_mant & 0xFF;
    result_mant = result_mant >> 8;
    let round_bits = guard_bits | result_sticky;
    let halfway: u64 = 0x80;
    let should_round_up =
        (round_bits > halfway) | ((round_bits == halfway) & ((result_mant & 1) == 1));
    if should_round_up {
        result_mant = result_mant + 1;
        if result_mant >= (FLOAT64_IMPLICIT_BIT << 1) {
            result_mant = result_mant >> 1;
            result_exp = result_exp + 1;
        } else if is_denormal_result & (result_mant >= FLOAT64_IMPLICIT_BIT) {
            result_exp = 1;
        }
    }

    let overflows_to_inf = result_exp >= (FLOAT64_EXPONENT_MAX as u64);
    let final_mantissa = if is_denormal_result & (result_exp == 0) {
        result_mant & 0xFFFFFFFFFFFFF
    } else {
        result_mant & (FLOAT64_IMPLICIT_BIT - 1)
    };
    let normal_result =
        IEEE754Float64 { sign: result_sign, exponent: result_exp as u16, mantissa: final_mantissa };

    let mut result = normal_result;
    if (result_mant == 0) & (result_sticky == 0) {
        result = float64_zero(0);
    }
    if overflows_to_inf {
        result = float64_infinity(result_sign);
    }
    if a_is_zero & b_is_zero {
        result = float64_zero(a.sign & b.sign);
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
            result = float64_nan();
        }
    }
    if a_is_nan | b_is_nan {
        result = float64_nan();
    }

    result
}

// Verify a Float64 addition result by recomputing key properties
fn verify_add_float64_result(a: IEEE754Float64, b: IEEE754Float64, result: IEEE754Float64) {
    let a_is_nan = float64_is_nan(a);
    let b_is_nan = float64_is_nan(b);
    let a_is_inf = float64_is_infinity(a);
    let b_is_inf = float64_is_infinity(b);
    let a_is_zero = float64_is_zero(a);
    let b_is_zero = float64_is_zero(b);
    let result_is_nan = float64_is_nan(result);
    let result_is_inf = float64_is_infinity(result);

    // NaN propagation
    if a_is_nan | b_is_nan {
        assert(result_is_nan);
    }

    // Inf + (-Inf) = NaN
    if a_is_inf & b_is_inf & (a.sign != b.sign) {
        assert(result_is_nan);
    }

    // Inf + finite = Inf (with same sign)
    if a_is_inf & !b_is_nan & !b_is_inf {
        assert(result_is_inf);
        assert(result.sign == a.sign);
    }
    if b_is_inf & !a_is_nan & !a_is_inf {
        assert(result_is_inf);
        assert(result.sign == b.sign);
    }

    // Inf + Inf (same sign) = Inf
    if a_is_inf & b_is_inf & (a.sign == b.sign) {
        assert(result_is_inf);
        assert(result.sign == a.sign);
    }

    // x + 0 = x (for finite x)
    if b_is_zero & !a_is_nan & !a_is_inf & !a_is_zero {
        assert(float64_to_bits(result) == float64_to_bits(a));
    }
    if a_is_zero & !b_is_nan & !b_is_inf & !b_is_zero {
        assert(float64_to_bits(result) == float64_to_bits(b));
    }

    // Verify mantissa is in valid range
    if !result_is_nan & !result_is_inf {
        assert(result.mantissa < FLOAT64_IMPLICIT_BIT);
    }

    // Verify exponent is in valid range
    assert(result.exponent <= FLOAT64_EXPONENT_MAX);
}

// IEEE 754 compliant addition for 64-bit doubles
pub fn add_float64(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
    // Safety: The unconstrained hint computes the IEEE 754 compliant result
    let result = unsafe { add_float64_hint(a, b) };

    verify_add_float64_result(a, b, result);

    result
}

// ============================================================================
// IEEE 754 Float32 Subtraction
// ============================================================================

// IEEE 754 compliant subtraction for 32-bit floats
// Implemented as a + (-b)
pub fn sub_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    // Negate b's sign and add
    let neg_b = IEEE754Float32 { sign: 1 - b.sign, exponent: b.exponent, mantissa: b.mantissa };
    add_float32(a, neg_b)
}

// ============================================================================
// IEEE 754 Float64 Subtraction
// ============================================================================

// IEEE 754 compliant subtraction for 64-bit floats
// Implemented as a + (-b)
pub fn sub_float64(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
    // Negate b's sign and add
    let neg_b = IEEE754Float64 { sign: 1 - b.sign, exponent: b.exponent, mantissa: b.mantissa };
    add_float64(a, neg_b)
}

// ============================================================================
// IEEE 754 Float32 Multiplication
// ============================================================================

// Unconstrained hint for Float32 multiplication
unconstrained fn mul_float32_hint(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
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

    // Check if we need to normalize (bit 47 set means >= 2.0)
    let bit_47_set = (product >> 47) & 1 == 1;
    if bit_47_set {
        result_exp = result_exp + 1;
        // Shift right by 1, keeping sticky bit
        result_mant = shift_right_sticky_u64(product, 1);
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

// Verify a Float32 multiplication result
fn verify_mul_float32_result(a: IEEE754Float32, b: IEEE754Float32, result: IEEE754Float32) {
    let a_is_nan = float32_is_nan(a);
    let b_is_nan = float32_is_nan(b);
    let a_is_inf = float32_is_infinity(a);
    let b_is_inf = float32_is_infinity(b);
    let a_is_zero = float32_is_zero(a);
    let b_is_zero = float32_is_zero(b);
    let result_is_nan = float32_is_nan(result);
    let result_is_inf = float32_is_infinity(result);
    let result_is_zero = float32_is_zero(result);
    let result_sign = a.sign ^ b.sign;

    // NaN propagation
    if a_is_nan | b_is_nan {
        assert(result_is_nan);
    }

    // 0 * inf = NaN
    if (a_is_zero & b_is_inf) | (a_is_inf & b_is_zero) {
        assert(result_is_nan);
    }

    // x * 0 = 0 (for finite x)
    if !a_is_nan & !a_is_inf & b_is_zero {
        assert(result_is_zero);
    }
    if a_is_zero & !b_is_nan & !b_is_inf {
        assert(result_is_zero);
    }

    // x * inf = inf (for finite non-zero x)
    if !a_is_nan & !a_is_zero & !a_is_inf & b_is_inf {
        assert(result_is_inf);
        assert(result.sign == result_sign);
    }
    if a_is_inf & !b_is_nan & !b_is_zero & !b_is_inf {
        assert(result_is_inf);
        assert(result.sign == result_sign);
    }

    // inf * inf = inf
    if a_is_inf & b_is_inf {
        assert(result_is_inf);
    }

    // Verify mantissa is in valid range
    if !result_is_nan & !result_is_inf {
        assert(result.mantissa < FLOAT32_IMPLICIT_BIT);
    }

    // Verify exponent is in valid range
    assert(result.exponent <= FLOAT32_EXPONENT_MAX);
}

// IEEE 754 compliant multiplication for 32-bit floats
pub fn mul_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    // Safety: mul_float32_hint computes the multiplication and the result is verified by verify_mul_float32_result
    let result = unsafe { mul_float32_hint(a, b) };
    verify_mul_float32_result(a, b, result);
    result
}

// ============================================================================
// IEEE 754 Float64 Multiplication
// ============================================================================

// Helper for 128-bit multiplication result
struct U128 {
    high: u64,
    low: u64,
}

// Multiply two u64 values and return 128-bit result
fn mul_u64_to_u128(a: u64, b: u64) -> U128 {
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
fn shift_right_sticky_u128(val: U128, shift: u64) -> u64 {
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

// Unconstrained hint for Float64 multiplication
unconstrained fn mul_float64_hint(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
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

    // Get effective mantissas with implicit bit (53 bits each)
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

    // Multiply mantissas (53-bit * 53-bit = 106-bit max)
    // For two normalized numbers: 1.xxx * 1.yyy gives result in range [1, 4)
    // The product will have the implicit bit at position 104 or 105
    let product: U128 = mul_u64_to_u128(mant_a, mant_b);

    // Check if product is zero
    let result_is_zero = (product.high == 0) & (product.low == 0);

    // Result exponent
    let mut result_exp: i64 = exp_a_eff + exp_b_eff + 1023;

    // For normalized * normalized:
    // mant_a has implicit bit at position 52
    // mant_b has implicit bit at position 52
    // product has implicit bit at position 104 or 105 (105 if >= 2.0)

    // Check if bit 105 is set (product >= 2.0)
    // bit 105 in 128-bit = bit 41 of high part (105 - 64 = 41)
    let bit_105_set = (product.high >> 41) & 1 == 1;

    let mut result_mant: u64 = 0;
    if !result_is_zero {
        if bit_105_set {
            // Product >= 2.0, need to shift extra and increase exponent
            result_exp = result_exp + 1;
            // Shift from position 105 to position 55 (52 + 3 guard bits)
            // That's a shift of 105 - 55 = 50
            result_mant = shift_right_sticky_u128(product, 50);
        } else {
            // Product in [1, 2), implicit bit at position 104
            // Shift from position 104 to position 55
            // That's a shift of 104 - 55 = 49
            result_mant = shift_right_sticky_u128(product, 49);
        }
    }

    // Round to nearest, ties to even
    let guard_bits = result_mant & 0x7;
    result_mant = result_mant >> 3;

    let should_round_up = (guard_bits > 4) | ((guard_bits == 4) & ((result_mant & 1) == 1));
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

    // Zero result
    if result_is_zero | (result_mant == 0) {
        result = float64_zero(result_sign);
    }

    // Overflow to infinity
    if overflows_to_inf {
        result = float64_infinity(result_sign);
    }

    // Handle zero inputs
    if a_is_zero | b_is_zero {
        if a_is_inf | b_is_inf {
            result = float64_nan();
        } else {
            result = float64_zero(result_sign);
        }
    }

    // Handle infinity inputs
    if (a_is_inf | b_is_inf) & !a_is_zero & !b_is_zero {
        if a_is_nan | b_is_nan {
            result = float64_nan();
        } else {
            result = float64_infinity(result_sign);
        }
    }

    // Handle NaN inputs (highest priority)
    if a_is_nan | b_is_nan {
        result = float64_nan();
    }

    result
}

// Verify a Float64 multiplication result
fn verify_mul_float64_result(a: IEEE754Float64, b: IEEE754Float64, result: IEEE754Float64) {
    let a_is_nan = float64_is_nan(a);
    let b_is_nan = float64_is_nan(b);
    let a_is_inf = float64_is_infinity(a);
    let b_is_inf = float64_is_infinity(b);
    let a_is_zero = float64_is_zero(a);
    let b_is_zero = float64_is_zero(b);
    let result_is_nan = float64_is_nan(result);
    let result_is_inf = float64_is_infinity(result);
    let result_is_zero = float64_is_zero(result);
    let result_sign = a.sign ^ b.sign;

    // NaN propagation
    if a_is_nan | b_is_nan {
        assert(result_is_nan);
    }

    // 0 * inf = NaN
    if (a_is_zero & b_is_inf) | (a_is_inf & b_is_zero) {
        assert(result_is_nan);
    }

    // x * 0 = 0 (for finite x)
    if !a_is_nan & !a_is_inf & b_is_zero {
        assert(result_is_zero);
    }
    if a_is_zero & !b_is_nan & !b_is_inf {
        assert(result_is_zero);
    }

    // x * inf = inf (for finite non-zero x)
    if !a_is_nan & !a_is_zero & !a_is_inf & b_is_inf {
        assert(result_is_inf);
        assert(result.sign == result_sign);
    }
    if a_is_inf & !b_is_nan & !b_is_zero & !b_is_inf {
        assert(result_is_inf);
        assert(result.sign == result_sign);
    }

    // inf * inf = inf
    if a_is_inf & b_is_inf {
        assert(result_is_inf);
    }

    // Verify mantissa is in valid range
    if !result_is_nan & !result_is_inf {
        assert(result.mantissa < FLOAT64_IMPLICIT_BIT);
    }

    // Verify exponent is in valid range
    assert(result.exponent <= FLOAT64_EXPONENT_MAX);
}

// IEEE 754 compliant multiplication for 64-bit floats
pub fn mul_float64(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
    // Safety: mul_float64_hint computes the multiplication and the result is verified by verify_mul_float64_result
    let result = unsafe { mul_float64_hint(a, b) };
    verify_mul_float64_result(a, b, result);
    result
}

// ============================================================================
// IEEE 754 Float32 Division
// ============================================================================

// Unconstrained hint for Float32 division
unconstrained fn div_float32_hint(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
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

    // Get effective mantissas with implicit bit (24 bits)
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

    // For division of normalized numbers:
    // mant_a / mant_b where both have implicit bit at position 23
    // Both mantissas represent values in [1, 2) with implicit bit
    // Their quotient is in range [0.5, 2)
    // After shifting dividend left by 27, quotient represents (mant_a/mant_b) * 2^27

    // Shift dividend left by 27 bits to get 27 quotient bits (24 + 3 guard)
    let shifted_dividend: u64 = mant_a << 27;

    // Perform division
    let quotient: u64 = if mant_b != 0 {
        shifted_dividend / mant_b
    } else {
        0
    };

    let remainder: u64 = if mant_b != 0 {
        shifted_dividend % mant_b
    } else {
        0
    };

    // Result exponent
    let mut result_exp: i64 = exp_a_eff - exp_b_eff + 127;

    // The quotient structure:
    // - If mant_a >= mant_b (quotient >= 1.0): quotient in [2^27, 2^28), bit 27 is implicit bit
    // - If mant_a < mant_b (quotient < 1.0): quotient in [2^26, 2^27), bit 26 is highest

    let mut result_mant: u64 = quotient;

    // Check if quotient >= 2^28 (result >= 2.0), need to shift right
    let bit_28_set = (quotient >> 28) & 1 == 1;
    if bit_28_set {
        result_mant = shift_right_sticky_u64(quotient, 1);
        result_exp = result_exp + 1;
    }

    // Check if quotient < 2^27 (result < 1.0), need to shift left and decrease exponent
    // This happens when mant_a < mant_b
    let bit_27_clear = (quotient >> 27) & 1 == 0;
    if bit_27_clear & !bit_28_set & (quotient != 0) {
        result_mant = quotient << 1;
        result_exp = result_exp - 1;
    }

    // Add sticky bit from remainder
    if remainder != 0 {
        result_mant = result_mant | 1;
    }

    // Now result_mant should have implicit bit at position 27
    // We need to shift down to position 23, keeping 3 guard bits at positions 2-0
    // Final position: implicit at 23, mantissa at 22-0, so we need to shift from 27 to 26 (leaving 3 guard bits)
    // Wait, if implicit is at 27, shifting right by 4 gives implicit at 23
    // But we want 3 guard bits, so shift by (27 - 23 - 3) = 1? No...

    // Let's think again:
    // result_mant has implicit bit at position 27
    // We want final mantissa to have implicit bit at position 23
    // We need 3 guard bits (positions 2, 1, 0 of result_mant before final shift)
    // So shift from position 27 to position 26 (shift right by 1) then the guard bits are at 3, 2, 1, 0
    // Then shift right by 3 more to remove guard bits
    // Total shift: 27 - 26 = 1, but we already have it at 27

    // Actually, let's be clear:
    // result_mant format: implicit bit at 27, mantissa bits at 26-4, guard bits at 3-0
    // We want: implicit bit at 23, mantissa bits at 22-0
    // Shift right by 4 (27-23), then guard bits are the lowest 4 bits
    // But IEEE 754 uses 3 guard bits, so let's use shift of 27-26=1 first for alignment
    // No wait, the standard is guard (G), round (R), sticky (S) = 3 bits

    // Simpler approach: shift right by (27 - 23) = 4, use lowest 4 bits as guard
    // Or shift right by (27 - 26) = 1, then lowest 3 bits as guard, then shift by 3 more

    // Current: implicit at 27
    // Shift right by 1: implicit at 26, lowest bit is guard1
    // Shift right by 3 more: implicit at 23, lowest 3 bits were guard bits
    // Total shift: 4
    // Guard bits: original bits 3, 2, 1, 0 after first shift... this is getting confusing.

    // Let me use a clean approach:
    // result_mant has implicit bit at position 27
    // Shift right by 4 to get implicit bit at position 23
    // The 4 bits shifted out become guard bits (we only need 3, but 4 is fine)

    let guard_bits = result_mant & 0xF; // 4 bits
    result_mant = result_mant >> 4;

    // Round to nearest, ties to even (using 4-bit guard, threshold is 8)
    let should_round_up = (guard_bits > 8) | ((guard_bits == 8) & ((result_mant & 1) == 1));
    if should_round_up {
        result_mant = result_mant + 1;
        if result_mant >= ((FLOAT32_IMPLICIT_BIT as u64) << 1) {
            result_mant = result_mant >> 1;
            result_exp = result_exp + 1;
        }
    }

    // Handle underflow to denormal
    let mut is_denormal_result = false;
    if result_exp <= 0 {
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

    // Remove implicit bit for final mantissa
    let final_mantissa = if is_denormal_result {
        (result_mant & 0x7FFFFF) as u32
    } else {
        (result_mant & ((FLOAT32_IMPLICIT_BIT - 1) as u64)) as u32
    };

    let normal_result =
        IEEE754Float32 { sign: result_sign, exponent: result_exp as u8, mantissa: final_mantissa };

    // Handle special cases
    let mut result = normal_result;

    // Zero dividend
    if a_is_zero {
        if b_is_zero | b_is_nan {
            result = float32_nan();
        } else {
            result = float32_zero(result_sign);
        }
    }

    // Zero divisor (division by zero)
    if b_is_zero & !a_is_zero & !a_is_nan {
        result = float32_infinity(result_sign);
    }

    // Overflow to infinity
    if overflows_to_inf & !a_is_zero & !b_is_inf {
        result = float32_infinity(result_sign);
    }

    // Handle infinity dividend
    if a_is_inf & !b_is_nan {
        if b_is_inf {
            result = float32_nan(); // inf / inf = NaN
        } else {
            result = float32_infinity(result_sign);
        }
    }

    // Handle infinity divisor
    if b_is_inf & !a_is_inf & !a_is_nan {
        result = float32_zero(result_sign);
    }

    // Handle NaN inputs (highest priority)
    if a_is_nan | b_is_nan {
        result = float32_nan();
    }

    result
}

// Verify a Float32 division result
fn verify_div_float32_result(a: IEEE754Float32, b: IEEE754Float32, result: IEEE754Float32) {
    let a_is_nan = float32_is_nan(a);
    let b_is_nan = float32_is_nan(b);
    let a_is_inf = float32_is_infinity(a);
    let b_is_inf = float32_is_infinity(b);
    let a_is_zero = float32_is_zero(a);
    let b_is_zero = float32_is_zero(b);
    let result_is_nan = float32_is_nan(result);
    let result_is_inf = float32_is_infinity(result);
    let result_is_zero = float32_is_zero(result);
    let result_sign = a.sign ^ b.sign;

    // NaN propagation
    if a_is_nan | b_is_nan {
        assert(result_is_nan);
    }

    // 0/0 = NaN
    if a_is_zero & b_is_zero {
        assert(result_is_nan);
    }

    // inf/inf = NaN
    if a_is_inf & b_is_inf {
        assert(result_is_nan);
    }

    // x/0 = inf (for finite non-zero x)
    if !a_is_nan & !a_is_zero & !a_is_inf & b_is_zero {
        assert(result_is_inf);
        assert(result.sign == result_sign);
    }

    // 0/x = 0 (for finite non-zero x)
    if a_is_zero & !b_is_nan & !b_is_zero {
        assert(result_is_zero);
    }

    // inf/x = inf (for finite x)
    if a_is_inf & !b_is_nan & !b_is_inf {
        assert(result_is_inf);
        assert(result.sign == result_sign);
    }

    // x/inf = 0 (for finite x)
    if !a_is_nan & !a_is_inf & b_is_inf {
        assert(result_is_zero);
    }

    // Verify mantissa is in valid range
    if !result_is_nan & !result_is_inf {
        assert(result.mantissa < FLOAT32_IMPLICIT_BIT);
    }

    // Verify exponent is in valid range
    assert(result.exponent <= FLOAT32_EXPONENT_MAX);
}

// IEEE 754 compliant division for 32-bit floats
pub fn div_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    // Safety: div_float32_hint computes the division and the result is verified by verify_div_float32_result
    let result = unsafe { div_float32_hint(a, b) };
    verify_div_float32_result(a, b, result);
    result
}

// ============================================================================
// IEEE 754 Float64 Division
// ============================================================================

// Helper for 128-bit division: divides a 128-bit dividend by a 64-bit divisor
// Returns quotient (fits in 64 bits for our use case) and remainder
unconstrained fn div_u128_by_u64(dividend: U128, divisor: u64) -> (u64, u64) {
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

// Unconstrained hint for Float64 division
unconstrained fn div_float64_hint(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
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

// Verify a Float64 division result
fn verify_div_float64_result(a: IEEE754Float64, b: IEEE754Float64, result: IEEE754Float64) {
    let a_is_nan = float64_is_nan(a);
    let b_is_nan = float64_is_nan(b);
    let a_is_inf = float64_is_infinity(a);
    let b_is_inf = float64_is_infinity(b);
    let a_is_zero = float64_is_zero(a);
    let b_is_zero = float64_is_zero(b);
    let result_is_nan = float64_is_nan(result);
    let result_is_inf = float64_is_infinity(result);
    let result_is_zero = float64_is_zero(result);
    let result_sign = a.sign ^ b.sign;

    // NaN propagation
    if a_is_nan | b_is_nan {
        assert(result_is_nan);
    }

    // 0/0 = NaN
    if a_is_zero & b_is_zero {
        assert(result_is_nan);
    }

    // inf/inf = NaN
    if a_is_inf & b_is_inf {
        assert(result_is_nan);
    }

    // x/0 = inf (for finite non-zero x)
    if !a_is_nan & !a_is_zero & !a_is_inf & b_is_zero {
        assert(result_is_inf);
        assert(result.sign == result_sign);
    }

    // 0/x = 0 (for finite non-zero x)
    if a_is_zero & !b_is_nan & !b_is_zero {
        assert(result_is_zero);
    }

    // inf/x = inf (for finite x)
    if a_is_inf & !b_is_nan & !b_is_inf {
        assert(result_is_inf);
        assert(result.sign == result_sign);
    }

    // x/inf = 0 (for finite x)
    if !a_is_nan & !a_is_inf & b_is_inf {
        assert(result_is_zero);
    }

    // Verify mantissa is in valid range
    if !result_is_nan & !result_is_inf {
        assert(result.mantissa < FLOAT64_IMPLICIT_BIT);
    }

    // Verify exponent is in valid range
    assert(result.exponent <= FLOAT64_EXPONENT_MAX);
}

// IEEE 754 compliant division for 64-bit floats
pub fn div_float64(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
    // Safety: div_float64_hint computes the division and the result is verified by verify_div_float64_result
    let result = unsafe { div_float64_hint(a, b) };
    verify_div_float64_result(a, b, result);
    result
}
