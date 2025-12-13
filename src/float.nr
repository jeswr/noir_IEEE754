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

// IEEE 754 compliant addition for 32-bit floats
// Implements proper alignment, normalization, and rounding (round to nearest, ties to even)
pub fn add_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    // Determine special cases with flags
    let a_is_nan = float32_is_nan(a);
    let b_is_nan = float32_is_nan(b);
    let a_is_inf = float32_is_infinity(a);
    let b_is_inf = float32_is_infinity(b);
    let a_is_zero = float32_is_zero(a);
    let b_is_zero = float32_is_zero(b);
    let a_is_denormal = float32_is_denormal(a);
    let b_is_denormal = float32_is_denormal(b);

    // Calculate result for the normal case first
    // Get the effective mantissa including implicit bit for normalized numbers
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

    // Get effective exponents (denormals have effective exponent of 1)
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

    // Shift mantissas left to provide guard, round, and sticky bits (3 extra bits)
    let mut mant_a: u64 = mant_a_raw << 3;
    let mut mant_b: u64 = mant_b_raw << 3;

    // Align mantissas by shifting the smaller exponent's mantissa right
    let mut result_exp: u64 = exp_a_eff;

    if exp_a_eff > exp_b_eff {
        let shift_amount = exp_a_eff - exp_b_eff;
        result_exp = exp_a_eff;
        mant_b = shift_right_sticky_u64(mant_b, shift_amount);
    } else if exp_b_eff > exp_a_eff {
        let shift_amount = exp_b_eff - exp_a_eff;
        result_exp = exp_b_eff;
        mant_a = shift_right_sticky_u64(mant_a, shift_amount);
    }

    // Perform addition or subtraction based on signs
    let mut result_mant: u64 = 0;
    let mut result_sign: u1 = 0;
    let same_sign = a.sign == b.sign;

    if same_sign {
        // Same sign: add mantissas
        result_mant = mant_a + mant_b;
        result_sign = a.sign;
    } else {
        // Different signs: subtract mantissas
        if mant_a >= mant_b {
            result_mant = mant_a - mant_b;
            result_sign = a.sign;
        } else {
            result_mant = mant_b - mant_a;
            result_sign = b.sign;
        }
    }

    // Normalize the result
    // First, handle overflow from addition (mantissa >= 2 * implicit bit position)
    let overflow_threshold: u64 = (FLOAT32_IMPLICIT_BIT as u64) << 4; // Account for 3 guard bits + 1 for overflow
    if result_mant >= overflow_threshold {
        // Shift right and increment exponent
        result_mant = shift_right_sticky_u64(result_mant, 1);
        result_exp = result_exp + 1;
    }

    // Handle underflow (leading zeros) - shift left until normalized
    let normal_position: u64 = (FLOAT32_IMPLICIT_BIT as u64) << 3;

    for _ in 0..26 {
        if (result_mant != 0) & (result_mant < normal_position) & (result_exp > 1) {
            result_mant = result_mant << 1;
            result_exp = result_exp - 1;
        }
    }

    // Check for underflow to denormal or zero
    // If result_exp == 1 and result_mant < normal_position, the result is naturally
    // denormal (no implicit bit present), so we just set exp = 0 without shifting.
    let mut is_denormal_result = false;
    if (result_exp == 0) | ((result_exp == 1) & (result_mant < normal_position)) {
        result_exp = 0;
        is_denormal_result = true;
    }

    // Round to nearest, ties to even
    let guard_bits = result_mant & 0x7; // Lower 3 bits (guard, round, sticky)
    result_mant = result_mant >> 3;

    // Round up if guard bits > 0b100, or if guard bits == 0b100 and LSB is 1 (ties to even)
    let should_round_up = (guard_bits > 4) | ((guard_bits == 4) & ((result_mant & 1) == 1));
    if should_round_up {
        result_mant = result_mant + 1;

        // Check if rounding caused overflow
        if result_mant >= ((FLOAT32_IMPLICIT_BIT as u64) << 1) {
            result_mant = result_mant >> 1;
            result_exp = result_exp + 1;
        } else if is_denormal_result & (result_mant >= (FLOAT32_IMPLICIT_BIT as u64)) {
            // Denormal became normal through rounding
            result_exp = 1;
        }
    }

    // Check for overflow to infinity
    let overflows_to_inf = result_exp >= (FLOAT32_EXPONENT_MAX as u64);

    // Remove implicit bit for final mantissa (only for normalized numbers)
    let final_mantissa = if is_denormal_result & (result_exp == 0) {
        (result_mant & 0x7FFFFF) as u32
    } else {
        (result_mant & ((FLOAT32_IMPLICIT_BIT - 1) as u64)) as u32
    };

    // Compute normal result
    let normal_result =
        IEEE754Float32 { sign: result_sign, exponent: result_exp as u8, mantissa: final_mantissa };

    // Build final result based on special case flags
    // Priority: NaN > Infinity > Zero > Normal

    // Result for NaN cases
    let nan_result = float32_nan();

    // Result for infinity + (-infinity) = NaN
    let inf_opposite_signs_result = float32_nan();

    // Determine final result using conditionals without early returns
    let mut result = normal_result;

    // Zero result check
    if result_mant == 0 {
        result = float32_zero(0);
    }

    // Overflow to infinity
    if overflows_to_inf {
        result = float32_infinity(result_sign);
    }

    // Handle zero inputs
    if a_is_zero & b_is_zero {
        // -0 + -0 = -0, otherwise +0
        result = float32_zero(a.sign & b.sign);
    } else if a_is_zero {
        result = b;
    } else if b_is_zero {
        result = a;
    }

    // Handle infinity inputs
    if a_is_inf & !b_is_inf {
        result = a;
    } else if b_is_inf & !a_is_inf {
        result = b;
    } else if a_is_inf & b_is_inf {
        if a.sign == b.sign {
            result = a;
        } else {
            result = inf_opposite_signs_result;
        }
    }

    // Handle NaN inputs (highest priority)
    if a_is_nan | b_is_nan {
        result = nan_result;
    }

    result
}

// ============================================================================
// IEEE 754 Float64 Addition
// ============================================================================

// IEEE 754 compliant addition for 64-bit doubles
// Implements proper alignment, normalization, and rounding (round to nearest, ties to even)
pub fn add_float64(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
    // Determine special cases with flags
    let a_is_nan = float64_is_nan(a);
    let b_is_nan = float64_is_nan(b);
    let a_is_inf = float64_is_infinity(a);
    let b_is_inf = float64_is_infinity(b);
    let a_is_zero = float64_is_zero(a);
    let b_is_zero = float64_is_zero(b);
    let a_is_denormal = float64_is_denormal(a);
    let b_is_denormal = float64_is_denormal(b);

    // Get the effective mantissa including implicit bit for normalized numbers
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

    // Get effective exponents (denormals have effective exponent of 1)
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

    // Shift mantissas left to provide guard, round, and sticky bits
    // For float64, we have 52-bit mantissa + 1 implicit bit = 53 bits
    // We can only shift by ~10 bits to stay within u64
    // We'll use 8 bits for guard/round/sticky
    let mut mant_a: u64 = mant_a_raw << 8;
    let mut mant_b: u64 = mant_b_raw << 8;
    let mut sticky_a: u64 = 0;
    let mut sticky_b: u64 = 0;

    // Align mantissas by shifting the smaller exponent's mantissa right
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

    // Perform addition or subtraction based on signs
    let mut result_mant: u64 = 0;
    let mut result_sticky: u64 = 0;
    let mut result_sign: u1 = 0;
    let same_sign = a.sign == b.sign;

    if same_sign {
        // Same sign: add mantissas
        result_mant = mant_a + mant_b;
        result_sticky = sticky_a | sticky_b;
        result_sign = a.sign;

        // Check for carry/overflow (if result < either operand due to wrap)
        if result_mant < mant_a {
            // Overflow in u64, shift right and adjust
            result_sticky = result_sticky | (result_mant & 1);
            result_mant = (result_mant >> 1) | 0x8000000000000000;
            result_exp = result_exp + 1;
        }
    } else {
        // Different signs: subtract mantissas
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
            // Mantissas are equal, check sticky bits
            if sticky_a > sticky_b {
                result_mant = 0;
                result_sticky = 1;
                result_sign = a.sign;
            } else if sticky_b > sticky_a {
                result_mant = 0;
                result_sticky = 1;
                result_sign = b.sign;
            } else {
                // Complete cancellation
                result_mant = 0;
                result_sticky = 0;
            }
        }
    }

    // Normalize the result
    // First, handle overflow from addition
    let overflow_threshold: u64 = FLOAT64_IMPLICIT_BIT << 9; // Account for 8 guard bits + 1 for overflow
    if result_mant >= overflow_threshold {
        result_sticky = result_sticky | (result_mant & 1);
        result_mant = result_mant >> 1;
        result_exp = result_exp + 1;
    }

    // Handle underflow (leading zeros) - shift left until normalized
    let normal_position: u64 = FLOAT64_IMPLICIT_BIT << 8;

    for _ in 0..60 {
        if (result_mant != 0) & (result_mant < normal_position) & (result_exp > 1) {
            result_mant = result_mant << 1;
            result_exp = result_exp - 1;
        }
    }

    // Check for underflow to denormal or zero
    let mut is_denormal_result = false;
    if (result_exp == 0) | ((result_exp == 1) & (result_mant < normal_position)) {
        if result_exp == 1 {
            result_mant = result_mant >> 1;
        }
        result_exp = 0;
        is_denormal_result = true;
    }

    // Round to nearest, ties to even
    // Extract guard bits (we shifted by 8, so lower 8 bits are guard/round/sticky)
    let guard_bits = result_mant & 0xFF;
    result_mant = result_mant >> 8;

    // Combine with sticky bit
    let round_bits = guard_bits | result_sticky;
    let halfway: u64 = 0x80; // 128 = 0b10000000

    // Round up if > halfway, or if exactly halfway and LSB is 1 (ties to even)
    let should_round_up =
        (round_bits > halfway) | ((round_bits == halfway) & ((result_mant & 1) == 1));
    if should_round_up {
        result_mant = result_mant + 1;

        // Check if rounding caused overflow
        if result_mant >= (FLOAT64_IMPLICIT_BIT << 1) {
            result_mant = result_mant >> 1;
            result_exp = result_exp + 1;
        } else if is_denormal_result & (result_mant >= FLOAT64_IMPLICIT_BIT) {
            // Denormal became normal through rounding
            result_exp = 1;
        }
    }

    // Check for overflow to infinity
    let overflows_to_inf = result_exp >= (FLOAT64_EXPONENT_MAX as u64);

    // Remove implicit bit for final mantissa (only for normalized numbers)
    let final_mantissa = if is_denormal_result & (result_exp == 0) {
        result_mant & 0xFFFFFFFFFFFFF
    } else {
        result_mant & (FLOAT64_IMPLICIT_BIT - 1)
    };

    // Compute normal result
    let normal_result =
        IEEE754Float64 { sign: result_sign, exponent: result_exp as u16, mantissa: final_mantissa };

    // Build final result based on special case flags
    let nan_result = float64_nan();
    let inf_opposite_signs_result = float64_nan();

    let mut result = normal_result;

    // Zero result check
    if (result_mant == 0) & (result_sticky == 0) {
        result = float64_zero(0);
    }

    // Overflow to infinity
    if overflows_to_inf {
        result = float64_infinity(result_sign);
    }

    // Handle zero inputs
    if a_is_zero & b_is_zero {
        // -0 + -0 = -0, otherwise +0
        result = float64_zero(a.sign & b.sign);
    } else if a_is_zero {
        result = b;
    } else if b_is_zero {
        result = a;
    }

    // Handle infinity inputs
    if a_is_inf & !b_is_inf {
        result = a;
    } else if b_is_inf & !a_is_inf {
        result = b;
    } else if a_is_inf & b_is_inf {
        if a.sign == b.sign {
            result = a;
        } else {
            result = inf_opposite_signs_result;
        }
    }

    // Handle NaN inputs (highest priority)
    if a_is_nan | b_is_nan {
        result = nan_result;
    }

    result
}
