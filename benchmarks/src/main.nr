// IEEE 754 Floating Point Benchmarks
// Run with: cd benchmarks && nargo info
// Or profile with: cd benchmarks && nargo compile && noir-profiler gates --artifact-path ./target/ieee754_benchmarks.json --backend-path bb --output ./target

// ============================================================================
// IEEE 754 Floating Point Types (copied from library for standalone benchmarking)
// ============================================================================

struct IEEE754Float32 {
    sign: u1,
    exponent: u8,
    mantissa: u32,
}

struct IEEE754Float64 {
    sign: u1,
    exponent: u16,
    mantissa: u64,
}

// Constants
global FLOAT32_EXPONENT_MAX: u8 = 255;
global FLOAT32_IMPLICIT_BIT: u32 = 0x800000;
global FLOAT64_EXPONENT_MAX: u16 = 2047;
global FLOAT64_IMPLICIT_BIT: u64 = 0x10000000000000;

// ============================================================================
// Helper functions
// ============================================================================

fn float32_is_nan(x: IEEE754Float32) -> bool {
    (x.exponent == FLOAT32_EXPONENT_MAX) & (x.mantissa != 0)
}

fn float32_is_infinity(x: IEEE754Float32) -> bool {
    (x.exponent == FLOAT32_EXPONENT_MAX) & (x.mantissa == 0)
}

fn float32_is_zero(x: IEEE754Float32) -> bool {
    (x.exponent == 0) & (x.mantissa == 0)
}

fn float32_is_denormal(x: IEEE754Float32) -> bool {
    (x.exponent == 0) & (x.mantissa != 0)
}

fn float32_nan() -> IEEE754Float32 {
    IEEE754Float32 { sign: 0, exponent: FLOAT32_EXPONENT_MAX, mantissa: 0x400000 }
}

fn float32_infinity(sign: u1) -> IEEE754Float32 {
    IEEE754Float32 { sign, exponent: FLOAT32_EXPONENT_MAX, mantissa: 0 }
}

fn float32_zero(sign: u1) -> IEEE754Float32 {
    IEEE754Float32 { sign, exponent: 0, mantissa: 0 }
}

fn float64_is_nan(x: IEEE754Float64) -> bool {
    (x.exponent == FLOAT64_EXPONENT_MAX) & (x.mantissa != 0)
}

fn float64_is_infinity(x: IEEE754Float64) -> bool {
    (x.exponent == FLOAT64_EXPONENT_MAX) & (x.mantissa == 0)
}

fn float64_is_zero(x: IEEE754Float64) -> bool {
    (x.exponent == 0) & (x.mantissa == 0)
}

fn float64_is_denormal(x: IEEE754Float64) -> bool {
    (x.exponent == 0) & (x.mantissa != 0)
}

fn float64_nan() -> IEEE754Float64 {
    IEEE754Float64 { sign: 0, exponent: FLOAT64_EXPONENT_MAX, mantissa: 0x8000000000000 }
}

fn float64_infinity(sign: u1) -> IEEE754Float64 {
    IEEE754Float64 { sign, exponent: FLOAT64_EXPONENT_MAX, mantissa: 0 }
}

fn float64_zero(sign: u1) -> IEEE754Float64 {
    IEEE754Float64 { sign, exponent: 0, mantissa: 0 }
}

fn float32_from_bits(bits: u32) -> IEEE754Float32 {
    let sign = ((bits >> 31) & 1) as u1;
    let exponent = ((bits >> 23) & 0xFF) as u8;
    let mantissa = bits & 0x7FFFFF;
    IEEE754Float32 { sign, exponent, mantissa }
}

fn float32_to_bits(f: IEEE754Float32) -> u32 {
    ((f.sign as u32) << 31) | ((f.exponent as u32) << 23) | f.mantissa
}

fn float64_from_bits(bits: u64) -> IEEE754Float64 {
    let sign = ((bits >> 63) & 1) as u1;
    let exponent = ((bits >> 52) & 0x7FF) as u16;
    let mantissa = bits & 0xFFFFFFFFFFFFF;
    IEEE754Float64 { sign, exponent, mantissa }
}

fn float64_to_bits(f: IEEE754Float64) -> u64 {
    ((f.sign as u64) << 63) | ((f.exponent as u64) << 52) | f.mantissa
}

fn shift_right_sticky_u64(value: u64, shift: u64) -> u64 {
    if shift == 0 {
        value
    } else if shift >= 64 {
        if value != 0 { 1 } else { 0 }
    } else {
        let mask = (1 << shift) - 1;
        let shifted_out = value & mask;
        let result = value >> shift;
        if shifted_out != 0 { result | 1 } else { result }
    }
}

// ============================================================================
// Float32 Addition (for benchmarking)
// ============================================================================

fn add_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    let a_is_nan = float32_is_nan(a);
    let b_is_nan = float32_is_nan(b);
    let a_is_inf = float32_is_infinity(a);
    let b_is_inf = float32_is_infinity(b);
    let a_is_zero = float32_is_zero(a);
    let b_is_zero = float32_is_zero(b);
    let a_is_denormal = float32_is_denormal(a);
    let b_is_denormal = float32_is_denormal(b);

    let mant_a_raw: u64 = if a_is_denormal { a.mantissa as u64 } else { (a.mantissa | FLOAT32_IMPLICIT_BIT) as u64 };
    let mant_b_raw: u64 = if b_is_denormal { b.mantissa as u64 } else { (b.mantissa | FLOAT32_IMPLICIT_BIT) as u64 };

    let exp_a_eff: u64 = if a_is_denormal | a_is_zero { 1 } else { a.exponent as u64 };
    let exp_b_eff: u64 = if b_is_denormal | b_is_zero { 1 } else { b.exponent as u64 };

    let mut mant_a: u64 = mant_a_raw << 3;
    let mut mant_b: u64 = mant_b_raw << 3;
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

    let mut result_mant: u64 = 0;
    let mut result_sign: u1 = 0;
    let same_sign = a.sign == b.sign;

    if same_sign {
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

    let overflow_threshold: u64 = (FLOAT32_IMPLICIT_BIT as u64) << 4;
    if result_mant >= overflow_threshold {
        result_mant = shift_right_sticky_u64(result_mant, 1);
        result_exp = result_exp + 1;
    }

    let normal_position: u64 = (FLOAT32_IMPLICIT_BIT as u64) << 3;

    for _ in 0..26 {
        if (result_mant != 0) & (result_mant < normal_position) & (result_exp > 1) {
            result_mant = result_mant << 1;
            result_exp = result_exp - 1;
        }
    }

    let mut is_denormal_result = false;
    if (result_exp == 0) | ((result_exp == 1) & (result_mant < normal_position)) {
        result_exp = 0;
        is_denormal_result = true;
    }

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

    let overflows_to_inf = result_exp >= (FLOAT32_EXPONENT_MAX as u64);

    let final_mantissa = if is_denormal_result & (result_exp == 0) {
        (result_mant & 0x7FFFFF) as u32
    } else {
        (result_mant & ((FLOAT32_IMPLICIT_BIT - 1) as u64)) as u32
    };

    let normal_result = IEEE754Float32 { sign: result_sign, exponent: result_exp as u8, mantissa: final_mantissa };

    let nan_result = float32_nan();
    let inf_opposite_signs_result = float32_nan();

    let mut result = normal_result;

    if result_mant == 0 { result = float32_zero(0); }
    if overflows_to_inf { result = float32_infinity(result_sign); }

    if a_is_zero & b_is_zero { result = float32_zero(a.sign & b.sign); }
    else if a_is_zero { result = b; }
    else if b_is_zero { result = a; }

    if a_is_inf & !b_is_inf { result = a; }
    else if b_is_inf & !a_is_inf { result = b; }
    else if a_is_inf & b_is_inf {
        if a.sign == b.sign { result = a; } else { result = inf_opposite_signs_result; }
    }

    if a_is_nan | b_is_nan { result = nan_result; }

    result
}

// ============================================================================
// Float32 Multiplication (for benchmarking)
// ============================================================================

fn mul_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    let a_is_nan = float32_is_nan(a);
    let b_is_nan = float32_is_nan(b);
    let a_is_inf = float32_is_infinity(a);
    let b_is_inf = float32_is_infinity(b);
    let a_is_zero = float32_is_zero(a);
    let b_is_zero = float32_is_zero(b);
    let a_is_denormal = float32_is_denormal(a);
    let b_is_denormal = float32_is_denormal(b);

    let result_sign: u1 = a.sign ^ b.sign;

    let mant_a: u64 = if a_is_denormal { a.mantissa as u64 } else if a_is_zero { 0 } else { (a.mantissa | FLOAT32_IMPLICIT_BIT) as u64 };
    let mant_b: u64 = if b_is_denormal { b.mantissa as u64 } else if b_is_zero { 0 } else { (b.mantissa | FLOAT32_IMPLICIT_BIT) as u64 };

    let exp_a_eff: i64 = if a_is_denormal { 1 - 127 } else if a_is_zero { 0 } else { (a.exponent as i64) - 127 };
    let exp_b_eff: i64 = if b_is_denormal { 1 - 127 } else if b_is_zero { 0 } else { (b.exponent as i64) - 127 };

    let product: u64 = mant_a * mant_b;
    let result_is_zero = product == 0;

    let mut result_exp: i64 = exp_a_eff + exp_b_eff + 127;
    let mut result_mant: u64 = product;

    let bit_47_set = (product >> 47) & 1 == 1;
    if bit_47_set {
        result_exp = result_exp + 1;
        result_mant = shift_right_sticky_u64(product, 1);
    }

    let guard_shift: u64 = 46 - 26;
    result_mant = shift_right_sticky_u64(result_mant, guard_shift);

    let guard_bits = result_mant & 0x7;
    result_mant = result_mant >> 3;

    let should_round_up = (guard_bits > 4) | ((guard_bits == 4) & ((result_mant & 1) == 1));
    if should_round_up {
        result_mant = result_mant + 1;
        if result_mant >= ((FLOAT32_IMPLICIT_BIT as u64) << 1) {
            result_mant = result_mant >> 1;
            result_exp = result_exp + 1;
        }
    }

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

    let overflows_to_inf = result_exp >= (FLOAT32_EXPONENT_MAX as i64);

    let final_mantissa = if is_denormal_result { (result_mant & 0x7FFFFF) as u32 } else { (result_mant & ((FLOAT32_IMPLICIT_BIT - 1) as u64)) as u32 };

    let normal_result = IEEE754Float32 { sign: result_sign, exponent: result_exp as u8, mantissa: final_mantissa };

    let mut result = normal_result;

    if result_is_zero | (result_mant == 0) { result = float32_zero(result_sign); }
    if overflows_to_inf { result = float32_infinity(result_sign); }

    if a_is_zero | b_is_zero {
        if a_is_inf | b_is_inf { result = float32_nan(); } else { result = float32_zero(result_sign); }
    }

    if (a_is_inf | b_is_inf) & !a_is_zero & !b_is_zero {
        if a_is_nan | b_is_nan { result = float32_nan(); } else { result = float32_infinity(result_sign); }
    }

    if a_is_nan | b_is_nan { result = float32_nan(); }

    result
}

// ============================================================================
// Main function for circuit compilation and benchmarking
// ============================================================================

fn main(a32: pub u32, b32: pub u32) -> pub (u32, u32) {
    let fa32 = float32_from_bits(a32);
    let fb32 = float32_from_bits(b32);
    
    let add32_result = float32_to_bits(add_float32(fa32, fb32));
    let mul32_result = float32_to_bits(mul_float32(fa32, fb32));
    
    (add32_result, mul32_result)
}
