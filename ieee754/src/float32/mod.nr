// ============================================================================
// IEEE 754 Float32 Operations Module
// ============================================================================

pub(crate) mod helpers;
pub(crate) mod add;
pub(crate) mod mul;
pub(crate) mod div;
pub(crate) mod cmp;
pub(crate) mod sqrt;
pub mod packed;

// Re-export all public functions
pub use helpers::{
    float32_from_bits, float32_infinity, float32_is_denormal, float32_is_infinity, float32_is_nan,
    float32_is_zero, float32_nan, float32_to_bits, float32_zero,
};

pub use add::add_float32;
pub use cmp::{
    float32_compare, float32_eq, float32_ge, float32_gt, float32_le, float32_lt, float32_ne,
    float32_unordered,
};
pub use div::div_float32;
pub use mul::mul_float32;
pub use sqrt::sqrt_float32;

// Re-export packed representation functions
pub use packed::{
    abs_float32_bits, add_float32_bits, div_float32_bits, float32_compare_bits, float32_eq_bits,
    float32_ge_bits, float32_gt_bits, float32_le_bits, float32_lt_bits, float32_ne_bits,
    float32_unordered_bits, mul_float32_bits, sqrt_float32_bits, sub_float32_bits,
};

// Subtraction is implemented as a + (-b)
use crate::types::IEEE754Float32;

// IEEE 754 compliant subtraction for 32-bit floats
// Implemented as a + (-b)
pub fn sub_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    // Negate b's sign and add
    let neg_b = IEEE754Float32 { sign: 1 - b.sign, exponent: b.exponent, mantissa: b.mantissa };
    add_float32(a, neg_b)
}

// IEEE 754 compliant absolute value for 32-bit floats
// Returns the magnitude of the float (sign bit set to 0)
// Note: abs(NaN) returns NaN (with sign bit cleared)
pub fn abs_float32(a: IEEE754Float32) -> IEEE754Float32 {
    IEEE754Float32 { sign: 0, exponent: a.exponent, mantissa: a.mantissa }
}
