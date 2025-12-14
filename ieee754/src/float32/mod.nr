// ============================================================================
// IEEE 754 Float32 Operations Module
// ============================================================================

pub(crate) mod helpers;
pub(crate) mod add;
pub(crate) mod mul;
pub(crate) mod div;
pub(crate) mod cmp;
pub(crate) mod sqrt;

// Re-export all public functions
pub use helpers::{
    float32_from_bits, float32_infinity, float32_is_denormal, float32_is_infinity, float32_is_nan,
    float32_is_zero, float32_nan, float32_to_bits, float32_zero,
};

pub use add::add_float32;
pub use div::div_float32;
pub use mul::mul_float32;
pub use cmp::{float32_eq, float32_ne, float32_lt, float32_le, float32_gt, float32_ge, float32_unordered, float32_compare};
pub use sqrt::sqrt_float32;

// Subtraction is implemented as a + (-b)
use crate::types::IEEE754Float32;

// IEEE 754 compliant subtraction for 32-bit floats
// Implemented as a + (-b)
pub fn sub_float32(a: IEEE754Float32, b: IEEE754Float32) -> IEEE754Float32 {
    // Negate b's sign and add
    let neg_b = IEEE754Float32 { sign: 1 - b.sign, exponent: b.exponent, mantissa: b.mantissa };
    add_float32(a, neg_b)
}
