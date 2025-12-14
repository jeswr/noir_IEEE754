// ============================================================================
// IEEE 754 Float64 Operations Module
// ============================================================================

pub(crate) mod helpers;
pub(crate) mod add;
pub(crate) mod mul;
pub(crate) mod div;
pub(crate) mod cmp;
pub(crate) mod sqrt;

// Re-export all public functions
pub use helpers::{
    float64_from_bits, float64_infinity, float64_is_denormal, float64_is_infinity, float64_is_nan,
    float64_is_zero, float64_nan, float64_to_bits, float64_zero,
};

pub use add::add_float64;
pub use div::div_float64;
pub use mul::mul_float64;
pub use cmp::{float64_eq, float64_ne, float64_lt, float64_le, float64_gt, float64_ge, float64_unordered, float64_compare};
pub use sqrt::sqrt_float64;

// Subtraction is implemented as a + (-b)
use crate::types::IEEE754Float64;

// IEEE 754 compliant subtraction for 64-bit floats
// Implemented as a + (-b)
pub fn sub_float64(a: IEEE754Float64, b: IEEE754Float64) -> IEEE754Float64 {
    // Negate b's sign and add
    let neg_b = IEEE754Float64 { sign: 1 - b.sign, exponent: b.exponent, mantissa: b.mantissa };
    add_float64(a, neg_b)
}
