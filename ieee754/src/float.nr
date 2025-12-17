// ============================================================================
// IEEE 754 Compliant Floating Point Library
// ============================================================================
//
// This module re-exports all IEEE 754 floating point types and operations
// from their respective submodules for backwards compatibility.
//
// Module structure:
// - types.nr: Core types (IEEE754Float32, IEEE754Float64) and constants
// - utils.nr: Shared utility functions (shift operations, U128 arithmetic)
// - float32/: Float32 operations (helpers, add, mul, div, packed)
// - float64/: Float64 operations (helpers, add, mul, div, packed)
// ============================================================================

// Re-export types and constants
pub use crate::types::{
    FLOAT32_EXPONENT_MAX, FLOAT32_IMPLICIT_BIT, FLOAT64_EXPONENT_MAX, FLOAT64_IMPLICIT_BIT,
    IEEE754Float32, IEEE754Float64,
    // Float32 special value constants
    FLOAT32_ZERO, FLOAT32_NEG_ZERO, FLOAT32_ONE, FLOAT32_NEG_ONE, FLOAT32_INFINITY,
    FLOAT32_NEG_INFINITY, FLOAT32_NAN, FLOAT32_SIGNALING_NAN,
    // Float64 special value constants
    FLOAT64_ZERO, FLOAT64_NEG_ZERO, FLOAT64_ONE, FLOAT64_NEG_ONE, FLOAT64_INFINITY,
    FLOAT64_NEG_INFINITY, FLOAT64_NAN, FLOAT64_SIGNALING_NAN,
};

// Re-export Float32 operations
pub use crate::float32::{
    add_float32, div_float32, float32_from_bits, float32_infinity, float32_is_denormal,
    float32_is_infinity, float32_is_nan, float32_is_zero, float32_nan, float32_to_bits,
    float32_zero, mul_float32, sub_float32, abs_float32,
    // Comparison operations
    float32_eq, float32_ne, float32_lt, float32_le, float32_gt, float32_ge, float32_unordered,
    float32_compare,
    // Square root
    sqrt_float32,
    // Packed representation operations (on u32 bit patterns)
    add_float32_bits, div_float32_bits, mul_float32_bits, sub_float32_bits, sqrt_float32_bits,
    abs_float32_bits, float32_compare_bits, float32_eq_bits, float32_ge_bits, float32_gt_bits,
    float32_le_bits, float32_lt_bits, float32_ne_bits, float32_unordered_bits,
};

// Re-export Float64 operations
pub use crate::float64::{
    add_float64, div_float64, float64_from_bits, float64_infinity, float64_is_denormal,
    float64_is_infinity, float64_is_nan, float64_is_zero, float64_nan, float64_to_bits,
    float64_zero, mul_float64, sub_float64, abs_float64,
    // Comparison operations
    float64_eq, float64_ne, float64_lt, float64_le, float64_gt, float64_ge, float64_unordered,
    float64_compare,
    // Square root
    sqrt_float64,
    // Packed representation operations (on u64 bit patterns)
    add_float64_bits, div_float64_bits, mul_float64_bits, sub_float64_bits, sqrt_float64_bits,
    abs_float64_bits, float64_compare_bits, float64_eq_bits, float64_ge_bits, float64_gt_bits,
    float64_le_bits, float64_lt_bits, float64_ne_bits, float64_unordered_bits,
};
