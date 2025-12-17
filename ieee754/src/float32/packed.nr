// ============================================================================
// IEEE 754 Float32 Packed Representation Operations
// ============================================================================
//
// This module provides operations on packed u32 representations of IEEE754
// floats, avoiding the overhead of struct construction/destruction.
// All functions maintain full IEEE754 compliance.
// ============================================================================

use crate::float32::add::add_float32;
use crate::float32::cmp::{
    float32_compare, float32_eq, float32_ge, float32_gt, float32_le, float32_lt, float32_ne,
    float32_unordered,
};
use crate::float32::div::div_float32;
use crate::float32::helpers::{float32_from_bits, float32_to_bits};
use crate::float32::mul::mul_float32;
use crate::float32::sqrt::sqrt_float32;
use crate::types::IEEE754Float32;

// ============================================================================
// Arithmetic Operations on Packed Representation
// ============================================================================

/// IEEE 754 addition on packed u32 representation
/// # Arguments
/// * `a` - First operand as u32 bit pattern
/// * `b` - Second operand as u32 bit pattern
/// # Returns
/// * Result as u32 bit pattern
pub fn add_float32_bits(a: u32, b: u32) -> u32 {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    let result = add_float32(a_float, b_float);
    float32_to_bits(result)
}

/// IEEE 754 subtraction on packed u32 representation
pub fn sub_float32_bits(a: u32, b: u32) -> u32 {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    // Implement subtraction as a + (-b)
    let neg_b = IEEE754Float32 {
        sign: 1 - b_float.sign,
        exponent: b_float.exponent,
        mantissa: b_float.mantissa,
    };
    let result = add_float32(a_float, neg_b);
    float32_to_bits(result)
}

/// IEEE 754 multiplication on packed u32 representation
pub fn mul_float32_bits(a: u32, b: u32) -> u32 {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    let result = mul_float32(a_float, b_float);
    float32_to_bits(result)
}

/// IEEE 754 division on packed u32 representation
pub fn div_float32_bits(a: u32, b: u32) -> u32 {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    let result = div_float32(a_float, b_float);
    float32_to_bits(result)
}

/// IEEE 754 square root on packed u32 representation
pub fn sqrt_float32_bits(a: u32) -> u32 {
    let a_float = float32_from_bits(a);
    let result = sqrt_float32(a_float);
    float32_to_bits(result)
}

/// IEEE 754 absolute value on packed u32 representation
pub fn abs_float32_bits(a: u32) -> u32 {
    let a_float = float32_from_bits(a);
    // Implement abs by clearing sign bit
    let result = IEEE754Float32 { sign: 0, exponent: a_float.exponent, mantissa: a_float.mantissa };
    float32_to_bits(result)
}

// ============================================================================
// Comparison Operations on Packed Representation
// ============================================================================

/// IEEE 754 equality comparison on packed u32 representation
pub fn float32_eq_bits(a: u32, b: u32) -> bool {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    float32_eq(a_float, b_float)
}

/// IEEE 754 not-equal comparison on packed u32 representation
pub fn float32_ne_bits(a: u32, b: u32) -> bool {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    float32_ne(a_float, b_float)
}

/// IEEE 754 less-than comparison on packed u32 representation
pub fn float32_lt_bits(a: u32, b: u32) -> bool {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    float32_lt(a_float, b_float)
}

/// IEEE 754 less-than-or-equal comparison on packed u32 representation
pub fn float32_le_bits(a: u32, b: u32) -> bool {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    float32_le(a_float, b_float)
}

/// IEEE 754 greater-than comparison on packed u32 representation
pub fn float32_gt_bits(a: u32, b: u32) -> bool {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    float32_gt(a_float, b_float)
}

/// IEEE 754 greater-than-or-equal comparison on packed u32 representation
pub fn float32_ge_bits(a: u32, b: u32) -> bool {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    float32_ge(a_float, b_float)
}

/// IEEE 754 unordered comparison (returns true if either operand is NaN)
pub fn float32_unordered_bits(a: u32, b: u32) -> bool {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    float32_unordered(a_float, b_float)
}

/// IEEE 754 total ordering comparison on packed u32 representation
/// Returns -1 if a < b, 0 if a == b, 1 if a > b
pub fn float32_compare_bits(a: u32, b: u32) -> i8 {
    let a_float = float32_from_bits(a);
    let b_float = float32_from_bits(b);
    float32_compare(a_float, b_float)
}
