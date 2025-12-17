// ============================================================================
// IEEE 754 Float64 Packed Representation Operations
// ============================================================================
//
// This module provides operations on packed u64 representations of IEEE754
// doubles, avoiding the overhead of struct construction/destruction.
// All functions maintain full IEEE754 compliance.
// ============================================================================

use crate::float64::add::add_float64;
use crate::float64::cmp::{
    float64_compare, float64_eq, float64_ge, float64_gt, float64_le, float64_lt, float64_ne,
    float64_unordered,
};
use crate::float64::div::div_float64;
use crate::float64::helpers::{float64_from_bits, float64_to_bits};
use crate::float64::mul::mul_float64;
use crate::float64::sqrt::sqrt_float64;
use crate::types::IEEE754Float64;

// ============================================================================
// Arithmetic Operations on Packed Representation
// ============================================================================

/// IEEE 754 addition on packed u64 representation
/// # Arguments
/// * `a` - First operand as u64 bit pattern
/// * `b` - Second operand as u64 bit pattern
/// # Returns
/// * Result as u64 bit pattern
pub fn add_float64_bits(a: u64, b: u64) -> u64 {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    let result = add_float64(a_float, b_float);
    float64_to_bits(result)
}

/// IEEE 754 subtraction on packed u64 representation
pub fn sub_float64_bits(a: u64, b: u64) -> u64 {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    // Implement subtraction as a + (-b)
    let neg_b = IEEE754Float64 {
        sign: 1 - b_float.sign,
        exponent: b_float.exponent,
        mantissa: b_float.mantissa,
    };
    let result = add_float64(a_float, neg_b);
    float64_to_bits(result)
}

/// IEEE 754 multiplication on packed u64 representation
pub fn mul_float64_bits(a: u64, b: u64) -> u64 {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    let result = mul_float64(a_float, b_float);
    float64_to_bits(result)
}

/// IEEE 754 division on packed u64 representation
pub fn div_float64_bits(a: u64, b: u64) -> u64 {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    let result = div_float64(a_float, b_float);
    float64_to_bits(result)
}

/// IEEE 754 square root on packed u64 representation
pub fn sqrt_float64_bits(a: u64) -> u64 {
    let a_float = float64_from_bits(a);
    let result = sqrt_float64(a_float);
    float64_to_bits(result)
}

/// IEEE 754 absolute value on packed u64 representation
pub fn abs_float64_bits(a: u64) -> u64 {
    let a_float = float64_from_bits(a);
    // Implement abs by clearing sign bit
    let result = IEEE754Float64 { sign: 0, exponent: a_float.exponent, mantissa: a_float.mantissa };
    float64_to_bits(result)
}

// ============================================================================
// Comparison Operations on Packed Representation
// ============================================================================

/// IEEE 754 equality comparison on packed u64 representation
pub fn float64_eq_bits(a: u64, b: u64) -> bool {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    float64_eq(a_float, b_float)
}

/// IEEE 754 not-equal comparison on packed u64 representation
pub fn float64_ne_bits(a: u64, b: u64) -> bool {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    float64_ne(a_float, b_float)
}

/// IEEE 754 less-than comparison on packed u64 representation
pub fn float64_lt_bits(a: u64, b: u64) -> bool {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    float64_lt(a_float, b_float)
}

/// IEEE 754 less-than-or-equal comparison on packed u64 representation
pub fn float64_le_bits(a: u64, b: u64) -> bool {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    float64_le(a_float, b_float)
}

/// IEEE 754 greater-than comparison on packed u64 representation
pub fn float64_gt_bits(a: u64, b: u64) -> bool {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    float64_gt(a_float, b_float)
}

/// IEEE 754 greater-than-or-equal comparison on packed u64 representation
pub fn float64_ge_bits(a: u64, b: u64) -> bool {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    float64_ge(a_float, b_float)
}

/// IEEE 754 unordered comparison (returns true if either operand is NaN)
pub fn float64_unordered_bits(a: u64, b: u64) -> bool {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    float64_unordered(a_float, b_float)
}

/// IEEE 754 total ordering comparison on packed u64 representation
/// Returns -1 if a < b, 0 if a == b, 1 if a > b
pub fn float64_compare_bits(a: u64, b: u64) -> i8 {
    let a_float = float64_from_bits(a);
    let b_float = float64_from_bits(b);
    float64_compare(a_float, b_float)
}
