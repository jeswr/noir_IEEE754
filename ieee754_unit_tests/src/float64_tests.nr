// ============================================================================
// IEEE 754 Float64 Unit Tests
// ============================================================================

use ieee754::float::{
    add_float64, div_float64, float64_from_bits, float64_infinity, float64_is_infinity,
    float64_is_nan, float64_is_zero, float64_nan, float64_to_bits, float64_zero, mul_float64,
    sub_float64,
    // Comparison operations
    float64_eq, float64_ne, float64_lt, float64_le, float64_gt, float64_ge,
    float64_unordered, float64_compare,
};

// ============================================================================
// Float64 Conversion Tests
// ============================================================================

#[test]
fn test_float64_from_to_bits() {
    // Test 1.0 = 0x3FF0000000000000
    let one_bits: u64 = 0x3FF0000000000000;
    let one = float64_from_bits(one_bits);
    assert(one.sign == 0);
    assert(one.exponent == 1023);
    assert(one.mantissa == 0);
    assert(float64_to_bits(one) == one_bits);

    // Test -1.0 = 0xBFF0000000000000
    let neg_one_bits: u64 = 0xBFF0000000000000;
    let neg_one = float64_from_bits(neg_one_bits);
    assert(neg_one.sign == 1);
    assert(neg_one.exponent == 1023);
    assert(neg_one.mantissa == 0);
    assert(float64_to_bits(neg_one) == neg_one_bits);
}

#[test]
fn test_float64_special_values() {
    // Test NaN
    let nan = float64_nan();
    assert(float64_is_nan(nan));
    assert(!float64_is_infinity(nan));
    assert(!float64_is_zero(nan));

    // Test Infinity
    let inf = float64_infinity(0);
    assert(!float64_is_nan(inf));
    assert(float64_is_infinity(inf));
    assert(!float64_is_zero(inf));

    // Test Zero
    let zero = float64_zero(0);
    assert(!float64_is_nan(zero));
    assert(!float64_is_infinity(zero));
    assert(float64_is_zero(zero));
}

// ============================================================================
// Float64 Addition Tests
// ============================================================================

#[test]
fn test_float64_add_basic() {
    // 1.0 + 1.0 = 2.0
    let one = float64_from_bits(0x3FF0000000000000);
    let result = add_float64(one, one);
    let expected = float64_from_bits(0x4000000000000000); // 2.0
    assert(float64_to_bits(result) == float64_to_bits(expected));
}

#[test]
fn test_float64_add_different_exponents() {
    // 1.0 + 0.5 = 1.5
    let one = float64_from_bits(0x3FF0000000000000); // 1.0
    let half = float64_from_bits(0x3FE0000000000000); // 0.5
    let result = add_float64(one, half);
    let expected = float64_from_bits(0x3FF8000000000000); // 1.5
    assert(float64_to_bits(result) == float64_to_bits(expected));
}

#[test]
fn test_float64_add_with_zero() {
    let one = float64_from_bits(0x3FF0000000000000);
    let zero = float64_zero(0);

    // 1.0 + 0.0 = 1.0
    let result = add_float64(one, zero);
    assert(float64_to_bits(result) == 0x3FF0000000000000);

    // 0.0 + 1.0 = 1.0
    let result2 = add_float64(zero, one);
    assert(float64_to_bits(result2) == 0x3FF0000000000000);
}

#[test]
fn test_float64_add_opposite_signs() {
    let one = float64_from_bits(0x3FF0000000000000); // 1.0
    let neg_one = float64_from_bits(0xBFF0000000000000); // -1.0

    // 1.0 + (-1.0) = 0.0
    let result = add_float64(one, neg_one);
    assert(float64_is_zero(result));
}

#[test]
fn test_float64_add_nan() {
    let nan = float64_nan();
    let one = float64_from_bits(0x3FF0000000000000);

    // NaN + anything = NaN
    let result = add_float64(nan, one);
    assert(float64_is_nan(result));

    // anything + NaN = NaN
    let result2 = add_float64(one, nan);
    assert(float64_is_nan(result2));
}

#[test]
fn test_float64_add_infinity() {
    let inf = float64_infinity(0);
    let neg_inf = float64_infinity(1);
    let one = float64_from_bits(0x3FF0000000000000);

    // Inf + 1.0 = Inf
    let result = add_float64(inf, one);
    assert(float64_is_infinity(result));
    assert(result.sign == 0);

    // Inf + Inf = Inf
    let result2 = add_float64(inf, inf);
    assert(float64_is_infinity(result2));

    // Inf + (-Inf) = NaN
    let result3 = add_float64(inf, neg_inf);
    assert(float64_is_nan(result3));
}

// ============================================================================
// Float64 Subtraction Tests
// ============================================================================

#[test]
fn test_float64_sub_basic() {
    // 3.0 - 2.0 = 1.0
    let three = float64_from_bits(0x4008000000000000); // 3.0
    let two = float64_from_bits(0x4000000000000000); // 2.0
    let result = sub_float64(three, two);
    assert(float64_to_bits(result) == 0x3FF0000000000000); // 1.0
}

// ============================================================================
// Float64 Multiplication Tests
// ============================================================================

#[test]
fn test_float64_mul_basic() {
    // 2.0 * 3.0 = 6.0
    let two = float64_from_bits(0x4000000000000000); // 2.0
    let three = float64_from_bits(0x4008000000000000); // 3.0
    let result = mul_float64(two, three);
    assert(float64_to_bits(result) == 0x4018000000000000); // 6.0
}

// ============================================================================
// Float64 Division Tests
// ============================================================================

#[test]
fn test_float64_div_basic() {
    // 6.0 / 2.0 = 3.0
    let six = float64_from_bits(0x4018000000000000); // 6.0
    let two = float64_from_bits(0x4000000000000000); // 2.0
    let result = div_float64(six, two);
    assert(float64_to_bits(result) == 0x4008000000000000); // 3.0
}

// ============================================================================
// Float64 Comparison Tests
// ============================================================================

#[test]
fn test_float64_eq_basic() {
    let one = float64_from_bits(0x3FF0000000000000); // 1.0
    let two = float64_from_bits(0x4000000000000000); // 2.0
    let one_copy = float64_from_bits(0x3FF0000000000000); // 1.0

    // 1.0 == 1.0
    assert(float64_eq(one, one_copy));

    // 1.0 != 2.0
    assert(!float64_eq(one, two));
}

#[test]
fn test_float64_eq_zeros() {
    let pos_zero = float64_zero(0);
    let neg_zero = float64_zero(1);

    // +0 == -0
    assert(float64_eq(pos_zero, neg_zero));
    assert(float64_eq(neg_zero, pos_zero));
}

#[test]
fn test_float64_eq_nan() {
    let nan = float64_nan();
    let one = float64_from_bits(0x3FF0000000000000);

    // NaN != NaN
    assert(!float64_eq(nan, nan));

    // NaN != 1.0
    assert(!float64_eq(nan, one));
    assert(!float64_eq(one, nan));
}

#[test]
fn test_float64_ne_basic() {
    let one = float64_from_bits(0x3FF0000000000000);
    let two = float64_from_bits(0x4000000000000000);

    assert(float64_ne(one, two));
    assert(!float64_ne(one, one));
}

#[test]
fn test_float64_lt_basic() {
    let one = float64_from_bits(0x3FF0000000000000); // 1.0
    let two = float64_from_bits(0x4000000000000000); // 2.0
    let neg_one = float64_from_bits(0xBFF0000000000000); // -1.0

    // 1.0 < 2.0
    assert(float64_lt(one, two));

    // 2.0 is not < 1.0
    assert(!float64_lt(two, one));

    // 1.0 is not < 1.0
    assert(!float64_lt(one, one));

    // -1.0 < 1.0
    assert(float64_lt(neg_one, one));

    // 1.0 is not < -1.0
    assert(!float64_lt(one, neg_one));
}

#[test]
fn test_float64_lt_zeros() {
    let pos_zero = float64_zero(0);
    let neg_zero = float64_zero(1);

    // +0 is not < -0 (they are equal)
    assert(!float64_lt(pos_zero, neg_zero));
    assert(!float64_lt(neg_zero, pos_zero));
}

#[test]
fn test_float64_lt_infinity() {
    let inf = float64_infinity(0);
    let neg_inf = float64_infinity(1);
    let one = float64_from_bits(0x3FF0000000000000);

    // 1.0 < Inf
    assert(float64_lt(one, inf));

    // -Inf < 1.0
    assert(float64_lt(neg_inf, one));

    // -Inf < Inf
    assert(float64_lt(neg_inf, inf));

    // Inf is not < 1.0
    assert(!float64_lt(inf, one));
}

#[test]
fn test_float64_lt_nan() {
    let nan = float64_nan();
    let one = float64_from_bits(0x3FF0000000000000);

    // NaN comparisons are always false
    assert(!float64_lt(nan, one));
    assert(!float64_lt(one, nan));
    assert(!float64_lt(nan, nan));
}

#[test]
fn test_float64_le_basic() {
    let one = float64_from_bits(0x3FF0000000000000);
    let two = float64_from_bits(0x4000000000000000);

    // 1.0 <= 2.0
    assert(float64_le(one, two));

    // 1.0 <= 1.0
    assert(float64_le(one, one));

    // 2.0 is not <= 1.0
    assert(!float64_le(two, one));
}

#[test]
fn test_float64_gt_basic() {
    let one = float64_from_bits(0x3FF0000000000000);
    let two = float64_from_bits(0x4000000000000000);

    // 2.0 > 1.0
    assert(float64_gt(two, one));

    // 1.0 is not > 2.0
    assert(!float64_gt(one, two));

    // 1.0 is not > 1.0
    assert(!float64_gt(one, one));
}

#[test]
fn test_float64_ge_basic() {
    let one = float64_from_bits(0x3FF0000000000000);
    let two = float64_from_bits(0x4000000000000000);

    // 2.0 >= 1.0
    assert(float64_ge(two, one));

    // 1.0 >= 1.0
    assert(float64_ge(one, one));

    // 1.0 is not >= 2.0
    assert(!float64_ge(one, two));
}

#[test]
fn test_float64_unordered() {
    let nan = float64_nan();
    let one = float64_from_bits(0x3FF0000000000000);

    // NaN makes comparison unordered
    assert(float64_unordered(nan, one));
    assert(float64_unordered(one, nan));
    assert(float64_unordered(nan, nan));

    // No NaN means ordered
    assert(!float64_unordered(one, one));
}

#[test]
fn test_float64_compare_basic() {
    let one = float64_from_bits(0x3FF0000000000000);
    let two = float64_from_bits(0x4000000000000000);

    // 1.0 < 2.0 -> -1
    assert(float64_compare(one, two) == -1);

    // 2.0 > 1.0 -> 1
    assert(float64_compare(two, one) == 1);

    // 1.0 == 1.0 -> 0
    assert(float64_compare(one, one) == 0);
}

#[test]
fn test_float64_compare_nan() {
    let nan = float64_nan();
    let neg_nan = ieee754::float::IEEE754Float64 { sign: 1, exponent: 2047, mantissa: 0x8000000000000 };
    let one = float64_from_bits(0x3FF0000000000000);

    // +NaN > everything
    assert(float64_compare(nan, one) == 1);

    // -NaN < everything
    assert(float64_compare(neg_nan, one) == -1);

    // everything < +NaN
    assert(float64_compare(one, nan) == -1);

    // everything > -NaN
    assert(float64_compare(one, neg_nan) == 1);
}
