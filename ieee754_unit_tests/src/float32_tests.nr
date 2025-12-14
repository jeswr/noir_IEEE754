// ============================================================================
// IEEE 754 Float32 Unit Tests
// ============================================================================

use ieee754::float::{
    add_float32, div_float32, float32_from_bits, float32_infinity, float32_is_infinity,
    float32_is_nan, float32_is_zero, float32_nan, float32_to_bits, float32_zero, mul_float32,
    sub_float32, abs_float32,
    // Comparison operations
    float32_eq, float32_ne, float32_lt, float32_le, float32_gt, float32_ge, float32_unordered,
    float32_compare,
    // Square root
    sqrt_float32,
};

// ============================================================================
// Float32 Addition Tests
// ============================================================================

#[test]
fn test_float32_add_basic() {
    // 1.0 + 1.0 = 2.0
    let one = float32_from_bits(0x3F800000);
    let result = add_float32(one, one);
    let expected = float32_from_bits(0x40000000); // 2.0
    assert(float32_to_bits(result) == float32_to_bits(expected));
}

#[test]
fn test_float32_add_different_exponents() {
    // 1.0 + 0.5 = 1.5
    let one = float32_from_bits(0x3F800000); // 1.0
    let half = float32_from_bits(0x3F000000); // 0.5
    let result = add_float32(one, half);
    let expected = float32_from_bits(0x3FC00000); // 1.5
    assert(float32_to_bits(result) == float32_to_bits(expected));
}

#[test]
fn test_float32_add_with_zero() {
    let one = float32_from_bits(0x3F800000);
    let zero = float32_zero(0);

    // 1.0 + 0.0 = 1.0
    let result = add_float32(one, zero);
    assert(float32_to_bits(result) == 0x3F800000);

    // 0.0 + 1.0 = 1.0
    let result2 = add_float32(zero, one);
    assert(float32_to_bits(result2) == 0x3F800000);
}

#[test]
fn test_float32_add_opposite_signs() {
    let one = float32_from_bits(0x3F800000); // 1.0
    let neg_one = float32_from_bits(0xBF800000); // -1.0

    // 1.0 + (-1.0) = 0.0
    let result = add_float32(one, neg_one);
    assert(float32_is_zero(result));
}

#[test]
fn test_float32_add_nan() {
    let nan = float32_nan();
    let one = float32_from_bits(0x3F800000);

    // NaN + anything = NaN
    let result = add_float32(nan, one);
    assert(float32_is_nan(result));

    // anything + NaN = NaN
    let result2 = add_float32(one, nan);
    assert(float32_is_nan(result2));
}

#[test]
fn test_float32_add_infinity() {
    let inf = float32_infinity(0);
    let neg_inf = float32_infinity(1);
    let one = float32_from_bits(0x3F800000);

    // Inf + 1.0 = Inf
    let result = add_float32(inf, one);
    assert(float32_is_infinity(result));
    assert(result.sign == 0);

    // Inf + Inf = Inf
    let result2 = add_float32(inf, inf);
    assert(float32_is_infinity(result2));

    // Inf + (-Inf) = NaN
    let result3 = add_float32(inf, neg_inf);
    assert(float32_is_nan(result3));
}

// ============================================================================
// Float32 Subtraction Tests
// ============================================================================

#[test]
fn test_float32_sub_basic() {
    // 3.0 - 2.0 = 1.0
    let three = float32_from_bits(0x40400000); // 3.0
    let two = float32_from_bits(0x40000000); // 2.0
    let result = sub_float32(three, two);
    assert(float32_to_bits(result) == 0x3F800000); // 1.0
}

#[test]
fn test_float32_sub_negative_result() {
    // 2.0 - 3.0 = -1.0
    let two = float32_from_bits(0x40000000); // 2.0
    let three = float32_from_bits(0x40400000); // 3.0
    let result = sub_float32(two, three);
    assert(float32_to_bits(result) == 0xBF800000); // -1.0
}

// ============================================================================
// Float32 Multiplication Tests
// ============================================================================

#[test]
fn test_float32_mul_basic() {
    // 2.0 * 3.0 = 6.0
    let two = float32_from_bits(0x40000000); // 2.0
    let three = float32_from_bits(0x40400000); // 3.0
    let result = mul_float32(two, three);
    assert(float32_to_bits(result) == 0x40C00000); // 6.0
}

#[test]
fn test_float32_mul_with_one() {
    // 5.0 * 1.0 = 5.0
    let five = float32_from_bits(0x40A00000); // 5.0
    let one = float32_from_bits(0x3F800000); // 1.0
    let result = mul_float32(five, one);
    assert(float32_to_bits(result) == 0x40A00000); // 5.0
}

#[test]
fn test_float32_mul_with_zero() {
    // 5.0 * 0.0 = 0.0
    let five = float32_from_bits(0x40A00000); // 5.0
    let zero = float32_zero(0);
    let result = mul_float32(five, zero);
    assert(float32_is_zero(result));
}

#[test]
fn test_float32_mul_negative() {
    // 2.0 * (-3.0) = -6.0
    let two = float32_from_bits(0x40000000); // 2.0
    let neg_three = float32_from_bits(0xC0400000); // -3.0
    let result = mul_float32(two, neg_three);
    assert(float32_to_bits(result) == 0xC0C00000); // -6.0

    // (-2.0) * (-3.0) = 6.0
    let neg_two = float32_from_bits(0xC0000000); // -2.0
    let result2 = mul_float32(neg_two, neg_three);
    assert(float32_to_bits(result2) == 0x40C00000); // 6.0
}

#[test]
fn test_float32_mul_infinity() {
    let inf = float32_infinity(0);
    let two = float32_from_bits(0x40000000); // 2.0
    let zero = float32_zero(0);

    // Inf * 2.0 = Inf
    let result = mul_float32(inf, two);
    assert(float32_is_infinity(result));

    // Inf * 0.0 = NaN
    let result2 = mul_float32(inf, zero);
    assert(float32_is_nan(result2));
}

// ============================================================================
// Float32 Division Tests
// ============================================================================

#[test]
fn test_float32_div_basic() {
    // 6.0 / 2.0 = 3.0
    let six = float32_from_bits(0x40C00000); // 6.0
    let two = float32_from_bits(0x40000000); // 2.0
    let result = div_float32(six, two);
    assert(float32_to_bits(result) == 0x40400000); // 3.0
}

#[test]
fn test_float32_div_with_one() {
    // 5.0 / 1.0 = 5.0
    let five = float32_from_bits(0x40A00000); // 5.0
    let one = float32_from_bits(0x3F800000); // 1.0
    let result = div_float32(five, one);
    assert(float32_to_bits(result) == 0x40A00000); // 5.0
}

#[test]
fn test_float32_div_by_zero() {
    // 5.0 / 0.0 = Inf
    let five = float32_from_bits(0x40A00000); // 5.0
    let zero = float32_zero(0);
    let result = div_float32(five, zero);
    assert(float32_is_infinity(result));
    assert(result.sign == 0);

    // 0.0 / 0.0 = NaN
    let result2 = div_float32(zero, zero);
    assert(float32_is_nan(result2));
}

#[test]
fn test_float32_div_zero_dividend() {
    // 0.0 / 5.0 = 0.0
    let zero = float32_zero(0);
    let five = float32_from_bits(0x40A00000); // 5.0
    let result = div_float32(zero, five);
    assert(float32_is_zero(result));
}

#[test]
fn test_float32_div_negative() {
    // 6.0 / (-2.0) = -3.0
    let six = float32_from_bits(0x40C00000); // 6.0
    let neg_two = float32_from_bits(0xC0000000); // -2.0
    let result = div_float32(six, neg_two);
    assert(float32_to_bits(result) == 0xC0400000); // -3.0

    // (-6.0) / (-2.0) = 3.0
    let neg_six = float32_from_bits(0xC0C00000); // -6.0
    let result2 = div_float32(neg_six, neg_two);
    assert(float32_to_bits(result2) == 0x40400000); // 3.0
}

#[test]
fn test_float32_div_infinity() {
    let inf = float32_infinity(0);
    let two = float32_from_bits(0x40000000); // 2.0

    // Inf * 2.0 = Inf (note: test uses mul, keeping original behavior)
    let result = mul_float32(inf, two);
    assert(float32_is_infinity(result));

    // Inf / Inf = NaN
    let result2 = div_float32(inf, inf);
    assert(float32_is_nan(result2));

    // 2.0 / Inf = 0.0
    let result3 = div_float32(two, inf);
    assert(float32_is_zero(result3));
}

// ============================================================================
// Float32 Comparison Tests
// ============================================================================

#[test]
fn test_float32_eq_basic() {
    let one = float32_from_bits(0x3F800000); // 1.0
    let two = float32_from_bits(0x40000000); // 2.0
    let one_copy = float32_from_bits(0x3F800000); // 1.0

    // 1.0 == 1.0
    assert(float32_eq(one, one_copy));

    // 1.0 != 2.0
    assert(!float32_eq(one, two));
}

#[test]
fn test_float32_eq_zeros() {
    let pos_zero = float32_zero(0);
    let neg_zero = float32_zero(1);

    // +0 == -0
    assert(float32_eq(pos_zero, neg_zero));
    assert(float32_eq(neg_zero, pos_zero));
}

#[test]
fn test_float32_eq_nan() {
    let nan = float32_nan();
    let one = float32_from_bits(0x3F800000);

    // NaN != NaN
    assert(!float32_eq(nan, nan));

    // NaN != 1.0
    assert(!float32_eq(nan, one));
    assert(!float32_eq(one, nan));
}

#[test]
fn test_float32_ne_basic() {
    let one = float32_from_bits(0x3F800000);
    let two = float32_from_bits(0x40000000);

    assert(float32_ne(one, two));
    assert(!float32_ne(one, one));
}

#[test]
fn test_float32_lt_basic() {
    let one = float32_from_bits(0x3F800000); // 1.0
    let two = float32_from_bits(0x40000000); // 2.0
    let neg_one = float32_from_bits(0xBF800000); // -1.0

    // 1.0 < 2.0
    assert(float32_lt(one, two));

    // 2.0 is not < 1.0
    assert(!float32_lt(two, one));

    // 1.0 is not < 1.0
    assert(!float32_lt(one, one));

    // -1.0 < 1.0
    assert(float32_lt(neg_one, one));

    // 1.0 is not < -1.0
    assert(!float32_lt(one, neg_one));
}

#[test]
fn test_float32_lt_zeros() {
    let pos_zero = float32_zero(0);
    let neg_zero = float32_zero(1);

    // +0 is not < -0 (they are equal)
    assert(!float32_lt(pos_zero, neg_zero));
    assert(!float32_lt(neg_zero, pos_zero));
}

#[test]
fn test_float32_lt_infinity() {
    let inf = float32_infinity(0);
    let neg_inf = float32_infinity(1);
    let one = float32_from_bits(0x3F800000);

    // 1.0 < Inf
    assert(float32_lt(one, inf));

    // -Inf < 1.0
    assert(float32_lt(neg_inf, one));

    // -Inf < Inf
    assert(float32_lt(neg_inf, inf));

    // Inf is not < 1.0
    assert(!float32_lt(inf, one));
}

#[test]
fn test_float32_lt_nan() {
    let nan = float32_nan();
    let one = float32_from_bits(0x3F800000);

    // NaN comparisons are always false
    assert(!float32_lt(nan, one));
    assert(!float32_lt(one, nan));
    assert(!float32_lt(nan, nan));
}

#[test]
fn test_float32_le_basic() {
    let one = float32_from_bits(0x3F800000);
    let two = float32_from_bits(0x40000000);

    // 1.0 <= 2.0
    assert(float32_le(one, two));

    // 1.0 <= 1.0
    assert(float32_le(one, one));

    // 2.0 is not <= 1.0
    assert(!float32_le(two, one));
}

#[test]
fn test_float32_gt_basic() {
    let one = float32_from_bits(0x3F800000);
    let two = float32_from_bits(0x40000000);

    // 2.0 > 1.0
    assert(float32_gt(two, one));

    // 1.0 is not > 2.0
    assert(!float32_gt(one, two));

    // 1.0 is not > 1.0
    assert(!float32_gt(one, one));
}

#[test]
fn test_float32_ge_basic() {
    let one = float32_from_bits(0x3F800000);
    let two = float32_from_bits(0x40000000);

    // 2.0 >= 1.0
    assert(float32_ge(two, one));

    // 1.0 >= 1.0
    assert(float32_ge(one, one));

    // 1.0 is not >= 2.0
    assert(!float32_ge(one, two));
}

#[test]
fn test_float32_unordered() {
    let nan = float32_nan();
    let one = float32_from_bits(0x3F800000);

    // NaN makes comparison unordered
    assert(float32_unordered(nan, one));
    assert(float32_unordered(one, nan));
    assert(float32_unordered(nan, nan));

    // No NaN means ordered
    assert(!float32_unordered(one, one));
}

#[test]
fn test_float32_compare_basic() {
    let one = float32_from_bits(0x3F800000);
    let two = float32_from_bits(0x40000000);

    // 1.0 < 2.0 -> -1
    assert(float32_compare(one, two) == -1);

    // 2.0 > 1.0 -> 1
    assert(float32_compare(two, one) == 1);

    // 1.0 == 1.0 -> 0
    assert(float32_compare(one, one) == 0);
}

#[test]
fn test_float32_compare_nan() {
    let nan = float32_nan();
    let neg_nan = ieee754::float::IEEE754Float32 { sign: 1, exponent: 255, mantissa: 0x400000 };
    let one = float32_from_bits(0x3F800000);

    // +NaN > everything
    assert(float32_compare(nan, one) == 1);

    // -NaN < everything
    assert(float32_compare(neg_nan, one) == -1);

    // everything < +NaN
    assert(float32_compare(one, nan) == -1);

    // everything > -NaN
    assert(float32_compare(one, neg_nan) == 1);
}

// ============================================================================
// Float32 Square Root Tests
// ============================================================================

#[test]
fn test_float32_sqrt_basic() {
    // sqrt(4.0) = 2.0
    let four = float32_from_bits(0x40800000); // 4.0
    let result = sqrt_float32(four);
    assert(float32_to_bits(result) == 0x40000000); // 2.0
}

#[test]
fn test_float32_sqrt_one() {
    // sqrt(1.0) = 1.0
    let one = float32_from_bits(0x3F800000); // 1.0
    let result = sqrt_float32(one);
    assert(float32_to_bits(result) == 0x3F800000); // 1.0
}

#[test]
fn test_float32_sqrt_two() {
    // sqrt(2.0) = ~1.4142135...
    let two = float32_from_bits(0x40000000); // 2.0
    let result = sqrt_float32(two);
    // Expected: 0x3FB504F3 (1.4142135f)
    assert(float32_to_bits(result) == 0x3FB504F3);
}

#[test]
fn test_float32_sqrt_nine() {
    // sqrt(9.0) = 3.0
    let nine = float32_from_bits(0x41100000); // 9.0
    let result = sqrt_float32(nine);
    assert(float32_to_bits(result) == 0x40400000); // 3.0
}

#[test]
fn test_float32_sqrt_quarter() {
    // sqrt(0.25) = 0.5
    let quarter = float32_from_bits(0x3E800000); // 0.25
    let result = sqrt_float32(quarter);
    assert(float32_to_bits(result) == 0x3F000000); // 0.5
}

#[test]
fn test_float32_sqrt_zero() {
    // sqrt(+0) = +0
    let pos_zero = float32_zero(0);
    let result = sqrt_float32(pos_zero);
    assert(float32_is_zero(result));
    assert(result.sign == 0);

    // sqrt(-0) = -0
    let neg_zero = float32_zero(1);
    let result2 = sqrt_float32(neg_zero);
    assert(float32_is_zero(result2));
    assert(result2.sign == 1);
}

#[test]
fn test_float32_sqrt_infinity() {
    // sqrt(+Inf) = +Inf
    let inf = float32_infinity(0);
    let result = sqrt_float32(inf);
    assert(float32_is_infinity(result));
    assert(result.sign == 0);
}

#[test]
fn test_float32_sqrt_negative() {
    // sqrt(negative) = NaN
    let neg_one = float32_from_bits(0xBF800000); // -1.0
    let result = sqrt_float32(neg_one);
    assert(float32_is_nan(result));

    // sqrt(-Inf) = NaN
    let neg_inf = float32_infinity(1);
    let result2 = sqrt_float32(neg_inf);
    assert(float32_is_nan(result2));
}

#[test]
fn test_float32_sqrt_nan() {
    // sqrt(NaN) = NaN
    let nan = float32_nan();
    let result = sqrt_float32(nan);
    assert(float32_is_nan(result));
}

// ============================================================================
// Float32 Absolute Value Tests
// ============================================================================

#[test]
fn test_float32_abs_positive() {
    // abs(1.0) = 1.0
    let one = float32_from_bits(0x3F800000); // 1.0
    let result = abs_float32(one);
    assert(float32_to_bits(result) == 0x3F800000);

    // abs(3.14) = 3.14
    let pi = float32_from_bits(0x4048F5C3); // ~3.14
    let result2 = abs_float32(pi);
    assert(float32_to_bits(result2) == 0x4048F5C3);
}

#[test]
fn test_float32_abs_negative() {
    // abs(-1.0) = 1.0
    let neg_one = float32_from_bits(0xBF800000); // -1.0
    let result = abs_float32(neg_one);
    assert(float32_to_bits(result) == 0x3F800000); // 1.0

    // abs(-3.14) = 3.14
    let neg_pi = float32_from_bits(0xC048F5C3); // ~-3.14
    let result2 = abs_float32(neg_pi);
    assert(float32_to_bits(result2) == 0x4048F5C3); // ~3.14
}

#[test]
fn test_float32_abs_zero() {
    // abs(+0) = +0
    let pos_zero = float32_zero(0);
    let result = abs_float32(pos_zero);
    assert(float32_is_zero(result));
    assert(result.sign == 0);

    // abs(-0) = +0
    let neg_zero = float32_zero(1);
    let result2 = abs_float32(neg_zero);
    assert(float32_is_zero(result2));
    assert(result2.sign == 0);
}

#[test]
fn test_float32_abs_infinity() {
    // abs(+Inf) = +Inf
    let pos_inf = float32_infinity(0);
    let result = abs_float32(pos_inf);
    assert(float32_is_infinity(result));
    assert(result.sign == 0);

    // abs(-Inf) = +Inf
    let neg_inf = float32_infinity(1);
    let result2 = abs_float32(neg_inf);
    assert(float32_is_infinity(result2));
    assert(result2.sign == 0);
}

#[test]
fn test_float32_abs_nan() {
    // abs(NaN) = NaN (with sign bit cleared)
    let nan = float32_nan();
    let result = abs_float32(nan);
    assert(float32_is_nan(result));
    assert(result.sign == 0);
}
