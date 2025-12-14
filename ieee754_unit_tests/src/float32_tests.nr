// ============================================================================
// IEEE 754 Float32 Unit Tests
// ============================================================================

use ieee754::float::{
    add_float32, div_float32, float32_from_bits, float32_infinity, float32_is_infinity,
    float32_is_nan, float32_is_zero, float32_nan, float32_to_bits, float32_zero, mul_float32,
    sub_float32,
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
