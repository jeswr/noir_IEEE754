// IEEE 754 Packed Representation Tests
// Tests for operations on packed u32/u64 bit patterns

use dep::ieee754;

// ============================================================================
// Float32 Packed Representation Tests
// ============================================================================

#[test]
fn test_packed_float32_addition() {
    // 1.0 + 2.0 = 3.0
    let a: u32 = 0x3F800000; // 1.0
    let b: u32 = 0x40000000; // 2.0
    let expected: u32 = 0x40400000; // 3.0

    let result = ieee754::add_float32_bits(a, b);
    assert(result == expected);
}

#[test]
fn test_packed_float32_subtraction() {
    // 5.0 - 2.0 = 3.0
    let a: u32 = 0x40A00000; // 5.0
    let b: u32 = 0x40000000; // 2.0
    let expected: u32 = 0x40400000; // 3.0

    let result = ieee754::sub_float32_bits(a, b);
    assert(result == expected);
}

#[test]
fn test_packed_float32_multiplication() {
    // 2.0 * 3.0 = 6.0
    let a: u32 = 0x40000000; // 2.0
    let b: u32 = 0x40400000; // 3.0
    let expected: u32 = 0x40C00000; // 6.0

    let result = ieee754::mul_float32_bits(a, b);
    assert(result == expected);
}

#[test]
fn test_packed_float32_division() {
    // 6.0 / 2.0 = 3.0
    let a: u32 = 0x40C00000; // 6.0
    let b: u32 = 0x40000000; // 2.0
    let expected: u32 = 0x40400000; // 3.0

    let result = ieee754::div_float32_bits(a, b);
    assert(result == expected);
}

#[test]
fn test_packed_float32_sqrt() {
    // sqrt(4.0) = 2.0
    let a: u32 = 0x40800000; // 4.0
    let expected: u32 = 0x40000000; // 2.0

    let result = ieee754::sqrt_float32_bits(a);
    assert(result == expected);
}

#[test]
fn test_packed_float32_abs() {
    // abs(-2.0) = 2.0
    let a: u32 = 0xC0000000; // -2.0
    let expected: u32 = 0x40000000; // 2.0

    let result = ieee754::abs_float32_bits(a);
    assert(result == expected);
}

#[test]
fn test_packed_float32_comparison() {
    let a: u32 = 0x40000000; // 2.0
    let b: u32 = 0x40400000; // 3.0

    assert(ieee754::float32_lt_bits(a, b));
    assert(!ieee754::float32_gt_bits(a, b));
    assert(ieee754::float32_le_bits(a, b));
    assert(ieee754::float32_ne_bits(a, b));
}

#[test]
fn test_packed_float32_equality() {
    let a: u32 = 0x40000000; // 2.0
    let b: u32 = 0x40000000; // 2.0

    assert(ieee754::float32_eq_bits(a, b));
    assert(!ieee754::float32_ne_bits(a, b));
}

// ============================================================================
// Float64 Packed Representation Tests
// ============================================================================

#[test]
fn test_packed_float64_addition() {
    // 1.0 + 2.0 = 3.0
    let a: u64 = 0x3FF0000000000000; // 1.0
    let b: u64 = 0x4000000000000000; // 2.0
    let expected: u64 = 0x4008000000000000; // 3.0

    let result = ieee754::add_float64_bits(a, b);
    assert(result == expected);
}

#[test]
fn test_packed_float64_subtraction() {
    // 5.0 - 2.0 = 3.0
    let a: u64 = 0x4014000000000000; // 5.0
    let b: u64 = 0x4000000000000000; // 2.0
    let expected: u64 = 0x4008000000000000; // 3.0

    let result = ieee754::sub_float64_bits(a, b);
    assert(result == expected);
}

#[test]
fn test_packed_float64_multiplication() {
    // 2.0 * 3.0 = 6.0
    let a: u64 = 0x4000000000000000; // 2.0
    let b: u64 = 0x4008000000000000; // 3.0
    let expected: u64 = 0x4018000000000000; // 6.0

    let result = ieee754::mul_float64_bits(a, b);
    assert(result == expected);
}

#[test]
fn test_packed_float64_division() {
    // 6.0 / 2.0 = 3.0
    let a: u64 = 0x4018000000000000; // 6.0
    let b: u64 = 0x4000000000000000; // 2.0
    let expected: u64 = 0x4008000000000000; // 3.0

    let result = ieee754::div_float64_bits(a, b);
    assert(result == expected);
}

#[test]
fn test_packed_float64_sqrt() {
    // sqrt(4.0) = 2.0
    let a: u64 = 0x4010000000000000; // 4.0
    let expected: u64 = 0x4000000000000000; // 2.0

    let result = ieee754::sqrt_float64_bits(a);
    assert(result == expected);
}

#[test]
fn test_packed_float64_abs() {
    // abs(-2.0) = 2.0
    let a: u64 = 0xC000000000000000; // -2.0
    let expected: u64 = 0x4000000000000000; // 2.0

    let result = ieee754::abs_float64_bits(a);
    assert(result == expected);
}

#[test]
fn test_packed_float64_comparison() {
    let a: u64 = 0x4000000000000000; // 2.0
    let b: u64 = 0x4008000000000000; // 3.0

    assert(ieee754::float64_lt_bits(a, b));
    assert(!ieee754::float64_gt_bits(a, b));
    assert(ieee754::float64_le_bits(a, b));
    assert(ieee754::float64_ne_bits(a, b));
}

#[test]
fn test_packed_float64_equality() {
    let a: u64 = 0x4000000000000000; // 2.0
    let b: u64 = 0x4000000000000000; // 2.0

    assert(ieee754::float64_eq_bits(a, b));
    assert(!ieee754::float64_ne_bits(a, b));
}
