# noir_IEEE754

IEEE 754 compliant floating-point arithmetic library for [Noir](https://noir-lang.org/), a domain-specific language for zero-knowledge proofs.

> [!CAUTION]
> **Security Warning**: This library has **not been security reviewed** and should not be used in production systems without a thorough audit.

> [!WARNING]
> **AI-Generated Code**: This library is **largely AI-generated**. While it has been tested against the IBM FPgen test suite, there may be edge cases or subtle bugs that have not been discovered.

> [!NOTE]
> **Test Coverage Limitations**: The following IEEE 754 test cases are skipped:
> - **Non-default rounding modes**: Only "round to nearest, ties to even" (`=0`) is supported. Tests using round toward +∞ (`>`), -∞ (`<`), zero (`0`), or nearest away (`=^`) are skipped.
> - **Non-binary operations**: FMA (`*+`) and remainder (`%`) operations are not implemented.
> - **Comparison and square root operations**: Not tested with the IBM FPgen suite (only unit tests).
> - **Known bad tests in IBM FPgen suite**: The test `b32/ =0 +1.2CEE1BP-64 +1.50EFBDP-30` from `Divide-Divide-By-Zero-Exception.fptest` is skipped due to an incorrect expected result in the test suite.
> - **Underflow edge cases**: Tests where operands underflow to zero during conversion are skipped.

## Overview

This library provides IEEE 754 standard floating-point operations in Noir, enabling verified floating-point computations in zero-knowledge circuits. The implementation supports both single-precision (binary32/float) and double-precision (binary64/double) formats.

### Features

- **IEEE 754 Float32** (single-precision): 1 sign bit, 8 exponent bits (bias 127), 23 mantissa bits
- **IEEE 754 Float64** (double-precision): 1 sign bit, 11 exponent bits (bias 1023), 52 mantissa bits
- Special value handling: ±Infinity, ±Zero, NaN, denormalized numbers
- Rounding mode: Round to nearest, ties to even (default IEEE 754 rounding)
- Bit-level conversion functions (`from_bits`, `to_bits`)

### Current Operations

| Operation | Float32 | Float64 |
|-----------|---------|---------|
| Addition  | ✅       | ✅       |
| Subtraction | ✅     | ✅      |
| Multiplication | ✅  | ✅      |
| Division | ✅       | ✅      |
| Square Root | ✅     | ✅      |
| Comparison (eq, ne, lt, le, gt, ge) | ✅ | ✅ |

## Installation

Add to your `Nargo.toml`:

```toml
[dependencies]
ieee754 = { git = "https://github.com/jeswr/noir_IEEE754", tag = "v0.1.0", directory = "ieee754" }
```

## Usage

```noir
use ieee754::float::{
    IEEE754Float32, IEEE754Float64,
    float32_from_bits, float32_to_bits,
    float64_from_bits, float64_to_bits,
    add_float32, add_float64,
    sub_float32, sub_float64,
    mul_float32, mul_float64,
    div_float32, div_float64,
    sqrt_float32, sqrt_float64,
    // Comparison operations
    float32_eq, float32_ne, float32_lt, float32_le, float32_gt, float32_ge,
    float64_eq, float64_ne, float64_lt, float64_le, float64_gt, float64_ge
};

fn main() {
    // Create floats from bit representation
    let a = float32_from_bits(0x40400000); // 3.0f
    let b = float32_from_bits(0x40000000); // 2.0f
    
    // Perform arithmetic operations
    let sum = add_float32(a, b);        // 3.0 + 2.0 = 5.0
    let diff = sub_float32(a, b);       // 3.0 - 2.0 = 1.0
    let product = mul_float32(a, b);    // 3.0 * 2.0 = 6.0
    let quotient = div_float32(a, b);   // 3.0 / 2.0 = 1.5
    let root = sqrt_float32(a);         // sqrt(3.0) ≈ 1.732
    
    // Convert back to bits
    let sum_bits = float32_to_bits(sum);         // 0x40A00000 = 5.0f
    let diff_bits = float32_to_bits(diff);       // 0x3F800000 = 1.0f
    let product_bits = float32_to_bits(product); // 0x40C00000 = 6.0f
    let quotient_bits = float32_to_bits(quotient); // 0x3FC00000 = 1.5f
}
```

### Helper Functions

```noir
// Special value checks
float32_is_nan(x)       // Check if NaN
float32_is_infinity(x)  // Check if ±Infinity
float32_is_zero(x)      // Check if ±0
float32_is_denormal(x)  // Check if denormalized

// Create special values
float32_nan()           // Returns NaN
float32_infinity(sign)  // Returns ±Infinity
float32_zero(sign)      // Returns ±0

// Square root (IEEE 754 compliant)
sqrt_float32(x)         // sqrt(x), returns NaN for negative inputs (except -0)

// Comparison functions (IEEE 754 compliant)
float32_eq(a, b)        // a == b (NaN != NaN, +0 == -0)
float32_ne(a, b)        // a != b
float32_lt(a, b)        // a < b (false if either is NaN)
float32_le(a, b)        // a <= b (false if either is NaN)
float32_gt(a, b)        // a > b (false if either is NaN)
float32_ge(a, b)        // a >= b (false if either is NaN)
float32_unordered(a, b) // true if either is NaN
float32_compare(a, b)   // -1, 0, or 1 (total ordering including NaN)

// Same functions available for float64 with float64_ prefix (except sqrt uses sqrt_float64)
```

## Development Status

### Current State

The library implements IEEE 754 arithmetic for both float32 and float64 with full support for:

- ✅ **All basic operations**: Addition, subtraction, multiplication, division, square root
- ✅ **Comparison operations**: eq, ne, lt, le, gt, ge, unordered, compare
- ✅ **Normalized numbers**: Standard floating-point values
- ✅ **Denormalized (subnormal) numbers**: Gradual underflow handling
- ✅ **Special values**: ±Zero, ±Infinity, NaN (quiet and signaling)
- ✅ **Round-to-nearest, ties-to-even**: Default IEEE 754 rounding mode
- ✅ **Guard, round, and sticky bits**: For correct rounding during alignment shifts

### Next Steps

1. **Support Multiple Rounding Modes**: Round toward +∞, -∞, zero
2. **Optimize Performance**: Reduce constraint count for ZK circuits
3. **Add FMA operation**: `fma_float32`/`fma_float64`

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for test infrastructure details and development workflow.

## License

[MIT License](LICENSE)

## References

- [IEEE 754-2019 Standard](https://ieeexplore.ieee.org/document/8766229)
- [Noir Language Documentation](https://noir-lang.org/docs)
