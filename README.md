# noir_IEEE754

IEEE 754 compliant floating-point arithmetic library for [Noir](https://noir-lang.org/), a domain-specific language for zero-knowledge proofs.

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
| Subtraction | 🚧     | 🚧      |
| Multiplication | 🚧  | 🚧      |
| Division | 🚧       | 🚧      |

> 🚧 = Planned / In Progress

## Installation

Add to your `Nargo.toml`:

```toml
[dependencies]
noir_IEEE754 = { git = "https://github.com/jeswr/noir_IEEE754", tag = "v0.1.0" }
```

## Usage

```noir
use noir_IEEE754::float::{
    IEEE754Float32, IEEE754Float64,
    float32_from_bits, float32_to_bits,
    float64_from_bits, float64_to_bits,
    add_float32, add_float64
};

fn main() {
    // Create floats from bit representation
    let a = float32_from_bits(0x3F800000); // 1.0f
    let b = float32_from_bits(0x40000000); // 2.0f
    
    // Perform addition
    let result = add_float32(a, b);
    
    // Convert back to bits
    let bits = float32_to_bits(result); // 0x40400000 = 3.0f
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

// Same functions available for float64 with float64_ prefix
```

## Development Status

### Current State

The library implements IEEE 754 addition for both float32 and float64 with full support for:

- ✅ **Normalized numbers**: Standard floating-point values
- ✅ **Denormalized (subnormal) numbers**: Gradual underflow handling
- ✅ **Special values**: ±Zero, ±Infinity, NaN (quiet and signaling)
- ✅ **Round-to-nearest, ties-to-even**: Default IEEE 754 rounding mode
- ✅ **Guard, round, and sticky bits**: For correct rounding during alignment shifts

### Next Steps

1. **Implement Remaining Operations**: Subtraction, multiplication, division
2. **Support Multiple Rounding Modes**: Round toward +∞, -∞, zero
3. **Optimize Performance**: Reduce constraint count for ZK circuits

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for test infrastructure details and development workflow.

## License

[MIT License](LICENSE)

## References

- [IEEE 754-2019 Standard](https://ieeexplore.ieee.org/document/8766229)
- [Noir Language Documentation](https://noir-lang.org/docs)
