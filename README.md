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

## Project Structure

```
noir_IEEE754/
├── Nargo.toml              # Noir project configuration
├── src/
│   ├── lib.nr              # Module exports and basic tests
│   ├── float.nr            # IEEE 754 implementation
│   └── ieee754_tests.nr    # Auto-generated test suite
├── scripts/
│   └── generate_tests.py   # Test generation script
└── .ieee754_test_cache/    # Downloaded test files (gitignored)
```

## Test Infrastructure

This project uses the [IBM FPgen IEEE 754 test suite](https://github.com/sergev/ieee754-test-suite) to validate the implementation against comprehensive edge cases.

### Generating Tests

The `generate_tests.py` script automatically downloads test files and generates Noir test code:

```bash
# Generate tests from Add-Shift.fptest (default)
python3 scripts/generate_tests.py -o src/ieee754_tests.nr

# Use specific test files
python3 scripts/generate_tests.py --files Add-Shift.fptest Rounding.fptest -o src/ieee754_tests.nr

# Generate tests from all available test files
python3 scripts/generate_tests.py --all -o src/ieee754_tests.nr

# List available test files
python3 scripts/generate_tests.py --list

# Filter by operation type
python3 scripts/generate_tests.py --operation add --precision f32 -o src/ieee754_tests.nr

# Clear cache and re-download
python3 scripts/generate_tests.py --clear-cache -o src/ieee754_tests.nr
```

### Available Test Files

| File | Description |
|------|-------------|
| `Add-Cancellation-And-Subnorm-Result.fptest` | Cancellation leading to subnormal results |
| `Add-Cancellation.fptest` | Catastrophic cancellation cases |
| `Add-Shift-And-Special-Significands.fptest` | Alignment shift with special significands |
| `Add-Shift.fptest` | Alignment shift edge cases |
| `Basic-Types-Inputs.fptest` | Basic input type testing |
| `Basic-Types-Intermediate.fptest` | Intermediate calculation types |
| `Compare-Different-Input-Field-Relations.fptest` | Comparison operations |
| `Corner-Rounding.fptest` | Corner case rounding scenarios |
| `Divide-Divide-By-Zero-Exception.fptest` | Division by zero handling |
| `Divide-Trailing-Zeros.fptest` | Division with trailing zeros |
| `Hamming-Distance.fptest` | Hamming distance edge cases |
| `Input-Special-Significand.fptest` | Special significand inputs |
| `Overflow.fptest` | Overflow boundary cases |
| `Rounding.fptest` | Rounding mode edge cases |
| `Sticky-Bit-Calculation.fptest` | Sticky bit handling |
| `Underflow.fptest` | Underflow edge cases |
| `Vicinity-Of-Rounding-Boundaries.fptest` | Near rounding boundary cases |

### Running Tests

```bash
# Run all tests (may take a long time with ~18k tests)
nargo test

# Run tests from a specific test file
nargo test ieee754_tests::test_add_shift::

# Run a single specific test
nargo test ieee754_tests::test_add_shift::test_f32_add_0

# Run the manual unit tests only
nargo test test_float32
nargo test test_float64
```

### Test Modules

Tests are split into separate modules by source file for faster execution:

| Module | Source File | Test Count |
|--------|-------------|------------|
| `test_add_shift` | Add-Shift.fptest | ~57 |
| `test_add_cancellation` | Add-Cancellation.fptest | ~18 |
| `test_add_cancellation_and_subnorm_result` | Add-Cancellation-And-Subnorm-Result.fptest | ~313 |
| `test_add_shift_and_special_significands` | Add-Shift-And-Special-Significands.fptest | ~16k |
| `test_basic_types_inputs` | Basic-Types-Inputs.fptest | ~882 |
| `test_basic_types_intermediate` | Basic-Types-Intermediate.fptest | ~40 |
| `test_hamming_distance` | Hamming-Distance.fptest | ~55 |
| `test_overflow` | Overflow.fptest | ~62 |
| `test_rounding` | Rounding.fptest | ~16 |
| `test_underflow` | Underflow.fptest | ~20 |
| `test_vicinity_of_rounding_boundaries` | Vicinity-Of-Rounding-Boundaries.fptest | ~31 |

## Development Status

### Current State

The library currently implements IEEE 754 addition for both float32 and float64. Initial testing shows ~40% of IBM FPgen test cases pass, with failures primarily in:

- **Rounding edge cases**: Ties-to-even in boundary conditions
- **Denormal number handling**: Gradual underflow scenarios
- **Sticky bit propagation**: During large alignment shifts

### Next Steps

1. **Expand Test Coverage**: Generate and run tests from all available IEEE 754 test files
2. **Fix Rounding Issues**: Analyze failing tests and correct rounding implementation
3. **Implement Remaining Operations**: Subtraction, multiplication, division
4. **Support Multiple Rounding Modes**: Round toward +∞, -∞, zero

---

## Agent Instructions

This section provides guidance for AI coding agents working on this library.

### Understanding the Codebase

1. **Core Implementation**: `src/float.nr` contains all IEEE 754 types and operations
2. **Test Generation**: `scripts/generate_tests.py` parses IBM FPgen format and generates Noir tests
3. **Generated Tests**: `src/ieee754_tests.nr` contains auto-generated test cases (do not edit manually)

### Noir Language Constraints

When implementing or fixing code in this library, be aware of these Noir-specific constraints:

- **No early returns**: All paths must reach the end of the function; use conditional assignment patterns
- **Fixed-width integers**: Use `u1`, `u8`, `u16`, `u32`, `u64` as appropriate
- **Shift operand types must match**: `value >> shift_amount` requires same-width operands
- **No floating-point primitives**: Everything must be implemented using integer arithmetic
- **`pub` visibility required**: Functions used across modules need `pub` keyword

### Test-Driven Development Workflow

1. **Generate comprehensive tests**:
   ```bash
   python3 scripts/generate_tests.py --all -o src/ieee754_tests.nr
   ```

2. **Run tests and identify failures**:
   ```bash
   nargo test 2>&1 | grep -E "(PASS|FAIL)"
   ```

3. **Analyze a specific failing test**:
   - Find the test in `src/ieee754_tests.nr`
   - Extract the input bit patterns and expected output
   - Trace through `add_float32` or `add_float64` logic
   - Use `--debug` flag in generator for println debugging:
     ```bash
     python3 scripts/generate_tests.py --debug -o src/ieee754_tests.nr
     ```

4. **Fix the implementation in `src/float.nr`** and re-run tests

### Common Bug Patterns to Watch For

1. **Off-by-one in bit positions**: Mantissa is 23 bits for float32, but implicit bit is at position 23 (bit 24)
2. **Guard/Round/Sticky bit handling**: Ensure proper preservation during alignment shifts
3. **Overflow detection**: Check after both addition and rounding
4. **Denormal transitions**: Numbers can transition between denormal and normal during rounding
5. **Zero sign handling**: `-0 + -0 = -0`, but `-0 + +0 = +0` (positive zero dominates)

### Key Implementation Details

**Float32 Addition Algorithm** (`add_float32` in `float.nr`):
1. Extract mantissas with implicit bit (denormals don't have implicit bit)
2. Shift mantissas left by 3 bits for guard/round/sticky
3. Align mantissas by shifting smaller exponent's mantissa right
4. Add or subtract based on signs
5. Normalize result (handle overflow and leading zeros)
6. Round to nearest, ties to even
7. Handle special cases: NaN, ±Infinity, ±Zero

**Test File Format** (`.fptest`):
```
b32+ =0 +1.000000P+0 +1.000000P+0 -> +1.000000P+1
│    │  │            │               │
│    │  │            │               └── Expected result
│    │  │            └── Second operand  
│    │  └── First operand (sign, significand, exponent)
│    └── Rounding mode (=0 = round to nearest even)
└── Precision (b32) and operation (+)
```

### Implementation Priority

When fixing or extending this library:

1. **First**: Fix all `Add-Shift.fptest` tests (alignment edge cases)
2. **Second**: Fix `Rounding.fptest` tests (rounding correctness)
3. **Third**: Fix `Cancellation.fptest` and `Sticky-Bit-Cancellation.fptest`
4. **Fourth**: Handle `Underflow-Shift.fptest` and `Overflow-Shift.fptest`
5. **Fifth**: Implement subtraction (negate second operand, call add)
6. **Sixth**: Implement multiplication and division

### Debugging Tips

- Compare your implementation against a reference (e.g., Python's `struct` module)
- Use `float32_to_bits` / `float64_to_bits` to compare bit patterns
- The test generator's `--debug` flag adds `println` statements showing intermediate values
- Pay special attention to cases where operands differ by exactly the mantissa width in exponent

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting PRs:

```bash
# Generate full test suite
python3 scripts/generate_tests.py --all -o src/ieee754_tests.nr

# Run all tests
nargo test
```

## License

[MIT License](LICENSE)

## References

- [IEEE 754-2019 Standard](https://ieeexplore.ieee.org/document/8766229)
- [IBM FPgen Test Suite](https://github.com/sergev/ieee754-test-suite)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- [Noir Language Documentation](https://noir-lang.org/docs)
