# Copilot Instructions for noir_IEEE754

IEEE 754 compliant floating-point arithmetic library for Noir (ZK proof DSL).

## Architecture Overview

- **`src/float.nr`**: Core IEEE 754 implementation with `IEEE754Float32`/`IEEE754Float64` structs and operations
- **`src/ieee754_tests/`**: Auto-generated tests from IBM FPgen suite (~18k tests) - **never edit manually**
- **`scripts/generate_tests.py`**: Downloads test cases from [sergev/ieee754-test-suite](https://github.com/sergev/ieee754-test-suite) and generates Noir tests using Python's `struct` module for IEEE 754 ground truth

## Noir Language Constraints

- **No early returns**: Use conditional assignment patterns; all paths must reach function end
- **Fixed-width integers**: `u1`, `u8`, `u16`, `u32`, `u64` only
- **Shift operand type matching**: Both operands in `value >> shift` must have same width
- **No floating-point primitives**: All FP arithmetic uses integer math
- **`pub` required**: Functions used across modules need `pub` keyword

## Key Implementation Patterns

### Float Addition Algorithm (`add_float32`/`add_float64`)
1. Extract mantissa with implicit bit (denormals lack implicit bit)
2. Left-shift mantissa by 3 bits for guard/round/sticky bits
3. Align by right-shifting smaller exponent's mantissa (preserve sticky bit)
4. Add/subtract based on signs
5. Normalize (handle overflow and leading zeros)
6. Round to nearest, ties to even
7. Handle special cases last: NaN > Infinity > Zero priority

### Sticky Bit Preservation
```noir
// shift_right_sticky_u64 preserves bits shifted out
let mask = (1 << shift) - 1;
let shifted_out = value & mask;
let result = value >> shift;
if shifted_out != 0 { result | 1 } else { result }
```

## Common Bug Patterns

- **Off-by-one bit positions**: Mantissa is 23 bits (float32) but implicit bit is at position 23
- **Denormal transitions**: Numbers can transition denormal↔normal during rounding
- **Zero sign handling**: `-0 + -0 = -0`, but `-0 + +0 = +0`
- **Denormal results**: When exponent underflows, don't re-shift mantissa—normalization loop handles it

## Developer Commands

```bash
# Generate tests (default: Add-Shift.fptest)
python3 scripts/generate_tests.py -o src/ieee754_tests.nr

# Generate all ~18k tests
python3 scripts/generate_tests.py --all -o src/ieee754_tests.nr

# Run specific test module (RECOMMENDED - full suite takes hours)
nargo test ieee754_tests::test_add_shift::

# Run single chunk (~25 tests, fastest iteration)
nargo test ieee754_tests::test_add_shift::chunk_0000::

# Run single test
nargo test ieee754_tests::test_add_shift::chunk_0000::test_f32_add_0

# Run manual unit tests only
nargo test test_float32
```

> ⚠️ **Warning**: Running `nargo test` without filters executes ~18k tests and takes several hours. Always run individual chunks or modules during development.

## Test File Format (`.fptest`)
```
b32+ =0 +1.000000P+0 +1.000000P+0 -> +1.000000P+1
│    │  │            │               │
│    │  operand1     operand2        expected result
│    └── rounding mode (=0 = round to nearest even)
└── precision (b32) and operation (+)
```

## Extending Operations

1. **Subtraction**: Negate second operand, call addition
2. **Multiplication/Division**: Need new test generation for `*`/`/` operations
3. **Rounding modes**: Add parameter for `>` (toward +∞), `<` (toward -∞), `0` (toward zero)

## Critical Files

| File | Purpose |
|------|---------|
| `src/float.nr` | All IEEE 754 types and arithmetic |
| `scripts/generate_tests.py` | Test generation from IBM FPgen |
| `src/ieee754_tests/mod.nr` | Test module index |
