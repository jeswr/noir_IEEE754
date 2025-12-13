#!/usr/bin/env python3
"""
IEEE 754 Test Suite to Noir Test Generator

This script parses .fptest files from the IBM FPgen test suite 
(https://github.com/sergev/ieee754-test-suite) and generates Noir test code.

Test files are automatically downloaded and cached locally.

Test file format:
  <precision><op> <rounding> [exception-flags] <operand1> [operand2] [operand3] -> <result> [result-flags]

Where:
  - precision: b32 (binary32/float), b64 (binary64/double), etc.
  - op: + (add), - (subtract), * (multiply), / (divide), *+ (fma), etc.
  - rounding: =0 (nearest even), =^ (nearest away), > (toward +inf), < (toward -inf), 0 (toward zero)
  - operand format: <sign><significand>P<exponent> or special values (+Inf, -Inf, +Zero, -Zero, Q, S)

Usage:
  python generate_tests.py [--output <output_file.nr>]
  python generate_tests.py --files Add-Shift.fptest Basic-Types-Inputs.fptest
  python generate_tests.py --all
"""

import argparse
import os
import re
import struct
import urllib.request
import urllib.error
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# GitHub raw URL base for the IEEE 754 test suite
TEST_SUITE_BASE_URL = "https://raw.githubusercontent.com/sergev/ieee754-test-suite/master"

# Available test files in the repository (binary floating-point only, not decimal)
AVAILABLE_TEST_FILES = [
    "Add-Cancellation-And-Subnorm-Result.fptest",
    "Add-Cancellation.fptest",
    "Add-Shift-And-Special-Significands.fptest",
    "Add-Shift.fptest",
    "Basic-Types-Inputs.fptest",
    "Basic-Types-Intermediate.fptest",
    "Compare-Different-Input-Field-Relations.fptest",
    "Corner-Rounding.fptest",
    "Divide-Divide-By-Zero-Exception.fptest",
    "Divide-Trailing-Zeros.fptest",
    "Hamming-Distance.fptest",
    "Input-Special-Significand.fptest",
    "Overflow.fptest",
    "Rounding.fptest",
    "Sticky-Bit-Calculation.fptest",
    "Underflow.fptest",
    "Vicinity-Of-Rounding-Boundaries.fptest",
]

# Default cache directory (relative to script location)
DEFAULT_CACHE_DIR = ".ieee754_test_cache"


class Precision(Enum):
    BINARY32 = "b32"
    BINARY64 = "b64"


class Operation(Enum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    FMA = "*+"
    SQRT = "V"
    REM = "%"


class RoundingMode(Enum):
    NEAREST_EVEN = "=0"
    NEAREST_AWAY = "=^"
    TOWARD_POSITIVE = ">"
    TOWARD_NEGATIVE = "<"
    TOWARD_ZERO = "0"


@dataclass
class FPValue:
    """Represents a floating-point value from the test file."""
    sign: int  # 0 for positive, 1 for negative
    significand: str  # hex string without the leading 1. part
    exponent: int  # unbiased exponent
    is_zero: bool = False
    is_inf: bool = False
    is_nan: bool = False
    is_snan: bool = False  # signaling NaN


@dataclass
class TestCase:
    """Represents a single test case."""
    precision: Precision
    operation: Operation
    rounding: RoundingMode
    operand1: FPValue
    operand2: Optional[FPValue]
    operand3: Optional[FPValue]  # for FMA
    result: FPValue
    exception_flags: str
    line_number: int
    raw_line: str


def parse_fp_value(value_str: str) -> FPValue:
    """
    Parse a floating-point value from the test file format.
    
    Format: <sign><significand>P<exponent>
    Examples: +1.01FD72P-118, -0.7FFFFFP-126, +Inf, -Zero, Q, S
    """
    value_str = value_str.strip()
    
    # Handle special values
    if value_str in ("+Inf", "Inf"):
        return FPValue(sign=0, significand="", exponent=0, is_inf=True)
    if value_str == "-Inf":
        return FPValue(sign=1, significand="", exponent=0, is_inf=True)
    if value_str in ("+Zero", "Zero"):
        return FPValue(sign=0, significand="", exponent=0, is_zero=True)
    if value_str == "-Zero":
        return FPValue(sign=1, significand="", exponent=0, is_zero=True)
    if value_str == "Q":  # Quiet NaN
        return FPValue(sign=0, significand="", exponent=0, is_nan=True)
    if value_str == "S":  # Signaling NaN
        return FPValue(sign=0, significand="", exponent=0, is_nan=True, is_snan=True)
    if value_str == "#":  # Generic result (don't care)
        return FPValue(sign=0, significand="", exponent=0, is_nan=True)
    
    # Parse regular values: <sign><significand>P<exponent>
    # Examples: +1.01FD72P-118, -0.7FFFFFP-126
    match = re.match(r'^([+-]?)(\d+)\.([0-9A-Fa-f]+)P([+-]?\d+)$', value_str)
    if not match:
        raise ValueError(f"Cannot parse FP value: {value_str}")
    
    sign_str, int_part, frac_part, exp_str = match.groups()
    sign = 1 if sign_str == '-' else 0
    exponent = int(exp_str)
    
    # Combine integer and fractional parts
    # int_part is typically 0 or 1
    significand = int_part + frac_part
    
    return FPValue(sign=sign, significand=significand, exponent=exponent)


def fp_value_to_bits32(val: FPValue) -> int:
    """Convert an FPValue to IEEE 754 binary32 bits."""
    if val.is_nan:
        # Return a canonical quiet NaN
        if val.is_snan:
            return 0x7F800001  # Signaling NaN
        return 0x7FC00000  # Quiet NaN
    
    if val.is_inf:
        return 0xFF800000 if val.sign else 0x7F800000
    
    if val.is_zero:
        return 0x80000000 if val.sign else 0x00000000
    
    # Parse the significand
    # Format: "1" + hex_fraction or "0" + hex_fraction (for denormals)
    # The hex_fraction in the test file represents the exact mantissa bits
    significand = val.significand
    is_denormal = significand[0] == '0'
    
    # Get the fraction part (after the implicit bit indicator)
    frac_hex = significand[1:]
    frac_bits = int(frac_hex, 16) if frac_hex else 0
    
    # The test file format uses 6 hex digits (24 bits) for the mantissa
    # IEEE 754 float32 has 23 bits, so we need to shift right by 1
    # to get the stored mantissa value
    hex_bits = len(frac_hex) * 4
    if hex_bits > 23:
        # Standard case: 6 hex digits = 24 bits, need 23
        frac_bits >>= (hex_bits - 23)
    elif hex_bits < 23:
        frac_bits <<= (23 - hex_bits)
    
    mantissa = frac_bits & 0x7FFFFF
    
    if is_denormal:
        # Denormalized number: exponent = 0
        biased_exp = 0
    else:
        # Normalized number
        # Calculate biased exponent
        biased_exp = val.exponent + 127
        
        # Handle overflow/underflow
        if biased_exp >= 255:
            # Overflow to infinity
            return (val.sign << 31) | 0x7F800000
        if biased_exp <= 0:
            # Underflow to zero or denormal
            if biased_exp < -23:
                return val.sign << 31  # Zero
            # Denormalize
            shift = 1 - biased_exp
            mantissa = ((1 << 23) | mantissa) >> shift
            biased_exp = 0
    
    return (val.sign << 31) | (biased_exp << 23) | mantissa


def fp_value_to_bits64(val: FPValue) -> int:
    """Convert an FPValue to IEEE 754 binary64 bits."""
    if val.is_nan:
        # Return a canonical quiet NaN
        if val.is_snan:
            return 0x7FF0000000000001  # Signaling NaN
        return 0x7FF8000000000000  # Quiet NaN
    
    if val.is_inf:
        return 0xFFF0000000000000 if val.sign else 0x7FF0000000000000
    
    if val.is_zero:
        return 0x8000000000000000 if val.sign else 0x0000000000000000
    
    # Parse the significand
    significand = val.significand
    is_denormal = significand[0] == '0'
    
    # Get the fraction part (after the implicit bit indicator)
    frac_hex = significand[1:]
    frac_bits = int(frac_hex, 16) if frac_hex else 0
    
    # Adjust to 52-bit mantissa
    hex_bits = len(frac_hex) * 4
    if hex_bits > 52:
        frac_bits >>= (hex_bits - 52)
    elif hex_bits < 52:
        frac_bits <<= (52 - hex_bits)
    
    mantissa = frac_bits & 0xFFFFFFFFFFFFF
    
    if is_denormal:
        biased_exp = 0
    else:
        biased_exp = val.exponent + 1023
        
        if biased_exp >= 2047:
            return (val.sign << 63) | 0x7FF0000000000000
        if biased_exp <= 0:
            if biased_exp < -52:
                return val.sign << 63
            shift = 1 - biased_exp
            mantissa = ((1 << 52) | mantissa) >> shift
            biased_exp = 0
    
    return (val.sign << 63) | (biased_exp << 52) | mantissa


def parse_test_line(line: str, line_number: int) -> Optional[TestCase]:
    """Parse a single test line from an .fptest file."""
    line = line.strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('--') or line.startswith('#'):
        return None
    
    # Skip headers like "Floating point tests: ..."
    if line.startswith('Floating point tests') or line.startswith('Copyright'):
        return None
    
    # Parse precision and operation
    # Format: b32+ or b64- or b32* etc.
    match = re.match(r'^(b32|b64)(\+|-|\*|/|\*\+|V|%)\s+(.*)$', line)
    if not match:
        return None
    
    precision_str, op_str, rest = match.groups()
    
    try:
        precision = Precision(precision_str)
    except ValueError:
        return None
    
    op_map = {
        '+': Operation.ADD,
        '-': Operation.SUBTRACT,
        '*': Operation.MULTIPLY,
        '/': Operation.DIVIDE,
        '*+': Operation.FMA,
        'V': Operation.SQRT,
        '%': Operation.REM,
    }
    
    if op_str not in op_map:
        return None
    operation = op_map[op_str]
    
    # Parse rounding mode
    rounding_match = re.match(r'^(=0|=\^|>|<|0)\s+(.*)$', rest)
    if not rounding_match:
        return None
    
    rounding_str, rest = rounding_match.groups()
    
    rounding_map = {
        '=0': RoundingMode.NEAREST_EVEN,
        '=^': RoundingMode.NEAREST_AWAY,
        '>': RoundingMode.TOWARD_POSITIVE,
        '<': RoundingMode.TOWARD_NEGATIVE,
        '0': RoundingMode.TOWARD_ZERO,
    }
    rounding = rounding_map.get(rounding_str)
    if not rounding:
        return None
    
    # Check for exception flags before operands
    exception_flags = ""
    exc_match = re.match(r'^([iI])\s+(.*)$', rest)
    if exc_match:
        exception_flags = exc_match.group(1)
        rest = exc_match.group(2)
    
    # Split by -> to get operands and result
    if ' -> ' not in rest:
        return None
    
    operands_str, result_str = rest.split(' -> ', 1)
    
    # Parse result and result flags
    result_parts = result_str.split()
    if not result_parts:
        return None
    
    result_value_str = result_parts[0]
    result_flags = ' '.join(result_parts[1:]) if len(result_parts) > 1 else ''
    
    # Parse operands (space-separated)
    operand_strs = operands_str.split()
    if not operand_strs:
        return None
    
    try:
        operand1 = parse_fp_value(operand_strs[0])
        operand2 = parse_fp_value(operand_strs[1]) if len(operand_strs) > 1 else None
        operand3 = parse_fp_value(operand_strs[2]) if len(operand_strs) > 2 else None
        result = parse_fp_value(result_value_str)
    except ValueError as e:
        # Skip malformed values
        return None
    
    return TestCase(
        precision=precision,
        operation=operation,
        rounding=rounding,
        operand1=operand1,
        operand2=operand2,
        operand3=operand3,
        result=result,
        exception_flags=exception_flags + result_flags,
        line_number=line_number,
        raw_line=line,
    )


def parse_fptest_file(filepath: str) -> list[TestCase]:
    """Parse all test cases from an .fptest file."""
    tests = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            test = parse_test_line(line, line_num)
            if test:
                tests.append(test)
    
    return tests


def generate_noir_test_name(test: TestCase, index: int) -> str:
    """Generate a Noir test function name for a test case."""
    precision = "f32" if test.precision == Precision.BINARY32 else "f64"
    op_names = {
        Operation.ADD: "add",
        Operation.SUBTRACT: "sub",
        Operation.MULTIPLY: "mul",
        Operation.DIVIDE: "div",
        Operation.FMA: "fma",
        Operation.SQRT: "sqrt",
        Operation.REM: "rem",
    }
    op = op_names.get(test.operation, "op")
    return f"test_{precision}_{op}_{index}"


def generate_noir_test(test: TestCase, index: int, add_debug: bool = False) -> Optional[str]:
    """Generate Noir test code for a single test case."""
    
    # Only support add and subtract for now (what's implemented)
    if test.operation not in (Operation.ADD, Operation.SUBTRACT):
        return None
    
    # Only support round-to-nearest-even for now
    if test.rounding != RoundingMode.NEAREST_EVEN:
        return None
    
    # Need two operands for add/subtract
    if test.operand2 is None:
        return None
    
    # Skip tests where result is # (generic/don't care)
    if test.result.is_nan and test.result.significand == "" and not test.result.is_snan:
        # Could be a don't-care result, check if it's the # symbol
        pass
    
    is_float32 = test.precision == Precision.BINARY32
    
    if is_float32:
        bits1 = fp_value_to_bits32(test.operand1)
        bits2 = fp_value_to_bits32(test.operand2)
        expected = fp_value_to_bits32(test.result)
        from_bits_fn = "float32_from_bits"
        to_bits_fn = "float32_to_bits"
        op_fn = "add_float32" if test.operation == Operation.ADD else "sub_float32"
        bits_type = "u32"
    else:
        bits1 = fp_value_to_bits64(test.operand1)
        bits2 = fp_value_to_bits64(test.operand2)
        expected = fp_value_to_bits64(test.result)
        from_bits_fn = "float64_from_bits"
        to_bits_fn = "float64_to_bits"
        op_fn = "add_float64" if test.operation == Operation.ADD else "sub_float64"
        bits_type = "u64"
    
    test_name = generate_noir_test_name(test, index)
    
    # For subtraction, we negate the second operand and use addition
    if test.operation == Operation.SUBTRACT:
        # a - b = a + (-b), so flip sign of operand2
        if is_float32:
            bits2 ^= 0x80000000  # Flip sign bit
        else:
            bits2 ^= 0x8000000000000000
        op_fn = "add_float32" if is_float32 else "add_float64"
    
    # Format the bits as hex with proper prefix
    if is_float32:
        bits1_str = f"0x{bits1:08X}"
        bits2_str = f"0x{bits2:08X}"
        expected_str = f"0x{expected:08X}"
    else:
        bits1_str = f"0x{bits1:016X}"
        bits2_str = f"0x{bits2:016X}"
        expected_str = f"0x{expected:016X}"
    
    # Handle NaN results specially - just check it's NaN
    if test.result.is_nan:
        if is_float32:
            return f"""#[test]
fn {test_name}() {{
    // {test.raw_line}
    let a = {from_bits_fn}({bits1_str});
    let b = {from_bits_fn}({bits2_str});
    let result = {op_fn}(a, b);
    assert(float32_is_nan(result));
}}
"""
        else:
            return f"""#[test]
fn {test_name}() {{
    // {test.raw_line}
    let a = {from_bits_fn}({bits1_str});
    let b = {from_bits_fn}({bits2_str});
    let result = {op_fn}(a, b);
    assert(float64_is_nan(result));
}}
"""
    
    if add_debug:
        # Add println statements for debugging
        return f"""#[test]
fn {test_name}() {{
    // {test.raw_line}
    let a = {from_bits_fn}({bits1_str});
    let b = {from_bits_fn}({bits2_str});
    let result = {op_fn}(a, b);
    let result_bits = {to_bits_fn}(result);
    println(f"a: {{{bits1_str}}} b: {{{bits2_str}}} result: {{result_bits}} expected: {expected_str}");
    assert(result_bits == {expected_str});
}}
"""
    
    return f"""#[test]
fn {test_name}() {{
    // {test.raw_line}
    let a = {from_bits_fn}({bits1_str});
    let b = {from_bits_fn}({bits2_str});
    let result = {op_fn}(a, b);
    let result_bits = {to_bits_fn}(result);
    assert(result_bits == {expected_str});
}}
"""


def generate_noir_file(tests: list[TestCase], output_path: str, source_files: list[str], add_debug: bool = False):
    """Generate a complete Noir test file from test cases."""
    
    # Header
    header = f"""// Auto-generated IEEE 754 test cases
// Generated from: {', '.join(source_files)}
// Test suite source: https://github.com/sergev/ieee754-test-suite

use crate::float::{{
    IEEE754Float32, IEEE754Float64,
    float32_from_bits, float32_to_bits, float32_is_nan,
    float64_from_bits, float64_to_bits, float64_is_nan,
    add_float32, add_float64,
}};

"""
    
    # Generate tests
    test_code = []
    test_index = 0
    
    for test in tests:
        code = generate_noir_test(test, test_index, add_debug)
        if code:
            test_code.append(code)
            test_index += 1
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(header)
        f.write('\n'.join(test_code))
    
    print(f"Generated {len(test_code)} tests to {output_path}")
    return len(test_code)


def generate_noir_file_per_source(
    tests_by_file: dict[str, list[TestCase]], 
    output_dir: str, 
    add_debug: bool = False,
    chunk_size: int = 25
) -> dict[str, int]:
    """Generate separate Noir test files for each source file, chunked into groups."""
    
    results = {}
    module_names = []
    
    for source_file, tests in tests_by_file.items():
        # Create module name from filename
        base_name = os.path.splitext(source_file)[0]
        # Convert to valid Noir module name (lowercase, underscores)
        module_name = "test_" + re.sub(r'[^a-zA-Z0-9]', '_', base_name).lower()
        
        # Generate all test code first
        all_test_code = []
        test_index = 0
        
        for test in tests:
            code = generate_noir_test(test, test_index, add_debug)
            if code:
                all_test_code.append(code)
                test_index += 1
        
        if not all_test_code:
            continue
        
        # Create folder for this test source (only if we have tests)
        module_dir = os.path.join(output_dir, module_name)
        os.makedirs(module_dir, exist_ok=True)
        
        # Header template for each chunk file
        header_template = """// Auto-generated IEEE 754 test cases
// Generated from: {source_file} (chunk {chunk_num})
// Test suite source: https://github.com/sergev/ieee754-test-suite

use crate::float::{{
    IEEE754Float32, IEEE754Float64,
    float32_from_bits, float32_to_bits, float32_is_nan,
    float64_from_bits, float64_to_bits, float64_is_nan,
    add_float32, add_float64,
}};

"""
        
        # Chunk the tests
        chunks = [all_test_code[i:i + chunk_size] for i in range(0, len(all_test_code), chunk_size)]
        chunk_names = []
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_name = f"chunk_{chunk_idx:04d}"
            chunk_names.append(chunk_name)
            chunk_path = os.path.join(module_dir, f"{chunk_name}.nr")
            
            header = header_template.format(source_file=source_file, chunk_num=chunk_idx)
            
            with open(chunk_path, 'w') as f:
                f.write(header)
                f.write('\n'.join(chunk))
        
        # Generate mod.nr for this module folder
        module_mod_path = os.path.join(module_dir, "mod.nr")
        with open(module_mod_path, 'w') as f:
            f.write(f"// Auto-generated module index for {source_file}\n")
            f.write(f"// Contains {len(all_test_code)} tests in {len(chunks)} chunks of {chunk_size}\n\n")
            for chunk_name in chunk_names:
                f.write(f"mod {chunk_name};\n")
        
        print(f"Generated {len(all_test_code)} tests in {len(chunks)} chunks to {module_dir}/")
        results[source_file] = len(all_test_code)
        module_names.append(module_name)
    
    # Generate top-level module index file
    index_path = os.path.join(output_dir, "mod.nr")
    with open(index_path, 'w') as f:
        f.write("// Auto-generated module index for IEEE 754 tests\n")
        f.write("// Each module corresponds to a source .fptest file\n\n")
        for module_name in sorted(module_names):
            f.write(f"mod {module_name};\n")
    
    print(f"\nGenerated module index at {index_path}")
    return results


def get_cache_dir() -> Path:
    """Get the cache directory path, creating it if needed."""
    # Cache directory is relative to the project root (parent of scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cache_dir = project_root / DEFAULT_CACHE_DIR
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def download_test_file(filename: str, cache_dir: Path) -> Path:
    """Download a test file from the IEEE 754 test suite repository."""
    cache_path = cache_dir / filename
    
    if cache_path.exists():
        print(f"Using cached: {filename}")
        return cache_path
    
    url = f"{TEST_SUITE_BASE_URL}/{filename}"
    print(f"Downloading: {filename}...")
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
            cache_path.write_bytes(content)
            print(f"  Downloaded {len(content):,} bytes")
            return cache_path
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to download {filename}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {filename}: {e.reason}") from e


def list_available_files():
    """Print list of available test files."""
    print("Available IEEE 754 test files:")
    for f in AVAILABLE_TEST_FILES:
        print(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Noir tests from IEEE 754 test suite files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Download and use Add-Shift.fptest (default)
  %(prog)s --files Add-Shift.fptest     # Use specific file(s)
  %(prog)s --all                        # Use all available test files
  %(prog)s --all --split                # Generate chunked test folders per source
  %(prog)s --all --split --chunk-size 50  # Use 50 tests per chunk file
  %(prog)s --list                       # List available test files
  %(prog)s --local test.fptest          # Use a local file instead of downloading
        """
    )
    parser.add_argument(
        '--files',
        nargs='+',
        metavar='FILE',
        help='Test files to download and use (from IEEE 754 test suite)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download and use all available test files'
    )
    parser.add_argument(
        '--local',
        nargs='+',
        metavar='PATH',
        help='Use local .fptest file(s) instead of downloading'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available test files and exit'
    )
    parser.add_argument(
        '--output', '-o',
        default='ieee754_tests.nr',
        help='Output Noir file (default: ieee754_tests.nr)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for split test files (use with --split)'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Generate separate test files for each source file'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=25,
        help='Number of tests per chunk file when using --split (default: 25)'
    )
    parser.add_argument(
        '--operation',
        choices=['add', 'sub', 'mul', 'div', 'all'],
        default='add',
        help='Filter by operation type (default: add)'
    )
    parser.add_argument(
        '--precision',
        choices=['f32', 'f64', 'all'],
        default='all',
        help='Filter by precision (default: all)'
    )
    parser.add_argument(
        '--max-tests',
        type=int,
        default=None,
        help='Maximum number of tests to generate (per file when using --split)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Add println statements for debugging'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the download cache before running'
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_available_files()
        return
    
    # Get cache directory
    cache_dir = get_cache_dir()
    
    # Handle --clear-cache
    if args.clear_cache:
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared cache: {cache_dir}")
        cache_dir.mkdir(exist_ok=True)
    
    # Collect test files
    fptest_files = []
    
    if args.local:
        # Use local files
        for path in args.local:
            if os.path.isfile(path):
                fptest_files.append(path)
            elif os.path.isdir(path):
                for f in sorted(os.listdir(path)):
                    if f.endswith('.fptest'):
                        fptest_files.append(os.path.join(path, f))
            else:
                parser.error(f'Local path does not exist: {path}')
    elif args.all:
        # Download all available files
        for filename in AVAILABLE_TEST_FILES:
            try:
                path = download_test_file(filename, cache_dir)
                fptest_files.append(str(path))
            except RuntimeError as e:
                print(f"Warning: {e}")
    elif args.files:
        # Download specified files
        for filename in args.files:
            # Check if it's in the available list
            if filename not in AVAILABLE_TEST_FILES:
                print(f"Warning: '{filename}' not in known test files. Trying anyway...")
            try:
                path = download_test_file(filename, cache_dir)
                fptest_files.append(str(path))
            except RuntimeError as e:
                print(f"Warning: {e}")
    else:
        # Default: use Add-Shift.fptest
        try:
            path = download_test_file("Add-Shift.fptest", cache_dir)
            fptest_files.append(str(path))
        except RuntimeError as e:
            parser.error(str(e))
    
    if not fptest_files:
        parser.error('No .fptest files found')
    
    # Define filter functions
    def filter_tests(tests: list[TestCase]) -> list[TestCase]:
        result = tests
        
        # Filter by operation
        if args.operation != 'all':
            op_map = {
                'add': Operation.ADD,
                'sub': Operation.SUBTRACT,
                'mul': Operation.MULTIPLY,
                'div': Operation.DIVIDE,
            }
            target_op = op_map[args.operation]
            result = [t for t in result if t.operation == target_op]
        
        # Filter by precision
        if args.precision != 'all':
            target_precision = Precision.BINARY32 if args.precision == 'f32' else Precision.BINARY64
            result = [t for t in result if t.precision == target_precision]
        
        # Apply max tests limit
        if args.max_tests and len(result) > args.max_tests:
            result = result[:args.max_tests]
        
        return result
    
    if args.split:
        # Generate separate files per source
        output_dir = args.output_dir or "src/ieee754_tests"
        os.makedirs(output_dir, exist_ok=True)
        
        tests_by_file = {}
        total_parsed = 0
        
        for filepath in fptest_files:
            source_name = os.path.basename(filepath)
            print(f"Parsing {source_name}...")
            tests = parse_fptest_file(filepath)
            total_parsed += len(tests)
            filtered = filter_tests(tests)
            if filtered:
                tests_by_file[source_name] = filtered
                print(f"  {len(tests)} parsed, {len(filtered)} after filtering")
        
        print(f"\nParsed {total_parsed} total test cases")
        
        results = generate_noir_file_per_source(
            tests_by_file, 
            output_dir, 
            add_debug=args.debug,
            chunk_size=args.chunk_size
        )
        
        total_generated = sum(results.values())
        print(f"\nTotal: {total_generated} tests generated across {len(results)} files")
        
    else:
        # Parse all test files into one list
        all_tests = []
        for filepath in fptest_files:
            print(f"Parsing {os.path.basename(filepath)}...")
            tests = parse_fptest_file(filepath)
            all_tests.extend(tests)
        
        print(f"Parsed {len(all_tests)} total test cases")
        
        all_tests = filter_tests(all_tests)
        print(f"After filtering: {len(all_tests)} tests")
        
        # Generate Noir file
        generate_noir_file(
            all_tests,
            args.output,
            [os.path.basename(f) for f in fptest_files],
            add_debug=args.debug
        )


if __name__ == '__main__':
    main()
