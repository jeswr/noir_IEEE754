use ieee754::{float32_from_bits, float32_to_bits, div_float32};

#[test]
fn test_div_732() {
    // b32/ =0 -0.000020P-126 +1.666666P-48 -> -1.0E38E4P-97 x
    let a = float32_from_bits(0x80000010);  // tiny denormal
    let b = float32_from_bits(0x27B33333);  // normal
    let result = div_float32(a, b);
    let result_bits = float32_to_bits(result);
    
    let expected: u32 = 0x8EB6DB6E;
    
    println(f"a bits: 0x80000010");
    println(f"b bits: 0x27B33333");
    println(f"result bits: {result_bits}");
    println(f"expected: {expected}");
    
    // Decode
    let result_sign = result_bits >> 31;
    let result_exp = (result_bits >> 23) & 0xFF;
    let result_mant = result_bits & 0x7FFFFF;
    
    let expected_sign = expected >> 31;
    let expected_exp = (expected >> 23) & 0xFF;
    let expected_mant = expected & 0x7FFFFF;
    
    println(f"result: sign={result_sign}, exp={result_exp}, mant={result_mant}");
    println(f"expected: sign={expected_sign}, exp={expected_exp}, mant={expected_mant}");
    
    assert(result_bits == expected);
}
