// ============================================================================
// IEEE 754 Compliant Floating Point Types
// ============================================================================

// IEEE 754 Single-Precision (32-bit) Float
// - 1 bit sign
// - 8 bits exponent (bias 127)
// - 23 bits mantissa (with implicit leading 1 for normalized numbers)
pub struct IEEE754Float32 {
    pub sign: u1, // 0 = positive, 1 = negative
    pub exponent: u8, // Biased exponent (0-255)
    pub mantissa: u32, // 23-bit mantissa stored in lower bits
}

// IEEE 754 Double-Precision (64-bit) Float
// - 1 bit sign
// - 11 bits exponent (bias 1023)
// - 52 bits mantissa (with implicit leading 1 for normalized numbers)
pub struct IEEE754Float64 {
    pub sign: u1, // 0 = positive, 1 = negative
    pub exponent: u16, // Biased exponent (0-2047), using u16 to hold 11 bits
    pub mantissa: u64, // 52-bit mantissa stored in lower bits
}

// Constants for IEEE 754 Float32
pub global FLOAT32_EXPONENT_MAX: u8 = 255;
pub global FLOAT32_IMPLICIT_BIT: u32 = 0x800000; // 1 << 23

// Constants for IEEE 754 Float64
pub global FLOAT64_EXPONENT_MAX: u16 = 2047;
pub global FLOAT64_IMPLICIT_BIT: u64 = 0x10000000000000; // 1 << 52
