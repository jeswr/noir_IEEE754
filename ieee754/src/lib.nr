pub mod types;
pub mod utils;
pub mod float32;
pub mod float64;
pub mod float;

// Re-export public API
pub use crate::float::{
    add_float32, add_float64, div_float32, div_float64, float32_from_bits, float32_infinity,
    float32_is_infinity, float32_is_nan, float32_is_zero, float32_nan, float32_to_bits,
    float32_zero, float64_from_bits, float64_infinity, float64_is_infinity, float64_is_nan,
    float64_is_zero, float64_nan, float64_to_bits, float64_zero, IEEE754Float32, IEEE754Float64,
    mul_float32, mul_float64, sub_float32, sub_float64,
};
