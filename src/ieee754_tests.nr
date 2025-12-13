// Auto-generated IEEE 754 test cases
// Generated from: Add-Shift.fptest
// Test suite source: https://github.com/sergev/ieee754-test-suite

use crate::float::{
    IEEE754Float32, IEEE754Float64,
    float32_from_bits, float32_to_bits, float32_is_nan,
    float64_from_bits, float64_to_bits, float64_is_nan,
    add_float32, add_float64,
};

#[test]
fn test_f32_add_0() {
    // b32+ =0 +1.01FD72P-118 +1.6C40BEP-91 -> +1.6C40BEP-91 x
    let a = float32_from_bits(0x0480FEB9);
    let b = float32_from_bits(0x1236205F);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x1236205F);
}

#[test]
fn test_f32_add_1() {
    // b32+ =0 +1.77E5B2P-85 +1.3A1492P-59 -> +1.3A1492P-59 x
    let a = float32_from_bits(0x153BF2D9);
    let b = float32_from_bits(0x221D0A49);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x221D0A49);
}

#[test]
fn test_f32_add_2() {
    // b32+ =0 +1.202CBCP-53 +1.0E6FB8P-28 -> +1.0E6FB8P-28 x
    let a = float32_from_bits(0x2510165E);
    let b = float32_from_bits(0x318737DC);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x318737DC);
}

#[test]
fn test_f32_add_3() {
    // b32+ =0 +1.15D4FBP-117 +1.5C038CP-93 -> +1.5C038DP-93 x
    let a = float32_from_bits(0x050AEA7D);
    let b = float32_from_bits(0x112E01C6);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x112E01C6);
}

#[test]
fn test_f32_add_4() {
    // b32+ =0 +1.0BBD3BP-116 +1.29D75FP-93 -> +1.29D760P-93 x
    let a = float32_from_bits(0x0585DE9D);
    let b = float32_from_bits(0x1114EBAF);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x1114EBB0);
}

#[test]
fn test_f32_add_5() {
    // b32+ =0 +1.01A57AP44 +1.77AB33P66 -> +1.77AB35P66 x
    let a = float32_from_bits(0x5580D2BD);
    let b = float32_from_bits(0x60BBD599);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x60BBD59A);
}

#[test]
fn test_f32_add_6() {
    // b32+ =0 +1.778DBAP45 +1.457F07P66 -> +1.457F0FP66 x
    let a = float32_from_bits(0x563BC6DD);
    let b = float32_from_bits(0x60A2BF83);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x60A2BF87);
}

#[test]
fn test_f32_add_7() {
    // b32+ =0 +1.1FD4C4P77 +1.195A2DP97 -> +1.195A37P97 x
    let a = float32_from_bits(0x660FEA62);
    let b = float32_from_bits(0x700CAD16);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x700CAD1B);
}

#[test]
fn test_f32_add_8() {
    // b32+ =0 +1.15BD03P108 +1.672E00P127 -> +1.672E13P127 x
    let a = float32_from_bits(0x758ADE81);
    let b = float32_from_bits(0x7F339700);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x7F339709);
}

#[test]
fn test_f32_add_9() {
    // b32+ =0 +1.594678P-113 +1.2EFA82P-95 -> +1.2EFAB8P-95 x
    let a = float32_from_bits(0x072CA33C);
    let b = float32_from_bits(0x10177D41);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x10177D5C);
}

#[test]
fn test_f32_add_10() {
    // b32+ =0 +1.4F2EB8P-81 +1.7CCE55P-64 -> +1.7CCEBDP-64 x
    let a = float32_from_bits(0x1727975C);
    let b = float32_from_bits(0x1FBE672A);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x1FBE675E);
}

#[test]
fn test_f32_add_11() {
    // b32+ =0 +1.77F5C1P-81 +1.50A97BP-65 -> +1.50AA73P-65 x
    let a = float32_from_bits(0x173BFAE0);
    let b = float32_from_bits(0x1F2854BD);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x1F285539);
}

#[test]
fn test_f32_add_12() {
    // b32+ =0 +0.00147EP-126 +1.1F51CDP-122 -> +1.1F5315P-122 x
    let a = float32_from_bits(0x00000A3F);
    let b = float32_from_bits(0x028FA8E6);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x028FA98A);
}

#[test]
fn test_f32_add_13() {
    // b32+ =0 +1.3EFCBEP-25 +1.6D65A1P-11 -> +1.6D689DP-11 x
    let a = float32_from_bits(0x331F7E5F);
    let b = float32_from_bits(0x3A36B2D0);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x3A36B44E);
}

#[test]
fn test_f32_add_14() {
    // b32+ =0 +1.3464FDP88 +1.3AB974P101 -> +1.3ABF17P101 x
    let a = float32_from_bits(0x6B9A327E);
    let b = float32_from_bits(0x721D5CBA);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x721D5F8B);
}

#[test]
fn test_f32_add_15() {
    // b32+ =0 +1.2A4D3DP41 +1.088D48P53 -> +1.0897EDP53 x
    let a = float32_from_bits(0x5415269E);
    let b = float32_from_bits(0x5A0446A4);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x5A044BF6);
}

#[test]
fn test_f32_add_16() {
    // b32+ =0 +1.529446P8 +1.5CE86EP19 -> +1.5D02C1P19 x
    let a = float32_from_bits(0x43A94A23);
    let b = float32_from_bits(0x492E7437);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x492E8160);
}

#[test]
fn test_f32_add_17() {
    // b32+ =0 +1.487C86P9 +1.2A3C42P19 -> +1.2A6E61P19 x
    let a = float32_from_bits(0x44243E43);
    let b = float32_from_bits(0x49151E21);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x49153730);
}

#[test]
fn test_f32_add_18() {
    // b32+ =0 +1.3E64C5P-87 +1.781015P-78 -> +1.786F47P-78 x
    let a = float32_from_bits(0x141F3262);
    let b = float32_from_bits(0x18BC080A);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x18BC37A3);
}

#[test]
fn test_f32_add_19() {
    // b32+ =0 +1.340D05P-126 +1.4623E9P-118 -> +1.46D7F6P-118 x
    let a = float32_from_bits(0x009A0682);
    let b = float32_from_bits(0x04A311F4);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x04A36BFB);
}

#[test]
fn test_f32_add_20() {
    // b32+ =0 +1.2A7544P-85 +1.13F7BDP-78 -> +1.154CA8P-78 x
    let a = float32_from_bits(0x15153AA2);
    let b = float32_from_bits(0x1889FBDE);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x188AA654);
}

#[test]
fn test_f32_add_21() {
    // b32+ =0 +1.523C4EP-86 +1.67D2E3P-80 -> +1.6B1BD4P-80 x
    let a = float32_from_bits(0x14A91E27);
    let b = float32_from_bits(0x17B3E971);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x17B58DEA);
}

#[test]
fn test_f32_add_22() {
    // b32+ =0 +1.1645C3P76 +1.2F1F64P81 -> +1.33D192P81 x
    let a = float32_from_bits(0x658B22E1);
    let b = float32_from_bits(0x68178FB2);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x6819E8C9);
}

#[test]
fn test_f32_add_23() {
    // b32+ =0 +1.0BAE03P108 +1.7D7337P112 -> +1.03170CP113 x
    let a = float32_from_bits(0x7585D701);
    let b = float32_from_bits(0x77BEB99B);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x78018B86);
}

#[test]
fn test_f32_add_24() {
    // b32+ =0 +1.5D4CBFP100 +1.4BDB89P103 -> +1.678521P103 x
    let a = float32_from_bits(0x71AEA65F);
    let b = float32_from_bits(0x7325EDC4);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x7333C290);
}

#[test]
fn test_f32_add_25() {
    // b32+ =0 +1.0513C9P100 +1.1FB6AFP102 -> +1.40FBA1P102 x
    let a = float32_from_bits(0x718289E4);
    let b = float32_from_bits(0x728FDB57);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x72A07DD0);
}

#[test]
fn test_f32_add_26() {
    // b32+ =0 +1.7B7C09P36 +1.6D8A83P37 -> +1.35A444P38 x
    let a = float32_from_bits(0x51BDBE04);
    let b = float32_from_bits(0x5236C541);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x529AD222);
}

#[test]
fn test_f32_add_27() {
    // b32+ =0 +1.716448P37 +1.3B5E57P37 -> +1.566150P38 x
    let a = float32_from_bits(0x5238B224);
    let b = float32_from_bits(0x521DAF2B);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x52AB30A8);
}

#[test]
fn test_f32_add_28() {
    // b32+ =0 +1.66CC88P-90 +1.09322AP-91 -> +1.15B2CEP-89 x
    let a = float32_from_bits(0x12B36644);
    let b = float32_from_bits(0x12049915);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x130AD967);
}

#[test]
fn test_f32_add_29() {
    // b32+ =0 +1.5CF4C7P-114 +1.5745FEP-116 -> +1.096323P-113 x
    let a = float32_from_bits(0x06AE7A63);
    let b = float32_from_bits(0x05ABA2FF);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x0704B191);
}

#[test]
fn test_f32_add_30() {
    // b32+ =0 +1.053BD1P-114 +1.2B2124P-117 -> +1.1A9FF6P-114 x
    let a = float32_from_bits(0x06829DE8);
    let b = float32_from_bits(0x05159092);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x068D4FFB);
}

#[test]
fn test_f32_add_31() {
    // b32+ =0 +1.7AA411P-90 +1.78F4F8P-94 -> +1.0519B0P-89 x
    let a = float32_from_bits(0x12BD5208);
    let b = float32_from_bits(0x10BC7A7C);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x13028CD8);
}

#[test]
fn test_f32_add_32() {
    // b32+ =0 +1.708C50P71 +1.46C8CBP66 -> +1.76C296P71 x
    let a = float32_from_bits(0x63384628);
    let b = float32_from_bits(0x60A36465);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x633B614B);
}

#[test]
fn test_f32_add_33() {
    // b32+ =0 +1.66F48FP72 +1.141C9FP66 -> +1.694501P72 x
    let a = float32_from_bits(0x63B37A47);
    let b = float32_from_bits(0x608A0E4F);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x63B4A280);
}

#[test]
fn test_f32_add_34() {
    // b32+ =0 +1.5C9CCFP8 +1.623073P1 -> +1.5E6130P8 x
    let a = float32_from_bits(0x43AE4E67);
    let b = float32_from_bits(0x40311839);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x43AF3098);
}

#[test]
fn test_f32_add_35() {
    // b32+ =0 +1.52850EP-55 +1.300446P-63 -> +1.533512P-55 x
    let a = float32_from_bits(0x24294287);
    let b = float32_from_bits(0x20180223);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x24299A89);
}

#[test]
fn test_f32_add_36() {
    // b32+ =0 +1.486D4EP9 +1.7DD81AP0 -> +1.48EC3AP9 x
    let a = float32_from_bits(0x442436A7);
    let b = float32_from_bits(0x3FBEEC0D);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x4424761D);
}

#[test]
fn test_f32_add_37() {
    // b32+ =0 +1.19CC0BP-95 +1.4C806BP-105 -> +1.19FF2BP-95 x
    let a = float32_from_bits(0x100CE605);
    let b = float32_from_bits(0x0B264035);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x100CFF95);
}

#[test]
fn test_f32_add_38() {
    // b32+ =0 +1.0FB44AP-62 +1.1A543FP-73 -> +1.0FC795P-62 x
    let a = float32_from_bits(0x2087DA25);
    let b = float32_from_bits(0x1B0D2A1F);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x2087E3CA);
}

#[test]
fn test_f32_add_39() {
    // b32+ =0 +1.37BB54P33 +1.6E6F65P21 -> +1.37CA3BP33 x
    let a = float32_from_bits(0x501BDDAA);
    let b = float32_from_bits(0x4A3737B2);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x501BE51D);
}

#[test]
fn test_f32_add_40() {
    // b32+ =0 +1.2DA393P-62 +1.3C4339P-75 -> +1.2DA975P-62 x
    let a = float32_from_bits(0x2096D1C9);
    let b = float32_from_bits(0x1A1E219C);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x2096D4BA);
}

#[test]
fn test_f32_add_41() {
    // b32+ =0 +1.238BD3P35 +1.09970CP21 -> +1.238DF9P35 x
    let a = float32_from_bits(0x5111C5E9);
    let b = float32_from_bits(0x4A04CB86);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x5111C6FC);
}

#[test]
fn test_f32_add_42() {
    // b32+ =0 +1.197412P99 +1.57EAE0P84 -> +1.1975C2P99 x
    let a = float32_from_bits(0x710CBA09);
    let b = float32_from_bits(0x69ABF570);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x710CBAE1);
}

#[test]
fn test_f32_add_43() {
    // b32+ =0 +1.0F5C52P100 +1.25BEB4P84 -> +1.0F5CF8P100 x
    let a = float32_from_bits(0x7187AE29);
    let b = float32_from_bits(0x6992DF5A);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x7187AE7C);
}

#[test]
fn test_f32_add_44() {
    // b32+ =0 +1.37E35CP67 +1.79D9DAP50 -> +1.37E3D9P67 x
    let a = float32_from_bits(0x611BF1AE);
    let b = float32_from_bits(0x58BCECED);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x611BF1EC);
}

#[test]
fn test_f32_add_45() {
    // b32+ =0 +1.2D4B9BP4 +1.472DAEP-14 -> +1.2D4BCDP4 x
    let a = float32_from_bits(0x4196A5CD);
    let b = float32_from_bits(0x38A396D7);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x4196A5E6);
}

#[test]
fn test_f32_add_46() {
    // b32+ =0 +1.2333DBP5 +1.150181P-14 -> +1.2333EEP5 x
    let a = float32_from_bits(0x421199ED);
    let b = float32_from_bits(0x388A80C0);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x421199F7);
}

#[test]
fn test_f32_add_47() {
    // b32+ =0 +1.191C1AP5 +1.62D555P-15 -> +1.191C28P5 x
    let a = float32_from_bits(0x420C8E0D);
    let b = float32_from_bits(0x38316AAA);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x420C8E14);
}

#[test]
fn test_f32_add_48() {
    // b32+ =0 +1.5CA58FP-25 +1.2A21D6P-46 -> +1.5CA594P-25 x
    let a = float32_from_bits(0x332E52C7);
    let b = float32_from_bits(0x289510EB);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x332E52CA);
}

#[test]
fn test_f32_add_49() {
    // b32+ =0 +1.052C99P-74 +1.7EBCFCP-96 -> +1.052C9DP-74 x
    let a = float32_from_bits(0x1A82964C);
    let b = float32_from_bits(0x0FBF5E7E);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x1A82964E);
}

#[test]
fn test_f32_add_50() {
    // b32+ =0 +1.560B56P-2 +1.4D654EP-25 -> +1.560B58P-2 x
    let a = float32_from_bits(0x3EAB05AB);
    let b = float32_from_bits(0x3326B2A7);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x3EAB05AC);
}

#[test]
fn test_f32_add_51() {
    // b32+ =0 +1.4C3395P95 +1.1AF921P71 -> +1.4C3396P95 x
    let a = float32_from_bits(0x6F2619CA);
    let b = float32_from_bits(0x630D7C90);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x6F2619CB);
}

#[test]
fn test_f32_add_52() {
    // b32+ =0 +1.421BD5P-1 +1.68CCF5P-26 -> +1.421BD5P-1 x
    let a = float32_from_bits(0x3F210DEA);
    let b = float32_from_bits(0x32B4667A);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x3F210DEA);
}

#[test]
fn test_f32_add_53() {
    // b32+ =0 +1.6A62DFP-33 +1.3CA81BP-59 -> +1.6A62DFP-33 x
    let a = float32_from_bits(0x2F35316F);
    let b = float32_from_bits(0x221E540D);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x2F35316F);
}

#[test]
fn test_f32_add_54() {
    // b32+ =0 +1.604B1EP32 +1.0AFBEFP5 -> +1.604B1EP32 x
    let a = float32_from_bits(0x4FB0258F);
    let b = float32_from_bits(0x42057DF7);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x4FB0258F);
}

#[test]
fn test_f32_add_55() {
    // b32+ =0 +1.56735DP-89 +1.580FC2P-60 -> +1.580FC2P-60 x
    let a = float32_from_bits(0x132B39AE);
    let b = float32_from_bits(0x21AC07E1);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x21AC07E1);
}

#[test]
fn test_f32_add_56() {
    // b32+ =0 +1.4BDB9DP68 +1.266396P4 -> +1.4BDB9DP68 x
    let a = float32_from_bits(0x61A5EDCE);
    let b = float32_from_bits(0x419331CB);
    let result = add_float32(a, b);
    let result_bits = float32_to_bits(result);
    assert(result_bits == 0x61A5EDCE);
}
