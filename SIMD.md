## Supported Intel CPU

| `__archspec` | Intel architecture | Launch | SIMD highlights | Supported? |

|---------------|--------------------|--------|-----------------|

| `haswell` | Xeon E5/E7 v3 | 2014–2015 | AVX2, FMA | Not Supported |

| `broadwell` | Xeon v4 | 2016 | AVX2 | Not Supported |

| `skylake_avx512` | Xeon Scalable Gen 1 | 2017 | First AVX-512 (F, DQ, CD, BW, VL) | Not Supported |

| `cascadelake` | Xeon Scalable Gen 2 | 2019 | AVX-512 + VNNI | Not Supported |

| `icelake` | Xeon Scalable Gen 3 | 2021 | More AVX-512 improvements (VBMI, VBMI2, BITALG, VPOPCNTDQ, GFNI, VAES, VPCLMULQDQ) | Supported |

| `sapphirerapids` | Xeon Scalable Gen 4 | 2023 | AVX-512, AMX, BF16, FP16, VNNI, etc. | Supported |

## Supported AMD CPU

| `__archspec` | AMD architecture | Launch | SIMD highlights | Supported? |

|---------------|------------------|--------|-----------------|

| `znver3` | EPYC Milan / Ryzen 5000 (Zen 3) | 2020–2021 | AVX2, FMA, BMI1/2, VAES, VPCLMULQDQ | Not Supported |

| `znver4` | EPYC Genoa/Bergamo / Ryzen 7000 (Zen 4) | 2022 | AVX-512 (F, DQ, CD, BW, VL, IFMA, VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, BF16), GFNI, VAES, VPCLMULQDQ | Supported |

| `znver5` | EPYC Turin / Ryzen AI 300 (Zen 5) | 2024 | Zen 4 ISA plus AVX-512 FP16 and additional microarchitectural improvements | Supported |
