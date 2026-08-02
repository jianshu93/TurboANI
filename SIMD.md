## Supported Intel CPUs

| `__archspec` | Intel architecture | Launch | SIMD highlights | Supported? |
|---|---|---:|---|---|
| `haswell` | Xeon E5/E7 v3 | 2014–2015 | AVX2, FMA | No |
| `broadwell` | Xeon v4 | 2016 | AVX2 | No |
| `skylake_avx512` | Xeon Scalable Gen 1 | 2017 | First AVX-512 generation (F, DQ, CD, BW, VL) | No |
| `cascadelake` | Xeon Scalable Gen 2 | 2019 | AVX-512 + VNNI | No |
| `icelake` | Xeon Scalable Gen 3 | 2021 | AVX-512 improvements (IFMA, VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, GFNI, VAES, VPCLMULQDQ) | Yes |
| `sapphirerapids` | Xeon Scalable Gen 4 | 2023 | AVX-512, AMX, BF16, FP16, VNNI, etc. | Yes |

## Supported AMD CPUs

| `__archspec` | AMD architecture | Launch | SIMD highlights | Supported? |
|---|---|---:|---|---|
| `znver3` | EPYC Milan / Ryzen 5000 (Zen 3) | 2020–2021 | AVX2, FMA, BMI1/2, VAES, VPCLMULQDQ | No |
| `znver4` | EPYC Genoa/Bergamo / Ryzen 7000 (Zen 4) | 2022 | AVX-512 (F, DQ, CD, BW, VL, IFMA, VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, BF16), GFNI, VAES, VPCLMULQDQ | Yes |
| `znver5` | EPYC Turin / Ryzen AI 300 (Zen 5) | 2024 | Zen 4 ISA plus AVX-512 FP16 and additional microarchitectural improvements | Yes |
