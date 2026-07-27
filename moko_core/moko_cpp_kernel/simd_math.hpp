#ifndef MOKO_SIMD_MATH_HPP
#define MOKO_SIMD_MATH_HPP

#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace moko {

// Precomputed QEV Lookup Table for extremely fast byte-to-dot-product operations
// We map the 2-bit values back to their actual QEV logic
static const int8_t QEV_VALS[4] = {-1, 0, 1, 2};
static int8_t QEV_LUT[256][256];
static bool lut_initialized = false;

inline void init_qev_lut() {
    if (lut_initialized) return;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            int8_t v1_0 = QEV_VALS[(i >> 6) & 3];
            int8_t v1_1 = QEV_VALS[(i >> 4) & 3];
            int8_t v1_2 = QEV_VALS[(i >> 2) & 3];
            int8_t v1_3 = QEV_VALS[i & 3];

            int8_t v2_0 = QEV_VALS[(j >> 6) & 3];
            int8_t v2_1 = QEV_VALS[(j >> 4) & 3];
            int8_t v2_2 = QEV_VALS[(j >> 2) & 3];
            int8_t v2_3 = QEV_VALS[j & 3];

            QEV_LUT[i][j] = (v1_0 * v2_0) + (v1_1 * v2_1) + (v1_2 * v2_2) + (v1_3 * v2_3);
        }
    }
    lut_initialized = true;
}

// Compute dot product of two 192-byte arrays using LUT
// Length is exactly 192
inline int compute_qev_dot_product_192(const uint8_t* __restrict__ a, const uint8_t* __restrict__ b) {
    int score = 0;
    // Loop unrolling for better pipeline prediction
    for (int i = 0; i < 192; i+=4) {
        score += QEV_LUT[a[i]][b[i]];
        score += QEV_LUT[a[i+1]][b[i+1]];
        score += QEV_LUT[a[i+2]][b[i+2]];
        score += QEV_LUT[a[i+3]][b[i+3]];
    }
    return score;
}

} // namespace moko

#endif // MOKO_SIMD_MATH_HPP
