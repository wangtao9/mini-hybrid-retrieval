/// Scalar BM25 scoring: simple dot product of idf and tf_norm vectors.
pub fn bm25_score_scalar(idf: &[f32], tf_norm: &[f32]) -> f32 {
    idf.iter().zip(tf_norm).map(|(i, t)| i * t).sum()
}

/// Runtime AVX-512 detection.
#[cfg(target_arch = "x86_64")]
fn is_avx512_available() -> bool {
    use std::arch::x86_64::__cpuid;
    unsafe {
        let leaf7 = __cpuid(7);
        // EBX bit 16 = AVX-512 F
        (leaf7.ebx & (1 << 16)) != 0
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
fn is_avx512_available() -> bool {
    false
}

/// AVX-512 BM25 scoring: processes 16 f32 elements at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn bm25_score_avx512(idf: &[f32], tf_norm: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_movehdup_ps,
        _mm256_movelh_ps, _mm512_add_ps, _mm512_extractf32x8_ps, _mm512_loadu_ps, _mm512_mul_ps,
        _mm512_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_movehl_ps,
    };

    let len = idf.len().min(tf_norm.len());
    let mut sum = _mm512_setzero_ps();
    let mut i = 0;

    // Process 16 f32 at a time
    while i + 16 <= len {
        let a = _mm512_loadu_ps(idf.as_ptr().add(i));
        let b = _mm512_loadu_ps(tf_norm.as_ptr().add(i));
        sum = _mm512_add_ps(sum, _mm512_mul_ps(a, b));
        i += 16;
    }

    // Horizontal sum of the 512-bit register
    // Extract two 256-bit halves from the 512-bit register and add them
    let hi = _mm512_extractf32x8_ps::<1>(sum);
    let lo = _mm512_extractf32x8_ps::<0>(sum);
    let v256 = _mm256_add_ps(lo, hi);

    // _mm256_movehdup_ps + add: pairwise sum of adjacent elements
    let shuf = _mm256_movehdup_ps(v256);
    let sums = _mm256_add_ps(v256, shuf);

    // Extract high 128-bit lane and add to low 128-bit lane
    let hi128 = _mm256_extractf128_ps::<1>(sums);
    let lo128 = _mm256_castps256_ps128(sums);
    let v128 = _mm_add_ps(lo128, hi128);

    // _mm_movehl_ps + add: combine remaining elements
    let hl = _mm_movehl_ps(v128, v128);
    let result = _mm_add_ps(v128, hl);

    // Extract the final scalar and add remainder
    _mm_cvtss_f32(result)
        + idf[i..len]
            .iter()
            .zip(&tf_norm[i..len])
            .map(|(a, b)| a * b)
            .sum::<f32>()
}

/// Runtime-dispatched BM25 batch scoring.
/// On x86_64 with AVX-512 available, uses the SIMD path; otherwise falls back to scalar.
pub fn bm25_score_batch(idf: &[f32], tf_norm: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx512_available() {
            unsafe { bm25_score_avx512(idf, tf_norm) }
        } else {
            bm25_score_scalar(idf, tf_norm)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        bm25_score_scalar(idf, tf_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_single_term() {
        let idf = [1.5f32];
        let tf_norm = [2.0f32];
        let result = bm25_score_scalar(&idf, &tf_norm);
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_multi_term() {
        let idf = [1.0f32, 2.0f32, 0.5f32];
        let tf_norm = [3.0f32, 1.5f32, 4.0f32];
        let result = bm25_score_scalar(&idf, &tf_norm);
        assert!((result - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_empty() {
        let idf: [f32; 0] = [];
        let tf_norm: [f32; 0] = [];
        let result = bm25_score_scalar(&idf, &tf_norm);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_matches_scalar() {
        let idf: Vec<f32> = (0..18).map(|i| (i + 1) as f32 * 0.1).collect();
        let tf_norm: Vec<f32> = (0..18).map(|i| (i + 1) as f32 * 0.2).collect();
        let expected = bm25_score_scalar(&idf, &tf_norm);
        let result = bm25_score_batch(&idf, &tf_norm);
        assert!((result - expected).abs() < 1e-4);
    }

    #[test]
    fn test_batch_small_input() {
        let idf = [0.5f32, 1.0f32, 2.0f32];
        let tf_norm = [1.0f32, 0.5f32, 0.25f32];
        let expected = bm25_score_scalar(&idf, &tf_norm);
        let result = bm25_score_batch(&idf, &tf_norm);
        assert!((result - expected).abs() < 1e-4);
    }

    #[test]
    fn test_batch_exact_16() {
        let idf: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.1).collect();
        let tf_norm: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.2).collect();
        let expected = bm25_score_scalar(&idf, &tf_norm);
        let result = bm25_score_batch(&idf, &tf_norm);
        assert!((result - expected).abs() < 1e-4);
    }
}