//! Custom histogram implementation for tracking vector length distribution
//!
//! This module provides a simple, efficient array-based histogram optimized for
//! tracking RISC-V vector instruction lengths (typically 1-2048).

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::AddAssign;

/// Largest VLEN, in bytes, a bucket may be recorded for: the RVV maximum VLEN
/// of 65536 bits. It was 2048 (a 16Ki-bit VLEN) while VLEN was capped by a
/// `u16` of bits, which `record_n`'s assertion turned into a panic for any
/// legal VLEN above that -- reachable at e8/m8 from a 16Ki-bit VLEN upwards.
///
/// Nothing is allocated for the range: `counts` grows to the largest VL
/// actually seen, so this is purely the bound of what may be seen.
const VLENB_MAX: usize = 8192;

/// LMUL allows x8 on the VLENB, so the largest architectural VL is at e8/m8.
const MAX_VL: usize = VLENB_MAX * 8;

/// Lazy sparse histogram for tracking vector length distribution.
///
/// The counts Vec starts empty (zero allocation) and grows on first use.
/// This means RoiInfo::new() and Clone on untouched histograms are free,
/// which matters because the vast majority of functions in a trace never
/// execute a vector instruction.
#[derive(Clone)]
pub struct VectorLengthHistogram {
    /// Counts for each vector length (index = VL, value = count).
    /// Empty until the first record() call; length is max_seen_vl+1 after that.
    counts: Vec<u64>,

    /// Total number of samples recorded
    total_count: u64,

    /// Per-LMUL counts (vlmul encoding: 0=m1,1=m2,2=m4,3=m8,5=mf8,6=mf4,7=mf2; 4=reserved).
    /// Index is the raw vlmul field value (0..8). Slot 4 is unused.
    per_lmul_counts: [Vec<u64>; 8],

    /// Per-LMUL total sample counts.
    per_lmul_total: [u64; 8],
}

impl std::fmt::Debug for VectorLengthHistogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorLengthHistogram")
            // .field("counts", &self.counts)
            .field("total_count", &self.total_count)
            .field("per_lmul_total", &self.per_lmul_total)
            .finish()
    }
}

impl Default for VectorLengthHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for VectorLengthHistogram {
    fn eq(&self, other: &Self) -> bool {
        self.total_count == other.total_count
            && self.counts[..] == other.counts[..]
            && self.per_lmul_total == other.per_lmul_total
            && self.per_lmul_counts == other.per_lmul_counts
    }
}

impl VectorLengthHistogram {
    /// Create a new empty histogram. Allocates nothing until the first record() call.
    pub fn new() -> Self {
        Self {
            counts: Vec::new(),
            total_count: 0,
            per_lmul_counts: Default::default(),
            per_lmul_total: [0; 8],
        }
    }

    /// Record a single occurrence of a vector length
    ///
    /// # Arguments
    /// * `vl` - Vector length to record (must be in range 1..=MAX_VL)
    ///
    /// # Panics
    /// Panics if vl is 0 or greater than MAX_VL
    #[inline]
    pub fn record(&mut self, vl: u64, lmul: Option<u8>) {
        self.record_n(vl, lmul, 1);
    }

    /// Record `count` occurrences of the same `(vl, lmul)` in one step.
    ///
    /// Exists for the batched enrichment path (see `run_cache`): a
    /// straight-line run executes at a single `vtype`, so all of its vector
    /// instructions land in exactly one bucket and can be folded in with one
    /// resize + two adds instead of `count` separate `record` calls. The result
    /// is by construction identical to `count` × `record`.
    ///
    /// # Panics
    /// Panics if `vl` is greater than `MAX_VL`.
    #[inline]
    pub fn record_n(&mut self, vl: u64, lmul: Option<u8>, count: u64) {
        assert!(
            vl <= MAX_VL as u64,
            "Vector length {} greater than allowed of range [0, {}]",
            vl,
            MAX_VL
        );
        let idx = vl as usize;
        if idx >= self.counts.len() {
            self.counts.resize(idx + 1, 0);
        }
        self.counts[idx] += count;
        self.total_count += count;

        if let Some(l) = lmul {
            if (l as usize) < 8 {
                let li = l as usize;
                if idx >= self.per_lmul_counts[li].len() {
                    self.per_lmul_counts[li].resize(idx + 1, 0);
                }
                self.per_lmul_counts[li][idx] += count;
                self.per_lmul_total[li] += count;
            }
        }
    }

    /// Get the total number of samples recorded
    pub fn len(&self) -> u64 {
        self.total_count
    }

    /// Check if histogram is empty
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Calculate the mean (average) vector length
    pub fn mean(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let mut sum: u64 = 0;
        for (vl, &count) in self.counts.iter().enumerate().skip(1) {
            sum += (vl as u64) * count;
        }

        sum as f64 / self.total_count as f64
    }

    /// Calculate a percentile value
    ///
    /// # Arguments
    /// * `quantile` - Quantile to calculate (0.0 to 1.0, e.g., 0.95 for p95)
    ///
    /// # Returns
    /// The vector length at which `quantile * 100`% of samples are at or below
    pub fn value_at_quantile(&self, quantile: f64) -> u64 {
        if !(0.0..=1.0).contains(&quantile) {
            panic!("Quantile must be between 0.0 and 1.0");
        }

        if self.total_count == 0 {
            return 0;
        }

        let target_count = (self.total_count as f64 * quantile).ceil() as u64;
        let mut cumulative_count = 0u64;

        for (vl, &count) in self.counts.iter().enumerate().skip(1) {
            cumulative_count += count;
            if cumulative_count >= target_count {
                return vl as u64;
            }
        }

        self.counts.len().saturating_sub(1) as u64
    }

    /// Get the count for a specific vector length
    pub fn count_at(&self, vl: usize) -> u64 {
        if vl >= self.counts.len() {
            return 0;
        }
        self.counts[vl]
    }

    /// Merge another histogram into this one
    pub fn add(&mut self, other: &VectorLengthHistogram) {
        if other.counts.len() > self.counts.len() {
            self.counts.resize(other.counts.len(), 0);
        }
        for (vl, &count) in other.counts.iter().enumerate().skip(1) {
            self.counts[vl] += count;
        }
        self.total_count += other.total_count;

        for i in 0..8 {
            if other.per_lmul_counts[i].len() > self.per_lmul_counts[i].len() {
                self.per_lmul_counts[i].resize(other.per_lmul_counts[i].len(), 0);
            }
            for (vl, &count) in other.per_lmul_counts[i].iter().enumerate().skip(1) {
                self.per_lmul_counts[i][vl] += count;
            }
            self.per_lmul_total[i] += other.per_lmul_total[i];
        }
    }

    /// Get minimum recorded vector length
    pub fn min(&self) -> Option<u64> {
        if self.total_count == 0 {
            return None;
        }

        for (vl, &count) in self.counts.iter().enumerate().skip(1) {
            if count > 0 {
                return Some(vl as u64);
            }
        }
        None
    }

    /// Rebin the histogram into `n` log₂-spaced buckets over VL range [1, MAX_VL].
    ///
    /// Bucket `i` accumulates all counts whose VL satisfies:
    ///   2^(log2(MAX_VL) * i / n)  <=  vl  <  2^(log2(MAX_VL) * (i+1) / n)
    ///
    /// This gives fine-grained resolution at small VL values (scalar / short-strip
    /// operations) and coarser buckets at the high end where hardware-max fills are
    /// expected. VL = 1 always lands in bucket 0; VL = MAX_VL always lands in the
    /// last bucket.
    ///
    /// Returns a `Vec` of `n` counts (all zeros when the histogram is empty).
    pub fn to_buckets(&self, n: usize) -> Vec<u64> {
        let mut buckets = vec![0u64; n];
        if self.total_count == 0 || n == 0 {
            return buckets;
        }
        let max_log = (MAX_VL as f64).log2();
        for (vl, &count) in self.counts.iter().enumerate().skip(1) {
            if count == 0 {
                continue;
            }
            let log_vl = (vl as f64).log2().max(0.0);
            let idx = ((log_vl / max_log) * n as f64) as usize;
            buckets[idx.min(n - 1)] += count;
        }
        buckets
    }

    /// Return all non-zero (vl, count) pairs as a flat vec of [vl, count] pairs.
    ///
    /// This gives the full-resolution histogram without any rebinning.  The
    /// result is sparse: only VL values that were actually observed are included,
    /// so the payload stays small even though the internal array can be large.
    pub fn to_sparse(&self, lmul: Option<u8>) -> Vec<[u64; 2]> {
        let source = match lmul {
            Some(l) if (l as usize) < 8 => &self.per_lmul_counts[l as usize],
            _ => &self.counts,
        };
        source
            .iter()
            .enumerate()
            .skip(1) // index 0 is unused (VL starts at 1)
            .filter(|(_, &c)| c > 0)
            .map(|(vl, &c)| [vl as u64, c])
            .collect()
    }

    /// Get maximum recorded vector length
    pub fn max(&self) -> Option<u64> {
        if self.total_count == 0 {
            return None;
        }

        for (vl, &count) in self.counts.iter().enumerate().skip(1).rev() {
            if count > 0 {
                return Some(vl as u64);
            }
        }
        None
    }

    pub fn to_json(&self) -> Result<String, std::io::Error> {
        serde_json::to_string(&self).map_err(Into::into)
    }

    pub fn as_base64(&self) -> Result<String, std::io::Error> {
        let json = self.to_json()?;
        use base64::Engine as _;
        Ok(base64::engine::general_purpose::STANDARD.encode(json.as_bytes()))
    }
}

impl AddAssign<&VectorLengthHistogram> for VectorLengthHistogram {
    fn add_assign(&mut self, rhs: &VectorLengthHistogram) {
        self.add(rhs);
    }
}

// Custom Serialize/Deserialize implementation to avoid Default trait requirement on large arrays
impl Serialize for VectorLengthHistogram {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("VectorLengthHistogram", 1)?;
        state.serialize_field("total_count", &self.total_count)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for VectorLengthHistogram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct HistogramData {
            total_count: u64,
        }

        let data = HistogramData::deserialize(deserializer)?;
        Ok(VectorLengthHistogram {
            counts: Vec::new(),
            total_count: data.total_count,
            per_lmul_counts: Default::default(),
            per_lmul_total: [0; 8],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_histogram_is_empty() {
        let hist = VectorLengthHistogram::new();
        assert_eq!(hist.len(), 0);
        assert!(hist.is_empty());
    }

    #[test]
    fn test_record_single_value() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);
        assert_eq!(hist.len(), 1);
        assert!(!hist.is_empty());
        assert_eq!(hist.count_at(10), 1);
    }

    #[test]
    fn test_record_multiple_values() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);
        hist.record(10, None);
        hist.record(20, None);

        assert_eq!(hist.len(), 3);
        assert_eq!(hist.count_at(10), 2);
        assert_eq!(hist.count_at(20), 1);
    }

    #[test]
    fn test_record_out_of_range_zero() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(0, None);
    }

    #[test]
    #[should_panic(expected = "greater than allowed of range [0,")]
    fn test_record_out_of_range_too_high() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(MAX_VL as u64 + 1, None);
    }

    #[test]
    fn test_mean_empty_histogram() {
        let hist = VectorLengthHistogram::new();
        assert_eq!(hist.mean(), 0.0);
    }

    #[test]
    fn test_mean_single_value() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);
        assert_eq!(hist.mean(), 10.0);
    }

    #[test]
    fn test_mean_multiple_values() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);
        hist.record(20, None);
        hist.record(30, None);

        // Mean should be (10 + 20 + 30) / 3 = 20.0
        assert_eq!(hist.mean(), 20.0);
    }

    #[test]
    fn test_mean_with_repeated_values() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);
        hist.record(10, None);
        hist.record(20, None);

        // Mean should be (10 + 10 + 20) / 3 = 13.333...
        let mean = hist.mean();
        assert!((mean - 13.333333).abs() < 0.001);
    }

    #[test]
    fn test_percentile_empty_histogram() {
        let hist = VectorLengthHistogram::new();
        assert_eq!(hist.value_at_quantile(0.95), 0);
    }

    #[test]
    fn test_percentile_single_value() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);

        assert_eq!(hist.value_at_quantile(0.5), 10);
        assert_eq!(hist.value_at_quantile(0.95), 10);
        assert_eq!(hist.value_at_quantile(0.99), 10);
    }

    #[test]
    fn test_percentile_p95() {
        let mut hist = VectorLengthHistogram::new();

        // Add 100 samples: 95 at VL=10, 5 at VL=100
        for _ in 0..95 {
            hist.record(10, None);
        }
        for _ in 0..5 {
            hist.record(100, None);
        }

        // P95 should be at the boundary where 95% are at or below
        // With 100 samples, 95% = 95 samples, all of which are at VL=10
        assert_eq!(hist.value_at_quantile(0.95), 10);

        // P96 should include the first VL=100 sample
        assert_eq!(hist.value_at_quantile(0.96), 100);
    }

    #[test]
    fn test_percentile_p50_median() {
        let mut hist = VectorLengthHistogram::new();

        // Add values: 1, 2, 3, 4, 5
        for i in 1..=5 {
            hist.record(i, None);
        }

        // Median (p50) should be 3
        assert_eq!(hist.value_at_quantile(0.5), 3);
    }

    #[test]
    fn test_min_max_empty() {
        let hist = VectorLengthHistogram::new();
        assert_eq!(hist.min(), None);
        assert_eq!(hist.max(), None);
    }

    #[test]
    fn test_min_max() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);
        hist.record(50, None);
        hist.record(30, None);
        hist.record(100, None);

        assert_eq!(hist.min(), Some(10));
        assert_eq!(hist.max(), Some(100));
    }

    #[test]
    fn test_add_histograms() {
        let mut hist1 = VectorLengthHistogram::new();
        hist1.record(10, None);
        hist1.record(20, None);

        let mut hist2 = VectorLengthHistogram::new();
        hist2.record(10, None);
        hist2.record(30, None);

        hist1.add(&hist2);

        assert_eq!(hist1.len(), 4);
        assert_eq!(hist1.count_at(10), 2);
        assert_eq!(hist1.count_at(20), 1);
        assert_eq!(hist1.count_at(30), 1);
    }

    #[test]
    fn test_add_assign() {
        let mut hist1 = VectorLengthHistogram::new();
        hist1.record(10, None);

        let mut hist2 = VectorLengthHistogram::new();
        hist2.record(20, None);

        hist1 += &hist2;

        assert_eq!(hist1.len(), 2);
        assert_eq!(hist1.count_at(10), 1);
        assert_eq!(hist1.count_at(20), 1);
    }

    #[test]
    fn test_mean_consistency() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);
        hist.record(20, None);

        // Multiple calls should give same result
        let mean1 = hist.mean();
        let mean2 = hist.mean();
        assert_eq!(mean1, mean2);

        // Adding data should change mean
        hist.record(30, None);
        let mean3 = hist.mean();
        assert_ne!(mean1, mean3);
    }

    #[test]
    fn test_percentile_consistency() {
        let mut hist = VectorLengthHistogram::new();
        for _ in 0..100 {
            hist.record(10, None);
        }

        // Multiple calls should give same result
        let p95_1 = hist.value_at_quantile(0.95);
        let p95_2 = hist.value_at_quantile(0.95);
        assert_eq!(p95_1, p95_2);
    }

    #[test]
    fn test_boundary_values() {
        let mut hist = VectorLengthHistogram::new();

        // Test minimum valid value
        hist.record(1, None);
        assert_eq!(hist.count_at(1), 1);

        // Test maximum valid value
        hist.record(MAX_VL as u64, None);
        assert_eq!(hist.count_at(MAX_VL), 1);
    }

    #[test]
    fn test_realistic_workload() {
        let mut hist = VectorLengthHistogram::new();

        // Simulate a realistic vector workload
        // Most operations at VL=16 (common vector size)
        for _ in 0..1000 {
            hist.record(16, None);
        }

        // Some at VL=8
        for _ in 0..100 {
            hist.record(8, None);
        }

        // Few at VL=32
        for _ in 0..50 {
            hist.record(32, None);
        }

        // Occasional large VL
        for _ in 0..10 {
            hist.record(128, None);
        }

        assert_eq!(hist.len(), 1160);

        let mean = hist.mean();
        assert!(mean > 15.0 && mean < 18.0, "Mean should be close to 16");

        let p95 = hist.value_at_quantile(0.95);
        assert!(p95 <= 32, "P95 should be at or below 32");

        assert_eq!(hist.min(), Some(8));
        assert_eq!(hist.max(), Some(128));
    }

    #[test]
    fn test_serialization() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);
        hist.record(20, None);

        // Test serialization
        let serialized = serde_json::to_string(&hist).unwrap();
        assert!(serialized.contains("total_count"));

        // Test deserialization
        let deserialized: VectorLengthHistogram = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.total_count, 2);
    }

    #[test]
    fn test_percentile_edge_cases() {
        let mut hist = VectorLengthHistogram::new();

        // Add 10 samples
        for i in 1..=10 {
            hist.record(i, None);
        }

        // Test various percentiles
        assert_eq!(hist.value_at_quantile(0.0), 1); // Minimum
        assert_eq!(hist.value_at_quantile(0.1), 1); // 10th percentile (1st sample)
        assert_eq!(hist.value_at_quantile(0.5), 5); // 50th percentile (5th sample)
        assert_eq!(hist.value_at_quantile(0.9), 9); // 90th percentile (9th sample)
        assert_eq!(hist.value_at_quantile(1.0), 10); // Maximum
    }

    #[test]
    fn test_large_counts() {
        let mut hist = VectorLengthHistogram::new();

        // Record very large counts
        for _ in 0..1_000_000 {
            hist.record(16, None);
        }
        for _ in 0..500_000 {
            hist.record(32, None);
        }

        assert_eq!(hist.len(), 1_500_000);

        // Mean should be weighted toward 16
        let mean = hist.mean();
        assert!(mean > 20.0 && mean < 22.0);
    }

    #[test]
    fn test_histogram_merge_preserves_statistics() {
        let mut hist1 = VectorLengthHistogram::new();
        for _ in 0..50 {
            hist1.record(10, None);
        }

        let mut hist2 = VectorLengthHistogram::new();
        for _ in 0..50 {
            hist2.record(20, None);
        }

        hist1.add(&hist2);

        assert_eq!(hist1.len(), 100);
        assert_eq!(hist1.mean(), 15.0); // (10*50 + 20*50) / 100
        assert_eq!(hist1.value_at_quantile(0.5), 10); // First 50 samples at VL=10
        assert_eq!(hist1.value_at_quantile(0.51), 20); // Next 50 samples at VL=20
    }

    #[test]
    fn test_all_same_value() {
        let mut hist = VectorLengthHistogram::new();

        // All samples at same VL
        for _ in 0..100 {
            hist.record(42, None);
        }

        assert_eq!(hist.mean(), 42.0);
        assert_eq!(hist.min(), Some(42));
        assert_eq!(hist.max(), Some(42));
        assert_eq!(hist.value_at_quantile(0.5), 42);
        assert_eq!(hist.value_at_quantile(0.95), 42);
        assert_eq!(hist.value_at_quantile(0.99), 42);
    }

    #[test]
    fn test_sparse_distribution() {
        let mut hist = VectorLengthHistogram::new();

        // Very sparse data
        hist.record(1, None);
        hist.record(500, None);
        hist.record(1000, None);
        hist.record(2048, None);

        assert_eq!(hist.len(), 4);
        assert_eq!(hist.min(), Some(1));
        assert_eq!(hist.max(), Some(2048));

        // Mean should be (1 + 500 + 1000 + 2048) / 4 = 887.25
        let mean = hist.mean();
        assert!((mean - 887.25).abs() < 0.01);
    }

    #[test]
    fn test_count_at_nonexistent() {
        let mut hist = VectorLengthHistogram::new();
        hist.record(10, None);

        assert_eq!(hist.count_at(5), 0);
        assert_eq!(hist.count_at(15), 0);
        assert_eq!(hist.count_at(10), 1);
        assert_eq!(hist.count_at(3000), 0); // Out of range
    }

    #[test]
    fn test_percentile_with_duplicates() {
        let mut hist = VectorLengthHistogram::new();

        // Create a distribution with duplicates
        for _ in 0..90 {
            hist.record(10, None);
        }
        for _ in 0..10 {
            hist.record(100, None);
        }

        assert_eq!(hist.len(), 100);

        // P50, P75, P90 should all be at VL=10
        assert_eq!(hist.value_at_quantile(0.50), 10);
        assert_eq!(hist.value_at_quantile(0.75), 10);
        assert_eq!(hist.value_at_quantile(0.90), 10);

        // P91 and above should be at VL=100
        assert_eq!(hist.value_at_quantile(0.91), 100);
        assert_eq!(hist.value_at_quantile(0.95), 100);
        assert_eq!(hist.value_at_quantile(0.99), 100);
    }
}
