# V2 Full-Text Search Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HashMap-based full-text search with FST + WAND + AVX-512 while keeping the Search trait interface unchanged.

**Architecture:** Three new modules (fst.rs, wand.rs, simd.rs) plus a rewrite of text.rs. FST handles term lookup with prefix search, WAND prunes low-scoring documents during traversal, AVX-512 accelerates BM25 score accumulation. TextIndex internal doc_id changes from String to u32, but external interface remains String-based.

**Tech Stack:** Rust, `std::arch::x86_64` for AVX-512 intrinsics, no new external dependencies

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/index/fst.rs` | Create | FST data structure, builder, exact lookup, prefix search |
| `src/index/simd.rs` | Create | AVX-512 batch BM25 scoring, scalar fallback, runtime detection |
| `src/index/wand.rs` | Create | WAND traversal with PostingCursor, pivot, advance |
| `src/index/text.rs` | Rewrite | New TextIndex with FST, DocIdMap, WAND integration |
| `src/index/mod.rs` | Modify | Add `pub mod fst; pub mod wand; pub mod simd;` |

Unchanged files: `vector.rs`, `graph.rs`, `fusion.rs`, `loader.rs`, `lib.rs`, `main.rs`, `types.rs`, `tests/integration.rs`

---

### Task 1: FST Module — Data Structure and Builder

**Files:**
- Create: `src/index/fst.rs`

- [ ] **Step 1: Write failing tests for FST builder and exact lookup**

```rust
// src/index/fst.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fst_single_key() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 42);
        let fst = builder.build();
        assert_eq!(fst.get(b"cat"), Some(42));
    }

    #[test]
    fn test_fst_multiple_keys() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        builder.insert(b"dog", 2);
        builder.insert(b"elephant", 3);
        let fst = builder.build();
        assert_eq!(fst.get(b"cat"), Some(1));
        assert_eq!(fst.get(b"dog"), Some(2));
        assert_eq!(fst.get(b"elephant"), Some(3));
    }

    #[test]
    fn test_fst_missing_key() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        let fst = builder.build();
        assert_eq!(fst.get(b"dog"), None);
    }

    #[test]
    fn test_fst_empty() {
        let builder = FstBuilder::new();
        let fst = builder.build();
        assert_eq!(fst.get(b"anything"), None);
    }

    #[test]
    fn test_fst_shared_prefix() {
        let mut builder = FstBuilder::new();
        builder.insert(b"car", 1);
        builder.insert(b"cat", 2);
        builder.insert(b"cab", 3);
        let fst = builder.build();
        assert_eq!(fst.get(b"car"), Some(1));
        assert_eq!(fst.get(b"cat"), Some(2));
        assert_eq!(fst.get(b"cab"), Some(3));
    }

    #[test]
    fn test_fst_output_accumulation() {
        // Keys with shared prefix should accumulate outputs along the path
        let mut builder = FstBuilder::new();
        builder.insert(b"a", 10);
        builder.insert(b"ab", 20);
        let fst = builder.build();
        assert_eq!(fst.get(b"a"), Some(10));
        assert_eq!(fst.get(b"ab"), Some(20));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib index::fst`
Expected: FAIL — `FstBuilder` and `Fst` not yet defined

- [ ] **Step 3: Implement FST data structure and builder**

```rust
// src/index/fst.rs

pub struct FstNode {
    pub transitions: Vec<(u8, FstTransition)>,
    pub is_final: bool,
    pub final_output: u64,
}

pub struct FstTransition {
    pub target: usize,
    pub output: u64,
}

pub struct Fst {
    nodes: Vec<FstNode>,
}

pub struct FstBuilder {
    entries: Vec<(Vec<u8>, u64)>,
}

impl FstBuilder {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn insert(&mut self, key: &[u8], value: u64) {
        self.entries.push((key.to_vec(), value));
    }

    pub fn build(mut self) -> Fst {
        // Sort lexicographically
        self.entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut nodes = vec![FstNode {
            transitions: Vec::new(),
            is_final: false,
            final_output: 0,
        }];

        for (key, value) in &self.entries {
            let mut current = 0; // root
            for (i, &byte) in key.iter().enumerate() {
                let is_last = i == key.len() - 1;
                if let Some(pos) = nodes[current].transitions.iter().position(|(b, _)| *b == byte) {
                    if is_last {
                        nodes[current].is_final = true;
                        nodes[current].final_output = *value;
                    }
                    current = nodes[current].transitions[pos].1.target;
                } else {
                    let new_node_idx = nodes.len();
                    nodes.push(FstNode {
                        transitions: Vec::new(),
                        is_final: is_last,
                        final_output: if is_last { *value } else { 0 },
                    });
                    nodes[current].transitions.push((byte, FstTransition {
                        target: new_node_idx,
                        output: if is_last { *value } else { 0 },
                    }));
                    // Sort transitions by byte for deterministic traversal
                    nodes[current].transitions.sort_by_key(|(b, _)| *b);
                    current = new_node_idx;
                }
            }
        }

        Fst { nodes }
    }
}

impl Fst {
    pub fn get(&self, key: &[u8]) -> Option<u64> {
        let mut current = 0;
        for &byte in key {
            let node = &self.nodes[current];
            match node.transitions.binary_search_by_key(&byte, |(b, _)| *b) {
                Ok(idx) => current = node.transitions[idx].1.target,
                Err(_) => return None,
            }
        }
        if self.nodes[current].is_final {
            Some(self.nodes[current].final_output)
        } else {
            None
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib index::fst`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/index/fst.rs
git commit -m "feat: FST data structure with builder and exact lookup"
```

---

### Task 2: FST Module — Prefix Search

**Files:**
- Modify: `src/index/fst.rs`

- [ ] **Step 1: Write failing tests for prefix search**

Add to `#[cfg(test)] mod tests` in `src/index/fst.rs`:

```rust
    #[test]
    fn test_fst_prefix_search_basic() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        builder.insert(b"car", 2);
        builder.insert(b"dog", 3);
        let fst = builder.build();
        let results = fst.prefix_search(b"ca");
        assert_eq!(results.len(), 2);
        let keys: Vec<&str> = results.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"car"));
        assert!(keys.contains(&"cat"));
    }

    #[test]
    fn test_fst_prefix_search_exact_match() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        builder.insert(b"car", 2);
        let fst = builder.build();
        let results = fst.prefix_search(b"cat");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "cat");
        assert_eq!(results[0].1, 1);
    }

    #[test]
    fn test_fst_prefix_search_no_match() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        let fst = builder.build();
        let results = fst.prefix_search(b"z");
        assert!(results.is_empty());
    }

    #[test]
    fn test_fst_prefix_search_empty_prefix() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        builder.insert(b"dog", 2);
        let fst = builder.build();
        let results = fst.prefix_search(b"");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_fst_prefix_search_deep() {
        let mut builder = FstBuilder::new();
        builder.insert(b"prog", 1);
        builder.insert(b"program", 2);
        builder.insert(b"programming", 3);
        builder.insert(b"project", 4);
        let fst = builder.build();
        let results = fst.prefix_search(b"prog");
        assert_eq!(results.len(), 3); // prog, program, programming
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib index::fst`
Expected: FAIL — `prefix_search` not yet defined

- [ ] **Step 3: Implement prefix_search**

Add to `impl Fst`:

```rust
    pub fn prefix_search(&self, prefix: &[u8]) -> Vec<(String, u64)> {
        let mut results = Vec::new();

        // Walk to the end of the prefix
        let mut current = 0;
        for &byte in prefix {
            let node = &self.nodes[current];
            match node.transitions.binary_search_by_key(&byte, |(b, _)| *b) {
                Ok(idx) => current = node.transitions[idx].1.target,
                Err(_) => return results, // prefix not found
            }
        }

        // DFS from the prefix node, collecting all final states
        let prefix_str = String::from_utf8_lossy(prefix).to_string();
        let mut stack: Vec<(usize, String)> = vec![(current, prefix_str)];

        while let Some((node_idx, path)) = stack.pop() {
            let node = &self.nodes[node_idx];
            if node.is_final {
                results.push((path.clone(), node.final_output));
            }
            // Push in reverse order so we visit in sorted order
            for (byte, transition) in node.transitions.iter().rev() {
                let new_path = format!("{}{}", path, *byte as char);
                stack.push((transition.target, new_path));
            }
        }

        results.sort_by(|a, b| a.0.cmp(&b.0));
        results
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib index::fst`
Expected: All 11 tests PASS (6 exact + 5 prefix)

- [ ] **Step 5: Commit**

```bash
git add src/index/fst.rs
git commit -m "feat: FST prefix search with DFS traversal"
```

---

### Task 3: SIMD Module — AVX-512 and Scalar BM25 Scoring

**Files:**
- Create: `src/index/simd.rs`

- [ ] **Step 1: Write failing tests for SIMD scoring**

```rust
// src/index/simd.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_single_term() {
        let idf = vec![1.5];
        let tf_norm = vec![2.0];
        let score = bm25_score_scalar(&idf, &tf_norm);
        assert!((score - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_multi_term() {
        let idf = vec![1.0, 2.0, 0.5];
        let tf_norm = vec![3.0, 1.5, 4.0];
        let score = bm25_score_scalar(&idf, &tf_norm);
        // 1.0*3.0 + 2.0*1.5 + 0.5*4.0 = 3.0 + 3.0 + 2.0 = 8.0
        assert!((score - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_empty() {
        let score = bm25_score_scalar(&[], &[]);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_matches_scalar() {
        let idf = vec![1.5, 2.3, 0.8, 3.1, 1.2, 0.7, 2.0, 1.1, 0.9, 1.8, 2.5, 0.6, 1.4, 3.0, 0.3, 1.7, 2.2, 0.5];
        let tf_norm = vec![0.5, 1.2, 3.0, 0.1, 2.0, 1.5, 0.8, 1.0, 2.5, 0.3, 1.1, 0.9, 1.6, 0.4, 2.0, 0.7, 1.3, 0.6];
        let scalar = bm25_score_scalar(&idf, &tf_norm);
        let batch = bm25_score_batch(&idf, &tf_norm);
        assert!((scalar - batch).abs() < 1e-4, "scalar={} batch={}", scalar, batch);
    }

    #[test]
    fn test_batch_small_input() {
        // Less than 16 elements, tests remainder handling
        let idf = vec![1.0, 2.0, 3.0];
        let tf_norm = vec![4.0, 5.0, 6.0];
        let scalar = bm25_score_scalar(&idf, &tf_norm);
        let batch = bm25_score_batch(&idf, &tf_norm);
        assert!((scalar - batch).abs() < 1e-4);
    }

    #[test]
    fn test_batch_exact_16() {
        let idf: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.1).collect();
        let tf_norm: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.2).collect();
        let scalar = bm25_score_scalar(&idf, &tf_norm);
        let batch = bm25_score_batch(&idf, &tf_norm);
        assert!((scalar - batch).abs() < 1e-4);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib index::simd`
Expected: FAIL — `bm25_score_scalar` and `bm25_score_batch` not yet defined

- [ ] **Step 3: Implement scalar and AVX-512 scoring with runtime dispatch**

```rust
// src/index/simd.rs

pub fn bm25_score_scalar(idf: &[f32], tf_norm: &[f32]) -> f32 {
    idf.iter().zip(tf_norm).map(|(i, t)| i * t).sum()
}

#[cfg(target_arch = "x86_64")]
fn is_avx512_available() -> bool {
    use std::arch::x86_64::__cpuid;
    unsafe {
        let eax7 = __cpuid(7);
        (eax7.ebx & (1 << 16)) != 0
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn is_avx512_available() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn bm25_score_avx512(idf: &[f32], tf_norm: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = idf.len();
    let mut sum512 = _mm512_setzero_ps();
    let mut i = 0;

    while i + 16 <= len {
        let idf_vec = _mm512_loadu_ps(idf.as_ptr().add(i));
        let tf_vec = _mm512_loadu_ps(tf_norm.as_ptr().add(i));
        let products = _mm512_mul_ps(idf_vec, tf_vec);
        sum512 = _mm512_add_ps(sum512, products);
        i += 16;
    }

    // Horizontal sum of __m512
    let mut total = {
        let lo256 = _mm512_extractf32x8_ps(sum512, 0);
        let hi256 = _mm512_extractf32x8_ps(sum512, 1);
        let sum256 = _mm256_add_ps(lo256, hi256);
        // Horizontal sum of __m256
        let shuf = _mm256_movehdup_ps(sum256);
        let sums = _mm256_add_ps(sum256, shuf);
        let shuf = _mm256_movehl_ps(shuf, sums);
        let sums = _mm256_add_ps(sums, shuf);
        let result = _mm256_extractf128_ps(sums, 1);
        let lo = _mm_castps128_ps128(result);
        _mm_cvtss_f32(lo)
    };

    // Remainder
    while i < len {
        total += idf[i] * tf_norm[i];
        i += 1;
    }

    total
}

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib index::simd`
Expected: All 6 tests PASS (on non-AVX-512 machines, batch falls back to scalar which matches exactly)

- [ ] **Step 5: Commit**

```bash
git add src/index/simd.rs
git commit -m "feat: AVX-512 and scalar BM25 batch scoring with runtime dispatch"
```

---

### Task 4: WAND Module

**Files:**
- Create: `src/index/wand.rs`

- [ ] **Step 1: Write failing tests for WAND**

```rust
// src/index/wand.rs

#[cfg(test)]
mod tests {
    use super::*;

    fn make_posting(doc_id: u32, tf: u32) -> Posting {
        Posting { doc_id, tf }
    }

    #[test]
    fn test_wand_single_term() {
        let posting_lists = vec![
            vec![make_posting(0, 3), make_posting(2, 1), make_posting(5, 2)],
        ];
        let upper_bounds = vec![2.0];
        let idf_values = vec![1.5];
        let doc_lengths = vec![10.0, 8.0, 12.0, 6.0, 9.0, 11.0];
        let avg_dl = 9.0;

        let results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 2);
        assert_eq!(results.len(), 2);
        // Doc 0 has tf=3 (highest), should rank first
        assert_eq!(results[0].doc_id, 0);
    }

    #[test]
    fn test_wand_multi_term() {
        let posting_lists = vec![
            vec![make_posting(0, 2), make_posting(1, 1), make_posting(3, 3)],
            vec![make_posting(0, 1), make_posting(2, 4), make_posting(3, 1)],
        ];
        let upper_bounds = vec![3.0, 2.5];
        let idf_values = vec![1.5, 2.0];
        let doc_lengths = vec![10.0, 8.0, 12.0, 6.0];
        let avg_dl = 9.0;

        let results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 2);
        assert_eq!(results.len(), 2);
        // doc 0 appears in both terms, doc 3 appears in both terms with high tf
        assert!(results.iter().any(|r| r.doc_id == 0));
        assert!(results.iter().any(|r| r.doc_id == 3));
    }

    #[test]
    fn test_wand_empty() {
        let posting_lists: Vec<Vec<Posting>> = vec![];
        let upper_bounds = vec![];
        let idf_values = vec![];
        let doc_lengths = vec![10.0];
        let avg_dl = 10.0;

        let results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_wand_matches_full_scan() {
        // WAND should produce same top-k as full scan
        let posting_lists = vec![
            vec![make_posting(0, 1), make_posting(1, 5), make_posting(3, 2), make_posting(4, 1)],
            vec![make_posting(0, 3), make_posting(2, 2), make_posting(3, 4)],
            vec![make_posting(1, 1), make_posting(2, 3), make_posting(4, 2)],
        ];
        let upper_bounds = vec![2.5, 3.0, 1.8];
        let idf_values = vec![1.5, 2.0, 1.2];
        let doc_lengths = vec![10.0, 8.0, 12.0, 6.0, 9.0];
        let avg_dl = 9.0;

        let wand_results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 3);
        let full_results = full_scan_search(&posting_lists, &idf_values, &doc_lengths, avg_dl, 3);

        assert_eq!(wand_results.len(), full_results.len());
        for (w, f) in wand_results.iter().zip(full_results.iter()) {
            assert_eq!(w.doc_id, f.doc_id);
            assert!((w.score - f.score).abs() < 1e-4, "wand={} full={}", w.score, f.score);
        }
    }

    #[test]
    fn test_wand_pruning_effect() {
        // With many docs, WAND should skip some
        let mut pl1 = vec![];
        let mut pl2 = vec![];
        for i in 0..100 {
            pl1.push(make_posting(i, 1));
            if i % 2 == 0 {
                pl2.push(make_posting(i, 1));
            }
        }
        let posting_lists = vec![pl1, pl2];
        let upper_bounds = vec![2.0, 1.5];
        let idf_values = vec![1.0, 1.0];
        let doc_lengths = vec![10.0; 100];
        let avg_dl = 10.0;

        let results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 3);
        assert_eq!(results.len(), 3);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib index::wand`
Expected: FAIL — `wand_search`, `full_scan_search`, `Posting` not yet defined

- [ ] **Step 3: Implement WAND traversal**

```rust
// src/index/wand.rs

use crate::index::simd::bm25_score_batch;

const K1: f32 = 1.2;
const B: f32 = 0.75;

pub struct Posting {
    pub doc_id: u32,
    pub tf: u32,
}

pub struct WandResult {
    pub doc_id: u32,
    pub score: f32,
}

struct PostingCursor<'a> {
    term_idx: usize,
    posting_list: &'a [Posting],
    current_pos: usize,
    upper_bound: f32,
}

impl<'a> PostingCursor<'a> {
    fn current_doc_id(&self) -> u32 {
        self.posting_list[self.current_pos].doc_id
    }

    fn advance_to(&mut self, target_doc_id: u32) {
        while self.current_pos < self.posting_list.len()
            && self.posting_list[self.current_pos].doc_id < target_doc_id
        {
            self.current_pos += 1;
        }
    }

    fn is_exhausted(&self) -> bool {
        self.current_pos >= self.posting_list.len()
    }
}

pub fn wand_search(
    posting_lists: &[Vec<Posting>],
    upper_bounds: &[f32],
    idf_values: &[f32],
    doc_lengths: &[f32],
    avg_dl: f32,
    top_k: usize,
) -> Vec<WandResult> {
    if posting_lists.is_empty() || top_k == 0 {
        return vec![];
    }

    let mut cursors: Vec<PostingCursor> = posting_lists
        .iter()
        .enumerate()
        .filter(|(_, pl)| !pl.is_empty())
        .map(|(i, pl)| PostingCursor {
            term_idx: i,
            posting_list: pl,
            current_pos: 0,
            upper_bound: upper_bounds[i],
        })
        .collect();

    if cursors.is_empty() {
        return vec![];
    }

    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(f32, u32)>> = std::collections::BinaryHeap::new();
    let mut threshold: f32 = 0.0;

    loop {
        // Sort cursors by current doc_id
        cursors.sort_by_key(|c| c.current_doc_id());

        // Find pivot: smallest doc_id where sum of upper bounds >= threshold
        let mut upper_sum: f32 = 0.0;
        let mut pivot_idx = None;
        for (i, cursor) in cursors.iter().enumerate() {
            upper_sum += cursor.upper_bound;
            if upper_sum > threshold {
                pivot_idx = Some(i);
                break;
            }
        }

        let pivot_idx = match pivot_idx {
            Some(idx) => idx,
            None => break, // No document can exceed threshold
        };

        let pivot_doc_id = cursors[pivot_idx].current_doc_id();

        if cursors[0].current_doc_id() == pivot_doc_id {
            // All cursors up to pivot point to same doc → score it
            let score = score_document(&cursors, pivot_doc_id, idf_values, doc_lengths, avg_dl);

            if heap.len() < top_k {
                heap.push(std::cmp::Reverse((score, pivot_doc_id)));
                if heap.len() == top_k {
                    threshold = heap.peek().map(|r| r.0.0).unwrap_or(0.0);
                }
            } else if score > threshold {
                heap.pop();
                heap.push(std::cmp::Reverse((score, pivot_doc_id)));
                threshold = heap.peek().map(|r| r.0.0).unwrap_or(0.0);
            }

            // Advance all cursors at pivot_doc_id
            for cursor in &mut cursors {
                if !cursor.is_exhausted() && cursor.current_doc_id() == pivot_doc_id {
                    cursor.current_pos += 1;
                }
            }
        } else {
            // Advance all cursors before pivot to pivot_doc_id
            for cursor in &mut cursors[..pivot_idx] {
                cursor.advance_to(pivot_doc_id);
            }
        }

        // Remove exhausted cursors
        cursors.retain(|c| !c.is_exhausted());

        if cursors.is_empty() {
            break;
        }
    }

    let mut results: Vec<WandResult> = heap
        .into_iter()
        .map(|Reverse((score, doc_id))| WandResult { doc_id, score })
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn score_document(
    cursors: &[PostingCursor],
    doc_id: u32,
    idf_values: &[f32],
    doc_lengths: &[f32],
    avg_dl: f32,
) -> f32 {
    let dl = doc_lengths.get(doc_id as usize).copied().unwrap_or(1.0);
    let doc_count = doc_lengths.len() as f32;

    let mut idf_list: Vec<f32> = Vec::new();
    let mut tf_norm_list: Vec<f32> = Vec::new();

    for cursor in cursors {
        if cursor.is_exhausted() || cursor.current_doc_id() != doc_id {
            continue;
        }
        let posting = &cursor.posting_list[cursor.current_pos];
        let tf = posting.tf as f32;
        let df = cursor.posting_list.len() as f32;

        let idf = (1.0 + (doc_count - df + 0.5) / (df + 0.5)).ln();
        let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avg_dl));

        idf_list.push(idf);
        tf_norm_list.push(tf_norm);
    }

    bm25_score_batch(&idf_list, &tf_norm_list)
}

/// Full-scan baseline for testing WAND correctness
pub fn full_scan_search(
    posting_lists: &[Vec<Posting>],
    idf_values: &[f32],
    doc_lengths: &[f32],
    avg_dl: f32,
    top_k: usize,
) -> Vec<WandResult> {
    let doc_count = doc_lengths.len() as f32;
    let mut scores: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();

    for (term_idx, postings) in posting_lists.iter().enumerate() {
        let df = postings.len() as f32;
        let idf = (1.0 + (doc_count - df + 0.5) / (df + 0.5)).ln();
        for posting in postings {
            let dl = doc_lengths.get(posting.doc_id as usize).copied().unwrap_or(1.0);
            let tf = posting.tf as f32;
            let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avg_dl));
            *scores.entry(posting.doc_id).or_insert(0.0) += idf * tf_norm;
        }
    }

    let mut results: Vec<WandResult> = scores
        .into_iter()
        .map(|(doc_id, score)| WandResult { doc_id, score })
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}
```

- [ ] **Step 4: Update `src/index/mod.rs` to add new modules**

```rust
pub mod vector;
pub mod text;
pub mod graph;
pub mod fst;
pub mod wand;
pub mod simd;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib index::wand`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/index/wand.rs src/index/mod.rs
git commit -m "feat: WAND traversal with BM25 scoring and pruning"
```

---

### Task 5: Rewrite TextIndex with FST + WAND + SIMD

**Files:**
- Modify: `src/index/text.rs`

- [ ] **Step 1: Write the new TextIndex with all V1 tests preserved plus new prefix search test**

Replace the entire `src/index/text.rs` with the new implementation. The key changes:
- `HashMap<String, PostingList>` → `Fst` + `Vec<PostingList>` + `Vec<String>` (term_strings)
- `String` doc_ids → `u32` internal doc_ids with `DocIdMap`
- Full-scan query → WAND traversal
- Scalar BM25 accumulation → `bm25_score_batch` dispatch
- New `prefix_search` method on TextIndex

```rust
// src/index/text.rs

use std::collections::HashMap;

use crate::types::{Document, Query, Search, SearchSource, SearchResult};
use crate::index::fst::{Fst, FstBuilder};
use crate::index::wand::{self, Posting, WandResult};

const K1: f32 = 1.2;
const B: f32 = 0.75;

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

struct DocIdMap {
    to_internal: HashMap<String, u32>,
    to_external: Vec<String>,
}

impl DocIdMap {
    fn new() -> Self {
        Self {
            to_internal: HashMap::new(),
            to_external: Vec::new(),
        }
    }

    fn get_or_insert(&mut self, external: &str) -> u32 {
        if let Some(&id) = self.to_internal.get(external) {
            return id;
        }
        let id = self.to_external.len() as u32;
        self.to_external.push(external.to_string());
        self.to_internal.insert(external.to_string(), id);
        id
    }

    fn to_external(&self, internal: u32) -> &str {
        &self.to_external[internal as usize]
    }
}

pub struct TextIndex {
    fst: Fst,
    posting_lists: Vec<wand::Posting>,
    posting_offsets: Vec<usize>,      // posting_offsets[i] = start of term i's postings
    posting_lengths: Vec<usize>,      // posting_lengths[i] = count of term i's postings
    term_strings: Vec<String>,        // index → term string
    term_upper_bounds: Vec<f32>,      // index → BM25 upper bound
    term_idf: Vec<f32>,               // index → IDF value
    doc_id_map: DocIdMap,
    doc_lengths: Vec<f32>,            // u32 doc_id → doc length
    avg_dl: f32,
    doc_count: usize,
    doc_texts: Vec<String>,           // u32 doc_id → original text
}

impl TextIndex {
    pub fn new(docs: Vec<Document>) -> Self {
        let doc_count = docs.len();
        let mut doc_id_map = DocIdMap::new();
        let mut doc_lengths: Vec<f32> = Vec::new();
        let mut doc_texts: Vec<String> = Vec::new();
        let mut total_length: usize = 0;

        // Collect all (term, doc_id, tf) tuples
        let mut term_doc_tf: Vec<(String, u32, u32)> = Vec::new();

        for doc in &docs {
            let internal_id = doc_id_map.get_or_insert(&doc.id);
            let tokens = tokenize(&doc.text);
            doc_lengths.push(tokens.len() as f32);
            total_length += tokens.len();
            doc_texts.push(doc.text.clone());

            // Count tf per term per doc
            let mut term_counts: HashMap<String, u32> = HashMap::new();
            for token in &tokens {
                *term_counts.entry(token.clone()).or_insert(0) += 1;
            }
            for (term, tf) in term_counts {
                term_doc_tf.push((term, internal_id, tf));
            }
        }

        // Ensure doc_lengths and doc_texts cover all doc_ids
        while doc_lengths.len() < doc_id_map.to_external.len() {
            doc_lengths.push(0.0);
            doc_texts.push(String::new());
        }

        let avg_dl = if doc_count > 0 {
            total_length as f32 / doc_count as f32
        } else {
            0.0
        };

        // Group by term, build posting lists sorted by doc_id
        term_doc_tf.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut all_postings: Vec<wand::Posting> = Vec::new();
        let mut posting_offsets: Vec<usize> = Vec::new();
        let mut posting_lengths: Vec<usize> = Vec::new();
        let mut term_strings: Vec<String> = Vec::new();
        let mut term_upper_bounds: Vec<f32> = Vec::new();
        let mut term_idf: Vec<f32> = Vec::new();

        let mut i = 0;
        while i < term_doc_tf.len() {
            let term = &term_doc_tf[i].0;
            let offset = all_postings.len();
            let mut df: usize = 0;

            while i < term_doc_tf.len() && term_doc_tf[i].0 == *term {
                let doc_id = term_doc_tf[i].1;
                let tf = term_doc_tf[i].2;
                all_postings.push(Posting { doc_id, tf });
                df += 1;
                i += 1;
            }

            posting_offsets.push(offset);
            posting_lengths.push(df);
            term_strings.push(term.clone());

            // Precompute upper bound: max BM25 score for this term (tf at maximum)
            let idf = (1.0 + (doc_count as f32 - df as f32 + 0.5) / (df as f32 + 0.5)).ln();
            // Upper bound: assume tf is very large → tf_norm → K1 + 1
            let upper_bound = idf * (K1 + 1.0);
            term_upper_bounds.push(upper_bound);
            term_idf.push(idf);
        }

        // Build FST
        let mut fst_builder = FstBuilder::new();
        for (idx, term) in term_strings.iter().enumerate() {
            fst_builder.insert(term.as_bytes(), idx as u64);
        }
        let fst = fst_builder.build();

        TextIndex {
            fst,
            posting_lists: all_postings,
            posting_offsets,
            posting_lengths,
            term_strings,
            term_upper_bounds,
            term_idf,
            doc_id_map,
            doc_lengths,
            avg_dl,
            doc_count,
            doc_texts,
        }
    }

    fn get_posting_list(&self, term_idx: usize) -> Vec<wand::Posting> {
        let offset = self.posting_offsets[term_idx];
        let len = self.posting_lengths[term_idx];
        self.posting_lists[offset..offset + len].to_vec()
    }

    pub fn prefix_search(&self, prefix: &str) -> Vec<String> {
        self.fst.prefix_search(prefix.as_bytes())
            .into_iter()
            .map(|(term, _)| term)
            .collect()
    }
}

impl Search for TextIndex {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult> {
        let query_text = match query {
            Query::Text(t) => t,
            Query::Hybrid { text, .. } => text,
            _ => return vec![],
        };
        let tokens = tokenize(query_text);
        if tokens.is_empty() {
            return vec![];
        }

        // Collect posting lists for all query terms via FST
        let mut posting_lists: Vec<Vec<wand::Posting>> = Vec::new();
        let mut upper_bounds: Vec<f32> = Vec::new();
        let mut idf_values: Vec<f32> = Vec::new();

        for token in &tokens {
            if let Some(idx) = self.fst.get(token.as_bytes()) {
                let idx = idx as usize;
                posting_lists.push(self.get_posting_list(idx));
                upper_bounds.push(self.term_upper_bounds[idx]);
                idf_values.push(self.term_idf[idx]);
            }
        }

        if posting_lists.is_empty() {
            return vec![];
        }

        // WAND search
        let wand_results = wand::wand_search(
            &posting_lists,
            &upper_bounds,
            &idf_values,
            &self.doc_lengths,
            self.avg_dl,
            top_k,
        );

        // Convert results
        wand_results
            .into_iter()
            .map(|r| {
                let external_id = self.doc_id_map.to_external(r.doc_id).to_string();
                let text = self.doc_texts.get(r.doc_id as usize).cloned().unwrap_or_default();
                SearchResult {
                    id: external_id,
                    score: r.score,
                    source: SearchSource::Text,
                    snippet: make_snippet_simple(&text, 80),
                }
            })
            .collect()
    }
}

fn make_snippet_simple(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        let mut end = max_len;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &text[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, world! This is Rust.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "rust"]);
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_text_index_search() {
        let docs = vec![
            Document { id: "d1".into(), text: "Rust is a systems programming language.".into(), vector: vec![] },
            Document { id: "d2".into(), text: "Python is a popular scripting language.".into(), vector: vec![] },
            Document { id: "d3".into(), text: "Rust programming is fun and productive.".into(), vector: vec![] },
        ];
        let index = TextIndex::new(docs);
        let results = index.search(&Query::Text("rust programming".into()), 2);
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"d3"), "d3 should be in top results");
        assert_eq!(results[0].source, SearchSource::Text);
    }

    #[test]
    fn test_text_index_no_match() {
        let docs = vec![
            Document { id: "d1".into(), text: "Rust is a systems language.".into(), vector: vec![] },
        ];
        let index = TextIndex::new(docs);
        let results = index.search(&Query::Text("python".into()), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_scoring() {
        let docs = vec![
            Document { id: "d1".into(), text: "rust rust rust".into(), vector: vec![] },
            Document { id: "d2".into(), text: "rust".into(), vector: vec![] },
        ];
        let index = TextIndex::new(docs);
        let results = index.search(&Query::Text("rust".into()), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "d1", "doc with higher tf should score higher");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_prefix_search() {
        let docs = vec![
            Document { id: "d1".into(), text: "Rust programming is fun".into(), vector: vec![] },
            Document { id: "d2".into(), text: "Python programming language".into(), vector: vec![] },
            Document { id: "d3".into(), text: "A program for data analysis".into(), vector: vec![] },
        ];
        let index = TextIndex::new(docs);
        let terms = index.prefix_search("prog");
        assert!(terms.contains(&"programming".to_string()) || terms.contains(&"program".to_string()),
            "prefix search for 'prog' should find 'programming' or 'program', got {:?}", terms);
    }
}
```

- [ ] **Step 2: Run all tests**

Run: `cargo test --lib`
Expected: All tests PASS (including text, fst, wand, simd, and other modules)

- [ ] **Step 3: Run integration tests**

Run: `cargo test`
Expected: All integration tests PASS

- [ ] **Step 4: Run CLI end-to-end test**

```bash
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json text --query "rust programming" --top-k 3
```
Expected: Returns ranked text search results

- [ ] **Step 5: Commit**

```bash
git add src/index/text.rs
git commit -m "feat: rewrite TextIndex with FST, WAND, and SIMD-accelerated BM25"
```

---

### Task 6: Update mod.rs and Verify Full System

**Files:**
- Modify: `src/index/mod.rs` (already done in Task 4, verify)
- Verify: all tests pass, clippy clean

- [ ] **Step 1: Verify `src/index/mod.rs` has all modules**

```rust
pub mod vector;
pub mod text;
pub mod graph;
pub mod fst;
pub mod wand;
pub mod simd;
```

- [ ] **Step 2: Run full test suite**

Run: `cargo test`
Expected: All unit + integration tests PASS

- [ ] **Step 3: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings. If AVX-512 intrinsics trigger warnings on non-x86_64, add appropriate `#[allow]` attributes.

- [ ] **Step 4: Run all CLI search modes**

```bash
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json text --query "systems programming" --top-k 3
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json vector --query-vector "0.9,0.1,0.0,0.0" --top-k 3
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json graph --entity "Rust" --hops 2
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json hybrid --query "systems programming" --query-vector "0.9,0.1,0.0,0.0" --top-k 3
```
Expected: All modes return meaningful results

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: V2 full-text search upgrade complete — FST + WAND + AVX-512"
```
