# Mini Hybrid Retrieval V2 — Full-Text Search Upgrade Design

## Overview

Upgrade the full-text search module from HashMap-based inverted index to FST + WAND + AVX-512, while keeping the rest of the system (vector, graph, fusion, CLI) unchanged. The TextIndex still implements the same `Search` trait, so no external interface changes.

## Scope

- **Only `src/index/text.rs` and new files under `src/index/`** — no changes to vector, graph, fusion, loader, Engine, or CLI
- Three new modules: `fst.rs`, `wand.rs`, `simd.rs`
- Internal doc_id changes from `String` to `u32` for WAND sorting; external interface still uses `String`

## Component 1: FST (Finite State Transducer)

### Data Structure

```rust
struct FstNode {
    transitions: Vec<(u8, FstTransition)>,
    is_final: bool,
    final_output: u64,
}

struct FstTransition {
    target: usize,
    output: u64,
}

struct Fst {
    nodes: Vec<FstNode>,  // index 0 is root
}
```

### Core Operations

- **`Fst::get(key: &[u8]) -> Option<u64>`** — exact lookup, walk byte path, accumulate output
- **`Fst::prefix_search(prefix: &[u8]) -> Vec<(String, u64)>`** — walk to prefix end, DFS collect all final states
- **`FstBuilder::insert(key: &[u8], value: u64)`** — insert key-value during build
- **`FstBuilder::build() -> Fst`** — freeze into immutable FST

### Build Algorithm

Lexicographic batch build:
1. Sort all key-value pairs by key
2. Insert one by one, sharing common prefixes
3. V2 implements minimal version: prefix sharing only, no full minimization

~400 lines total (build ~250, query ~150).

### Role in TextIndex

```
Old: HashMap<String, PostingList>  →  exact term lookup
New: Fst (term → u64 offset)       →  posting_lists[offset]
                                   →  term_strings[offset]  (reverse lookup)
```

FST output is the posting list index. Prefix search returns all matching term indices.

## Component 2: WAND Pruning

### Algorithm

WAND skips documents that cannot enter the top-k by maintaining per-term upper bounds:

1. Open a `PostingCursor` per query term
2. All posting lists sorted by doc_id (guaranteed at build time)
3. Loop:
   - Sort cursors by current doc_id
   - Accumulate upper bounds from smallest doc_id until sum >= threshold
   - If pivot cursor's doc_id matches the smallest → score that doc exactly
   - Otherwise → advance the smallest cursor to pivot doc_id
   - After scoring, if doc enters top-k, update threshold
4. Return top-k results

### Data Structure

```rust
struct PostingCursor<'a> {
    term_idx: usize,
    posting_list: &'a [Posting],
    current_pos: usize,
    upper_bound: f32,
}
```

### Key Design Points

- **Upper bound precomputation**: at build time, compute `max_bm25_score` for each term (tf at maximum), store in `term_upper_bounds`
- **DocID as u32**: WAND requires sortable doc_ids; internal `u32` with `DocIdMap` for String↔u32 bidirectional mapping
- **Posting list sorted by doc_id**: guaranteed at build time, WAND relies on this for pivot and advance

### FST Integration

```
query "rust prog"
  → FST exact lookup "rust" → posting_idx 5
  → FST exact lookup "prog" → posting_idx 12
  → FST prefix search "prog" → posting_idx [12, 13, 14]

  → Create WAND cursors for all posting lists
  → WAND traverses cursors, skips low-scoring docs
  → Exact BM25 scoring for candidate docs
```

Prefix-expanded terms share a single WAND traversal — no separate queries to merge.

## Component 3: AVX-512 SIMD Acceleration

### Target

BM25 score accumulation across multiple query terms:

```
score(d) = Σ_{t ∈ query} IDF(t) × tf_norm(t, d)
```

AVX-512 `__m512` processes 16 × f32 per instruction.

### Implementation

```rust
#[target_feature(enable = "avx512f")]
unsafe fn bm25_score_avx512(idf: &[f32], tf_norm: &[f32]) -> f32 {
    // Process 16 f32 at a time with _mm512_mul_ps + _mm512_add_ps
    // Scalar fallback for remainder
    // Horizontal sum for final reduction
}

fn bm25_score_scalar(idf: &[f32], tf_norm: &[f32]) -> f32 {
    idf.iter().zip(tf_norm).map(|(i, t)| i * t).sum()
}

pub fn bm25_score_batch(idf: &[f32], tf_norm: &[f32]) -> f32 {
    // Runtime CPUID check → dispatch to AVX-512 or scalar
}
```

### Runtime Detection

```rust
fn is_avx512_available() -> bool {
    // CPUID leaf 7, EBX bit 16 = AVX-512 F
}
```

### Design Points

- `#[target_feature(enable = "avx512f")]` — only compiled on x86_64 with AVX-512 support
- Runtime dispatch: detect CPU, choose AVX-512 or scalar automatically
- Non-x86_64 platforms (ARM, etc.) compile scalar path only, no errors
- Horizontal sum: hand-written `_mm512_extractf32x8_ps` + `_mm256_reduce_ps` since `_mm512_reduce_add_ps` may not be available

### What NOT to Accelerate

- WAND pivot/advance: integer comparisons, not SIMD-friendly
- FST traversal: byte-level state machine, not SIMD-friendly
- Only the final scoring phase (multi-term BM25 accumulation) uses AVX-512

## New TextIndex Structure

```rust
pub struct TextIndex {
    fst: Fst,
    posting_lists: Vec<PostingList>,
    term_strings: Vec<String>,
    term_upper_bounds: Vec<f32>,
    doc_id_map: DocIdMap,
    doc_lengths: Vec<f32>,
    avg_dl: f32,
    doc_count: usize,
    doc_texts: Vec<String>,
}

struct DocIdMap {
    to_internal: HashMap<String, u32>,
    to_external: Vec<String>,
}

struct PostingList {
    entries: Vec<Posting>,
    df: usize,
}

struct Posting {
    doc_id: u32,
    tf: u32,
}
```

### Build Flow

1. Tokenize all docs → collect (term, doc_id, tf) tuples
2. Map doc_ids: String → u32
3. Group by term, build posting lists (sorted by doc_id)
4. Precompute BM25 upper bound per term
5. Sort terms lexicographically, build FST (term → posting list index)

### Query Flow

1. Tokenize query → query_tokens
2. For each token: FST exact lookup → posting list index; optionally FST prefix search
3. Collect all posting list indices → create WAND cursors
4. WAND traversal: pivot, skip, score candidates
5. For candidate docs: collect idf[] and tf_norm[], compute BM25 via AVX-512 batch scoring
6. Top-k heap sort → SearchResult

## File Structure

```
src/index/
├── mod.rs          # add: pub mod fst; pub mod wand; pub mod simd;
├── vector.rs       # unchanged
├── graph.rs        # unchanged
├── text.rs         # rewrite: new TextIndex + DocIdMap
├── fst.rs          # new: FST implementation
├── wand.rs         # new: WAND traversal
└── simd.rs         # new: AVX-512 + scalar fallback
```

## Search Trait Compatibility

TextIndex still implements `Search` trait — no interface changes. Engine, CLI, and other modules require zero modifications.

## Testing

- **fst.rs**: exact lookup, prefix search, empty FST, single-key FST, shared prefix
- **wand.rs**: WAND top-k correctness (compare against full-scan), empty query, single term, multi-term
- **simd.rs**: AVX-512 path vs scalar path result consistency
- **text.rs**: inherit all V1 tests (tokenize, BM25 ranking, no-match), add prefix search test

## V2 Scope Boundaries

- Only full-text search module changes
- Vector, graph, fusion, loader, Engine, CLI unchanged
- No persistence changes
- No new dependencies — FST, WAND, SIMD all hand-written
