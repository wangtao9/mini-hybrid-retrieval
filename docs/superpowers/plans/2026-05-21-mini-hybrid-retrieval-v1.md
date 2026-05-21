# Mini Hybrid Retrieval V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal multimodal hybrid retrieval engine in Rust with vector, full-text, and graph search, plus RRF fusion.

**Architecture:** Modular layered — three independent index modules (VectorIndex, TextIndex, GraphIndex) behind a unified Search trait, orchestrated by an Engine that applies RRF fusion. CLI via clap, data loaded from JSON files, everything in-memory.

**Tech Stack:** Rust, serde + serde_json (deserialization), clap (CLI)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `Cargo.toml` | Project manifest with 3 dependencies |
| `src/types.rs` | Document, Triple, SearchResult, SearchSource, Query, Search trait |
| `src/index/mod.rs` | Module re-exports |
| `src/index/vector.rs` | VectorIndex: brute-force KNN with cosine similarity |
| `src/index/text.rs` | TextIndex: inverted index, tokenizer, BM25 scoring, snippet extraction |
| `src/index/graph.rs` | GraphIndex: bidirectional adjacency list, BFS traversal, hop-decay scoring |
| `src/fusion.rs` | RRF (Reciprocal Rank Fusion) implementation |
| `src/loader.rs` | JSON data loading for documents, triples, entity_docs |
| `src/lib.rs` | Engine struct, query orchestration, public re-exports |
| `src/main.rs` | CLI entry point with clap subcommands |
| `data/documents.json` | Sample documents with vectors |
| `data/triples.json` | Sample knowledge graph triples |
| `data/entity_docs.json` | Sample entity-to-document mappings |
| `tests/integration.rs` | End-to-end integration test |

---

### Task 1: Project Scaffolding and Core Types

**Files:**
- Create: `Cargo.toml`
- Create: `src/types.rs`
- Create: `src/lib.rs` (minimal, just re-exports types)

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "mini-hybrid-retrieval"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
```

- [ ] **Step 2: Create src/types.rs with all core types**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SearchSource {
    Vector,
    Text,
    Graph,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub source: SearchSource,
    pub snippet: String,
}

#[derive(Debug, Clone)]
pub enum Query {
    Vector(Vec<f32>),
    Text(String),
    Graph { entity: String, hops: usize },
    Hybrid {
        text: String,
        vector: Vec<f32>,
        entity: Option<String>,
    },
}

pub trait Search {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult>;
}
```

- [ ] **Step 3: Create minimal src/lib.rs**

```rust
pub mod types;
pub mod index;
pub mod fusion;
pub mod loader;

pub use types::{Document, Query, Search, SearchSource, SearchResult, Triple};
```

- [ ] **Step 4: Create placeholder src/index/mod.rs**

```rust
pub mod vector;
pub mod text;
pub mod graph;
```

- [ ] **Step 5: Create placeholder src/fusion.rs**

```rust
// Will implement RRF fusion
```

- [ ] **Step 6: Create placeholder src/loader.rs**

```rust
// Will implement JSON data loading
```

- [ ] **Step 7: Create placeholder src/main.rs**

```rust
fn main() {
    println!("mini-hybrid-retrieval v0.1.0");
}
```

- [ ] **Step 8: Verify project compiles**

Run: `cargo build`
Expected: Compiles successfully with no errors

- [ ] **Step 9: Commit**

```bash
git add Cargo.toml src/
git commit -m "feat: project scaffolding with core types"
```

---

### Task 2: VectorIndex — Brute-force KNN

**Files:**
- Create: `src/index/vector.rs`

- [ ] **Step 1: Write failing tests for cosine_similarity and VectorIndex**

```rust
// src/index/vector.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_vector_search_top_k() {
        let docs = vec![
            Document { id: "d1".into(), text: String::new(), vector: vec![1.0, 0.0] },
            Document { id: "d2".into(), text: String::new(), vector: vec![0.0, 1.0] },
            Document { id: "d3".into(), text: String::new(), vector: vec![0.9, 0.1] },
        ];
        let index = VectorIndex::new(docs);
        let results = index.search(&Query::Vector(vec![1.0, 0.0]), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "d1");
        assert_eq!(results[1].id, "d3");
    }

    #[test]
    fn test_vector_search_empty_index() {
        let index = VectorIndex::new(vec![]);
        let results = index.search(&Query::Vector(vec![1.0, 0.0]), 5);
        assert!(results.is_empty());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib index::vector`
Expected: FAIL — `cosine_similarity` and `VectorIndex` not yet defined

- [ ] **Step 3: Implement cosine_similarity and VectorIndex**

```rust
// src/index/vector.rs

use crate::types::{Document, Query, Search, SearchSource, SearchResult};

pub struct VectorIndex {
    docs: Vec<Document>,
    dim: usize,
}

impl VectorIndex {
    pub fn new(docs: Vec<Document>) -> Self {
        let dim = docs.first().map(|d| d.vector.len()).unwrap_or(0);
        Self { docs, dim }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

impl Search for VectorIndex {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult> {
        let query_vec = match query {
            Query::Vector(v) => v,
            _ => return vec![],
        };
        let mut scored: Vec<_> = self.docs.iter()
            .map(|doc| {
                let sim = cosine_similarity(query_vec, &doc.vector);
                (doc.id.clone(), sim, doc.text.clone())
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter()
            .take(top_k)
            .map(|(id, score, text)| SearchResult {
                id,
                score,
                source: SearchSource::Vector,
                snippet: truncate(&text, 80),
            })
            .collect()
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..s.char_indices().take(max_len).last().map(|(i, _)| i).unwrap_or(0)])
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib index::vector`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/index/vector.rs
git commit -m "feat: VectorIndex with brute-force KNN and cosine similarity"
```

---

### Task 3: TextIndex — Inverted Index with BM25

**Files:**
- Create: `src/index/text.rs`

- [ ] **Step 1: Write failing tests for tokenizer, BM25, and TextIndex**

```rust
// src/index/text.rs

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
            Document { id: "d1".into(), text: "Rust is a systems programming language".into(), vector: vec![] },
            Document { id: "d2".into(), text: "Python is a scripting language".into(), vector: vec![] },
            Document { id: "d3".into(), text: "Rust programming is fun".into(), vector: vec![] },
        ];
        let index = TextIndex::new(docs);
        let results = index.search(&Query::Text("rust programming".into()), 2);
        assert_eq!(results.len(), 2);
        // d3 has both "rust" and "programming", should rank higher
        assert_eq!(results[0].id, "d3");
    }

    #[test]
    fn test_text_index_no_match() {
        let docs = vec![
            Document { id: "d1".into(), text: "Hello world".into(), vector: vec![] },
        ];
        let index = TextIndex::new(docs);
        let results = index.search(&Query::Text("rust".into()), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_scoring() {
        // Document with higher tf for query term should score higher
        let docs = vec![
            Document { id: "d1".into(), text: "rust rust rust".into(), vector: vec![] },
            Document { id: "d2".into(), text: "rust".into(), vector: vec![] },
        ];
        let index = TextIndex::new(docs);
        let results = index.search(&Query::Text("rust".into()), 2);
        assert_eq!(results[0].id, "d1");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib index::text`
Expected: FAIL — `tokenize` and `TextIndex` not yet defined

- [ ] **Step 3: Implement tokenizer, TextIndex, and BM25**

```rust
// src/index/text.rs

use std::collections::HashMap;

use crate::types::{Document, Query, Search, SearchSource, SearchResult};

const K1: f32 = 1.2;
const B: f32 = 0.75;

pub struct Posting {
    pub doc_id: String,
    pub tf: usize,
    pub positions: Vec<usize>,
}

pub struct PostingList {
    pub entries: Vec<Posting>,
}

pub struct TextIndex {
    inverted: HashMap<String, PostingList>,
    doc_lengths: HashMap<String, usize>,
    avg_dl: f32,
    doc_count: usize,
    doc_texts: HashMap<String, String>,
}

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

impl TextIndex {
    pub fn new(docs: Vec<Document>) -> Self {
        let doc_count = docs.len();
        let mut inverted: HashMap<String, PostingList> = HashMap::new();
        let mut doc_lengths: HashMap<String, usize> = HashMap::new();
        let mut doc_texts: HashMap<String, String> = HashMap::new();
        let mut total_length: usize = 0;

        for doc in &docs {
            let tokens = tokenize(&doc.text);
            doc_lengths.insert(doc.id.clone(), tokens.len());
            total_length += tokens.len();
            doc_texts.insert(doc.id.clone(), doc.text.clone());

            let mut term_positions: HashMap<String, Vec<usize>> = HashMap::new();
            for (pos, token) in tokens.iter().enumerate() {
                term_positions.entry(token.clone()).or_default().push(pos);
            }

            for (term, positions) in term_positions {
                let tf = positions.len();
                let posting = Posting {
                    doc_id: doc.id.clone(),
                    tf,
                    positions,
                };
                inverted.entry(term).or_insert_with(|| PostingList { entries: vec![] })
                    .entries.push(posting);
            }
        }

        let avg_dl = if doc_count > 0 {
            total_length as f32 / doc_count as f32
        } else {
            0.0
        };

        Self { inverted, doc_lengths, avg_dl, doc_count, doc_texts }
    }

    fn bm25_score(&self, tf: usize, df: usize, dl: usize) -> f32 {
        let idf = ((self.doc_count - df) as f32 + 0.5) / (df as f32 + 0.5);
        let idf = idf.ln_1p(); // ln(1 + idf) = log((N - df + 0.5) / (df + 0.5) + 1) variant
        let tf_norm = (tf as f32 * (K1 + 1.0))
            / (tf as f32 + K1 * (1.0 - B + B * (dl as f32 / self.avg_dl)));
        idf * tf_norm
    }
}

impl Search for TextIndex {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult> {
        let query_text = match query {
            Query::Text(t) => t,
            _ => return vec![],
        };
        let query_tokens = tokenize(query_text);
        if query_tokens.is_empty() {
            return vec![];
        }

        let mut scores: HashMap<String, f32> = HashMap::new();
        let mut match_positions: HashMap<String, Vec<usize>> = HashMap::new();

        for term in &query_tokens {
            if let Some(posting_list) = self.inverted.get(term) {
                let df = posting_list.entries.len();
                for posting in &posting_list.entries {
                    let dl = self.doc_lengths.get(&posting.doc_id).copied().unwrap_or(0);
                    let score = self.bm25_score(posting.tf, df, dl);
                    *scores.entry(posting.doc_id.clone()).or_insert(0.0) += score;
                    match_positions.entry(posting.doc_id.clone())
                        .or_default()
                        .extend_from_slice(&posting.positions);
                }
            }
        }

        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked.into_iter()
            .take(top_k)
            .map(|(doc_id, score)| {
                let text = self.doc_texts.get(&doc_id).cloned().unwrap_or_default();
                let snippet = make_snippet(&text, match_positions.get(&doc_id));
                SearchResult {
                    id: doc_id,
                    score,
                    source: SearchSource::Text,
                    snippet,
                }
            })
            .collect()
    }
}

fn make_snippet(text: &str, positions: Option<&Vec<usize>>) -> String {
    let positions = match positions {
        Some(p) if !p.is_empty() => p,
        _ => return truncate(text, 80),
    };
    let first = positions[0];
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let start = first.saturating_sub(3);
    let end = (first + 5).min(tokens.len());
    let snippet_tokens = &tokens[start..end];
    let mut snippet = snippet_tokens.join(" ");
    if start > 0 {
        snippet = format!("...{}", snippet);
    }
    if end < tokens.len() {
        snippet = format!("{}...", snippet);
    }
    snippet
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..s.char_indices().take(max_len).last().map(|(i, _)| i).unwrap_or(0)])
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib index::text`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/index/text.rs
git commit -m "feat: TextIndex with inverted index, BM25 scoring, and tokenizer"
```

---

### Task 4: GraphIndex — Adjacency List with BFS

**Files:**
- Create: `src/index/graph.rs`

- [ ] **Step 1: Write failing tests for GraphIndex**

```rust
// src/index/graph.rs

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_graph_bfs_1_hop() {
        let triples = vec![
            Triple { subject: "Rust".into(), predicate: "influenced_by".into(), object: "C++".into() },
            Triple { subject: "Rust".into(), predicate: "used_in".into(), object: "Linux".into() },
        ];
        let entity_docs = HashMap::new();
        let index = GraphIndex::new(triples, entity_docs);
        let results = index.search(&Query::Graph { entity: "Rust".into(), hops: 1 }, 10);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.score == 1.0));
    }

    #[test]
    fn test_graph_bfs_2_hops() {
        let triples = vec![
            Triple { subject: "Rust".into(), predicate: "influenced_by".into(), object: "C++".into() },
            Triple { subject: "C++".into(), predicate: "influenced_by".into(), object: "C".into() },
        ];
        let entity_docs = HashMap::new();
        let index = GraphIndex::new(triples, entity_docs);
        let results = index.search(&Query::Graph { entity: "Rust".into(), hops: 2 }, 10);
        assert_eq!(results.len(), 2);
        // C++ at hop 1 = score 1.0, C at hop 2 = score 0.5
        let cpp = results.iter().find(|r| r.id == "C++").unwrap();
        let c = results.iter().find(|r| r.id == "C").unwrap();
        assert!((cpp.score - 1.0).abs() < 1e-6);
        assert!((c.score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_graph_bidirectional() {
        let triples = vec![
            Triple { subject: "Rust".into(), predicate: "influenced_by".into(), object: "C++".into() },
        ];
        let entity_docs = HashMap::new();
        let index = GraphIndex::new(triples, entity_docs);
        // Search from C++ should find Rust via incoming edge
        let results = index.search(&Query::Graph { entity: "C++".into(), hops: 1 }, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "Rust");
    }

    #[test]
    fn test_graph_entity_docs() {
        let triples = vec![
            Triple { subject: "Rust".into(), predicate: "influenced_by".into(), object: "C++".into() },
        ];
        let mut entity_docs = HashMap::new();
        entity_docs.insert("Rust".into(), vec!["doc_1".into()]);
        entity_docs.insert("C++".into(), vec!["doc_2".into()]);
        let index = GraphIndex::new(triples, entity_docs);
        let results = index.search(&Query::Graph { entity: "Rust".into(), hops: 1 }, 10);
        // Results should include entity info with associated doc_ids in snippet
        assert!(results[0].snippet.contains("doc_"));
    }

    #[test]
    fn test_graph_nonexistent_entity() {
        let index = GraphIndex::new(vec![], HashMap::new());
        let results = index.search(&Query::Graph { entity: "NoSuch".into(), hops: 2 }, 10);
        assert!(results.is_empty());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib index::graph`
Expected: FAIL — `GraphIndex` and `Triple` not yet defined

- [ ] **Step 3: Implement GraphIndex with BFS and hop-decay scoring**

```rust
// src/index/graph.rs

use std::collections::{HashMap, HashSet, VecDeque};

use crate::types::{Query, Search, SearchSource, SearchResult, Triple};

pub struct GraphIndex {
    outgoing: HashMap<String, Vec<(String, String)>>,
    incoming: HashMap<String, Vec<(String, String)>>,
    entity_docs: HashMap<String, Vec<String>>,
}

impl GraphIndex {
    pub fn new(triples: Vec<Triple>, entity_docs: HashMap<String, Vec<String>>) -> Self {
        let mut outgoing: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut incoming: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for triple in triples {
            outgoing.entry(triple.subject.clone())
                .or_default()
                .push((triple.predicate.clone(), triple.object.clone()));
            incoming.entry(triple.object.clone())
                .or_default()
                .push((triple.predicate, triple.subject));
        }

        Self { outgoing, incoming, entity_docs }
    }

    pub fn get_entity_docs(&self, entity: &str) -> &[String] {
        self.entity_docs.get(entity).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

impl Search for GraphIndex {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult> {
        let (entity, hops) = match query {
            Query::Graph { entity, hops } => (entity.clone(), *hops),
            _ => return vec![],
        };

        if !self.outgoing.contains_key(&entity) && !self.incoming.contains_key(&entity) {
            return vec![];
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut results: Vec<SearchResult> = Vec::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        // Add all neighbors from both outgoing and incoming edges
        let mut enqueue_neighbors = |current: &str, hop: usize, results: &mut Vec<SearchResult>| {
            let neighbors: Vec<(String, String)> = self.outgoing.get(current)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .chain(self.incoming.get(current).cloned().unwrap_or_default())
                .collect();

            for (predicate, neighbor) in neighbors {
                if visited.insert(neighbor.clone()) {
                    let score = 1.0 / (2_usize.pow(hop as u32)) as f32;
                    let doc_ids = self.entity_docs.get(&neighbor)
                        .map(|v| v.join(", "))
                        .unwrap_or_default();
                    let snippet = if doc_ids.is_empty() {
                        format!("{} --{}--> {}", current, predicate, neighbor)
                    } else {
                        format!("{} --{}--> {} (docs: {})", current, predicate, neighbor, doc_ids)
                    };
                    results.push(SearchResult {
                        id: neighbor,
                        score,
                        source: SearchSource::Graph,
                        snippet,
                    });
                    if hop < hops {
                        queue.push_back((neighbor.clone(), hop + 1));
                    }
                }
            }
        };

        visited.insert(entity.clone());
        enqueue_neighbors(&entity, 1, &mut results);

        while let Some((current, hop)) = queue.pop_front() {
            enqueue_neighbors(&current, hop, &mut results);
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib index::graph`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/index/graph.rs
git commit -m "feat: GraphIndex with BFS traversal and hop-decay scoring"
```

---

### Task 5: RRF Fusion

**Files:**
- Modify: `src/fusion.rs`

- [ ] **Step 1: Write failing tests for RRF**

```rust
// src/fusion.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SearchSource;

    #[test]
    fn test_rrf_two_lists() {
        let r1 = vec![
            SearchResult { id: "a".into(), score: 0.9, source: SearchSource::Vector, snippet: String::new() },
            SearchResult { id: "b".into(), score: 0.8, source: SearchSource::Vector, snippet: String::new() },
            SearchResult { id: "c".into(), score: 0.7, source: SearchSource::Vector, snippet: String::new() },
        ];
        let r2 = vec![
            SearchResult { id: "b".into(), score: 10.0, source: SearchSource::Text, snippet: String::new() },
            SearchResult { id: "a".into(), score: 9.0, source: SearchSource::Text, snippet: String::new() },
            SearchResult { id: "d".into(), score: 8.0, source: SearchSource::Text, snippet: String::new() },
        ];
        let fused = reciprocal_rank_fusion(&[r1, r2], 60);
        // "a" rank 0 + rank 1 = 1/60 + 1/61 = 0.0328
        // "b" rank 1 + rank 0 = 1/61 + 1/60 = 0.0328
        // Both a and b should rank above c and d
        assert_eq!(fused.len(), 4);
        assert!(fused[0].score > fused[2].score);
    }

    #[test]
    fn test_rrf_single_list() {
        let r1 = vec![
            SearchResult { id: "a".into(), score: 1.0, source: SearchSource::Vector, snippet: String::new() },
            SearchResult { id: "b".into(), score: 0.5, source: SearchSource::Vector, snippet: String::new() },
        ];
        let fused = reciprocal_rank_fusion(&[r1], 60);
        assert_eq!(fused[0].id, "a");
        assert!(fused[0].score > fused[1].score);
    }

    #[test]
    fn test_rrf_empty() {
        let fused = reciprocal_rank_fusion(&[], 60);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_deduplication() {
        let r1 = vec![
            SearchResult { id: "a".into(), score: 1.0, source: SearchSource::Vector, snippet: String::new() },
        ];
        let r2 = vec![
            SearchResult { id: "a".into(), score: 1.0, source: SearchSource::Text, snippet: String::new() },
        ];
        let fused = reciprocal_rank_fusion(&[r1, r2], 60);
        assert_eq!(fused.len(), 1);
        // Score should be sum of both ranks: 1/60 + 1/60
        assert!((fused[0].score - (1.0/60.0 + 1.0/60.0)).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib fusion`
Expected: FAIL — `reciprocal_rank_fusion` not yet defined

- [ ] **Step 3: Implement RRF**

```rust
// src/fusion.rs

use std::collections::HashMap;

use crate::types::{SearchResult, SearchSource};

pub fn reciprocal_rank_fusion(result_sets: &[Vec<SearchResult>], k: usize) -> Vec<SearchResult> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for results in result_sets {
        for (rank, result) in results.iter().enumerate() {
            let rrf_score = 1.0 / (k as f32 + rank as f32 + 1.0);
            *scores.entry(result.id.clone()).or_insert(0.0) += rrf_score;
        }
    }

    let mut fused: Vec<(String, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    fused.into_iter()
        .map(|(id, score)| SearchResult {
            id,
            score,
            source: SearchSource::Vector, // fused results don't have a single source
            snippet: String::new(),
        })
        .collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib fusion`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fusion.rs
git commit -m "feat: RRF (Reciprocal Rank Fusion) implementation"
```

---

### Task 6: JSON Data Loader

**Files:**
- Modify: `src/loader.rs`

- [ ] **Step 1: Write failing tests for loader**

```rust
// src/loader.rs

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_json(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_load_documents() {
        let json = r#"[{"id":"d1","text":"hello world","vector":[1.0,0.0]}]"#;
        let f = write_temp_json(json);
        let docs = load_documents(f.path()).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "d1");
        assert_eq!(docs[0].vector, vec![1.0, 0.0]);
    }

    #[test]
    fn test_load_documents_dimension_mismatch() {
        let json = r#"[{"id":"d1","text":"a","vector":[1.0,0.0]},{"id":"d2","text":"b","vector":[1.0]}]"#;
        let f = write_temp_json(json);
        let result = load_documents(f.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_triples() {
        let json = r#"[{"subject":"Rust","predicate":"influenced_by","object":"C++"}]"#;
        let f = write_temp_json(json);
        let triples = load_triples(f.path()).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, "Rust");
    }

    #[test]
    fn test_load_entity_docs() {
        let json = r#"{"Rust":["d1","d2"],"C++":["d3"]}"#;
        let f = write_temp_json(json);
        let map = load_entity_docs(f.path()).unwrap();
        assert_eq!(map.get("Rust").unwrap().len(), 2);
    }
}
```

- [ ] **Step 2: Add tempfile dev-dependency to Cargo.toml**

Add to `Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --lib loader`
Expected: FAIL — `load_documents`, `load_triples`, `load_entity_docs` not yet defined

- [ ] **Step 4: Implement loader functions**

```rust
// src/loader.rs

use std::collections::HashMap;
use std::path::Path;

use crate::types::{Document, Triple};

pub fn load_documents(path: &Path) -> Result<Vec<Document>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    let docs: Vec<Document> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse documents: {}", e))?;

    if docs.len() > 1 {
        let dim = docs[0].vector.len();
        for (i, doc) in docs.iter().enumerate().skip(1) {
            if doc.vector.len() != dim {
                return Err(format!(
                    "Vector dimension mismatch: doc {} has {} dims, expected {}",
                    doc.id, doc.vector.len(), dim
                ));
            }
        }
    }

    Ok(docs)
}

pub fn load_triples(path: &Path) -> Result<Vec<Triple>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse triples: {}", e))
}

pub fn load_entity_docs(path: &Path) -> Result<HashMap<String, Vec<String>>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse entity_docs: {}", e))
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib loader`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/loader.rs Cargo.toml
git commit -m "feat: JSON data loader with dimension validation"
```

---

### Task 7: Engine Orchestration

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Write failing tests for Engine**

```rust
// src/lib.rs (test module at bottom)

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_engine() -> Engine {
        let docs = vec![
            Document { id: "d1".into(), text: "Rust is a systems programming language".into(), vector: vec![1.0, 0.0] },
            Document { id: "d2".into(), text: "Python is a scripting language".into(), vector: vec![0.0, 1.0] },
            Document { id: "d3".into(), text: "Rust programming is fun".into(), vector: vec![0.9, 0.1] },
        ];
        let triples = vec![
            Triple { subject: "Rust".into(), predicate: "influenced_by".into(), object: "C++".into() },
            Triple { subject: "C++".into(), predicate: "influenced_by".into(), object: "C".into() },
        ];
        let mut entity_docs = HashMap::new();
        entity_docs.insert("Rust".into(), vec!["d1".into(), "d3".into()]);
        entity_docs.insert("C++".into(), vec!["d2".into()]);
        Engine::new(docs, triples, entity_docs)
    }

    #[test]
    fn test_engine_vector_search() {
        let engine = sample_engine();
        let results = engine.search(&Query::Vector(vec![1.0, 0.0]), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "d1");
    }

    #[test]
    fn test_engine_text_search() {
        let engine = sample_engine();
        let results = engine.search(&Query::Text("rust programming".into()), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "d3");
    }

    #[test]
    fn test_engine_graph_search() {
        let engine = sample_engine();
        let results = engine.search(&Query::Graph { entity: "Rust".into(), hops: 2 }, 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_engine_hybrid_search() {
        let engine = sample_engine();
        let results = engine.search(
            &Query::Hybrid {
                text: "rust".into(),
                vector: vec![1.0, 0.0],
                entity: None,
            },
            5,
        );
        assert!(!results.is_empty());
    }

    #[test]
    fn test_engine_hybrid_with_entity_filter() {
        let engine = sample_engine();
        let results = engine.search(
            &Query::Hybrid {
                text: "rust".into(),
                vector: vec![1.0, 0.0],
                entity: Some("Rust".into()),
            },
            5,
        );
        // Should only return docs linked to "Rust" entity: d1, d3
        assert!(results.iter().all(|r| r.id == "d1" || r.id == "d3"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib`
Expected: FAIL — `Engine` not yet defined

- [ ] **Step 3: Implement Engine**

```rust
// src/lib.rs

pub mod types;
pub mod index;
pub mod fusion;
pub mod loader;

pub use types::{Document, Query, Search, SearchSource, SearchResult, Triple};
pub use index::vector::VectorIndex;
pub use index::text::TextIndex;
pub use index::graph::GraphIndex;
pub use fusion::reciprocal_rank_fusion;

use std::collections::HashMap;

pub struct Engine {
    vector: VectorIndex,
    text: TextIndex,
    graph: GraphIndex,
}

impl Engine {
    pub fn new(
        docs: Vec<Document>,
        triples: Vec<Triple>,
        entity_docs: HashMap<String, Vec<String>>,
    ) -> Self {
        let vector = VectorIndex::new(docs.clone());
        let text = TextIndex::new(docs);
        let graph = GraphIndex::new(triples, entity_docs);
        Self { vector, text, graph }
    }

    pub fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult> {
        match query {
            Query::Vector(_) => self.vector.search(query, top_k),
            Query::Text(_) => self.text.search(query, top_k),
            Query::Graph { .. } => self.graph.search(query, top_k),
            Query::Hybrid { text, vector, entity } => {
                let candidate_ids: Option<Vec<String>> = entity.as_ref().map(|e| {
                    let mut ids: Vec<String> = Vec::new();
                    // Collect docs for the entity itself
                    ids.extend(self.graph.get_entity_docs(e).iter().cloned());
                    // Collect docs for 1-hop neighbors
                    let graph_results = self.graph.search(
                        &Query::Graph { entity: e.clone(), hops: 1 },
                        100,
                    );
                    for r in &graph_results {
                        ids.extend(self.graph.get_entity_docs(&r.id).iter().cloned());
                    }
                    ids.sort();
                    ids.dedup();
                    ids
                });

                let vector_results = self.vector.search(&Query::Vector(vector.clone()), top_k * 2);
                let text_results = self.text.search(&Query::Text(text.clone()), top_k * 2);

                let filtered_vector: Vec<SearchResult> = match &candidate_ids {
                    Some(ids) => vector_results.into_iter()
                        .filter(|r| ids.contains(&r.id))
                        .collect(),
                    None => vector_results,
                };
                let filtered_text: Vec<SearchResult> = match &candidate_ids {
                    Some(ids) => text_results.into_iter()
                        .filter(|r| ids.contains(&r.id))
                        .collect(),
                    None => text_results,
                };

                let mut fused = reciprocal_rank_fusion(&[filtered_vector, filtered_text], 60);
                fused.truncate(top_k);
                fused
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // ... tests from step 1
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib`
Expected: All tests PASS (including previous module tests)

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs
git commit -m "feat: Engine orchestration with hybrid search and entity filtering"
```

---

### Task 8: Sample Data

**Files:**
- Create: `data/documents.json`
- Create: `data/triples.json`
- Create: `data/entity_docs.json`

- [ ] **Step 1: Create data/documents.json**

A set of 12 documents about programming languages with 4-dimensional vectors:

```json
[
  {"id": "doc_001", "text": "Rust is a systems programming language focused on safety and performance", "vector": [0.9, 0.1, 0.0, 0.0]},
  {"id": "doc_002", "text": "Python is a high-level scripting language popular for data science", "vector": [0.1, 0.9, 0.0, 0.1]},
  {"id": "doc_003", "text": "C++ is a systems language with manual memory management and high performance", "vector": [0.8, 0.0, 0.2, 0.0]},
  {"id": "doc_004", "text": "JavaScript is the language of the web used for frontend and backend development", "vector": [0.0, 0.7, 0.0, 0.8]},
  {"id": "doc_005", "text": "Go is a compiled language designed for simplicity and concurrency", "vector": [0.6, 0.2, 0.5, 0.0]},
  {"id": "doc_006", "text": "TypeScript adds static typing to JavaScript for large scale applications", "vector": [0.0, 0.6, 0.1, 0.9]},
  {"id": "doc_007", "text": "Rust ownership model prevents memory bugs without garbage collection", "vector": [0.95, 0.0, 0.0, 0.05]},
  {"id": "doc_008", "text": "Java is a compiled language running on the JVM used in enterprise systems", "vector": [0.3, 0.5, 0.6, 0.0]},
  {"id": "doc_009", "text": "Haskell is a purely functional programming language with strong type system", "vector": [0.5, 0.3, 0.0, 0.4]},
  {"id": "doc_010", "text": "C is the foundational systems language that influenced many modern languages", "vector": [0.85, 0.0, 0.15, 0.0]},
  {"id": "doc_011", "text": "Swift is Apples language for iOS and macOS development with safety features", "vector": [0.7, 0.1, 0.3, 0.1]},
  {"id": "doc_012", "text": "Kotlin is a modern language on the JVM used for Android development", "vector": [0.2, 0.5, 0.5, 0.1]}
]
```

- [ ] **Step 2: Create data/triples.json**

```json
[
  {"subject": "Rust", "predicate": "influenced_by", "object": "C++"},
  {"subject": "Rust", "predicate": "influenced_by", "object": "Haskell"},
  {"subject": "Rust", "predicate": "used_in", "object": "Linux"},
  {"subject": "Rust", "predicate": "used_in", "object": "WebAssembly"},
  {"subject": "C++", "predicate": "influenced_by", "object": "C"},
  {"subject": "C++", "predicate": "used_in", "object": "GameEngine"},
  {"subject": "C", "predicate": "used_in", "object": "Linux"},
  {"subject": "C", "predicate": "used_in", "object": "Embedded"},
  {"subject": "Python", "predicate": "used_in", "object": "DataScience"},
  {"subject": "Python", "predicate": "used_in", "object": "MachineLearning"},
  {"subject": "JavaScript", "predicate": "used_in", "object": "WebFrontend"},
  {"subject": "JavaScript", "predicate": "used_in", "object": "WebBackend"},
  {"subject": "TypeScript", "predicate": "superset_of", "object": "JavaScript"},
  {"subject": "TypeScript", "predicate": "used_in", "object": "WebFrontend"},
  {"subject": "Go", "predicate": "used_in", "object": "CloudInfrastructure"},
  {"subject": "Go", "predicate": "used_in", "object": "DevOps"},
  {"subject": "Java", "predicate": "used_in", "object": "Enterprise"},
  {"subject": "Java", "predicate": "used_in", "object": "Android"},
  {"subject": "Kotlin", "predicate": "used_in", "object": "Android"},
  {"subject": "Kotlin", "predicate": "runs_on", "object": "JVM"},
  {"subject": "Java", "predicate": "runs_on", "object": "JVM"},
  {"subject": "Swift", "predicate": "used_in", "object": "iOS"},
  {"subject": "Swift", "predicate": "used_in", "object": "macOS"},
  {"subject": "Haskell", "predicate": "influenced", "object": "Rust"},
  {"subject": "Linux", "predicate": "written_in", "object": "C"},
  {"subject": "WebAssembly", "predicate": "supports", "object": "Rust"}
]
```

- [ ] **Step 3: Create data/entity_docs.json**

```json
{
  "Rust": ["doc_001", "doc_007"],
  "C++": ["doc_003"],
  "Python": ["doc_002"],
  "JavaScript": ["doc_004"],
  "TypeScript": ["doc_006"],
  "Go": ["doc_005"],
  "Java": ["doc_008"],
  "Kotlin": ["doc_012"],
  "Swift": ["doc_011"],
  "Haskell": ["doc_009"],
  "C": ["doc_010"]
}
```

- [ ] **Step 4: Validate JSON files**

Run: `for f in data/*.json; do python3 -m json.tool "$f" > /dev/null && echo "$f OK" || echo "$f INVALID"; done`
Expected: All 3 files print "OK"

- [ ] **Step 5: Commit**

```bash
git add data/
git commit -m "feat: sample data for demo (12 docs, 26 triples, 11 entity mappings)"
```

---

### Task 9: CLI Entry Point

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Write the CLI with clap derive macros**

```rust
// src/main.rs

use std::collections::HashMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use mini_hybrid_retrieval::{Document, Engine, Query, Triple};

#[derive(Parser)]
#[command(name = "mini-hybrid", version = "0.1.0")]
#[command(about = "A mini multimodal hybrid retrieval engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build index and run queries
    Index {
        /// Path to documents JSON file
        #[arg(long)]
        docs: PathBuf,
        /// Path to triples JSON file
        #[arg(long)]
        triples: PathBuf,
        /// Path to entity-docs mapping JSON file
        #[arg(long)]
        entity_docs: Option<PathBuf>,
        /// Query subcommand
        #[command(subcommand)]
        query: QueryCommands,
    },
}

#[derive(Subcommand)]
enum QueryCommands {
    /// Vector similarity search
    Vector {
        /// Query vector as comma-separated floats, e.g. "0.9,0.1,0.0,0.0"
        #[arg(long)]
        query_vector: String,
        /// Number of results
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Full-text search
    Text {
        /// Query text
        #[arg(long)]
        query: String,
        /// Number of results
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Knowledge graph search
    Graph {
        /// Seed entity name
        #[arg(long)]
        entity: String,
        /// Number of hops
        #[arg(long, default_value_t = 2)]
        hops: usize,
    },
    /// Hybrid search combining vector, text, and optional graph filter
    Hybrid {
        /// Query text
        #[arg(long)]
        query: String,
        /// Query vector as comma-separated floats
        #[arg(long)]
        query_vector: String,
        /// Optional entity name for graph filtering
        #[arg(long)]
        entity: Option<String>,
        /// Number of results
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
}

fn parse_vector(s: &str) -> Vec<f32> {
    s.split(',')
        .map(|v| v.trim().parse::<f32>().expect("Invalid float in vector"))
        .collect()
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { docs, triples, entity_docs, query } => {
            let documents = mini_hybrid_retrieval::loader::load_documents(&docs)
                .unwrap_or_else(|e| { eprintln!("Error loading documents: {}", e); std::process::exit(1); });
            let triples_data = mini_hybrid_retrieval::loader::load_triples(&triples)
                .unwrap_or_else(|e| { eprintln!("Error loading triples: {}", e); std::process::exit(1); });
            let entity_docs_data: HashMap<String, Vec<String>> = match entity_docs {
                Some(path) => mini_hybrid_retrieval::loader::load_entity_docs(&path)
                    .unwrap_or_else(|e| { eprintln!("Error loading entity_docs: {}", e); std::process::exit(1); }),
                None => HashMap::new(),
            };

            let engine = Engine::new(documents, triples_data, entity_docs_data);

            let results = match query {
                QueryCommands::Vector { query_vector, top_k } => {
                    let vec = parse_vector(&query_vector);
                    engine.search(&Query::Vector(vec), top_k)
                }
                QueryCommands::Text { query, top_k } => {
                    engine.search(&Query::Text(query), top_k)
                }
                QueryCommands::Graph { entity, hops } => {
                    engine.search(&Query::Graph { entity, hops }, 100)
                }
                QueryCommands::Hybrid { query, query_vector, entity, top_k } => {
                    let vec = parse_vector(&query_vector);
                    engine.search(&Query::Hybrid { text: query, vector: vec, entity }, top_k)
                }
            };

            for r in &results {
                println!("[{}] {} ({:.4}): {}", 
                    match r.source {
                        mini_hybrid_retrieval::SearchSource::Vector => "VEC",
                        mini_hybrid_retrieval::SearchSource::Text => "TXT",
                        mini_hybrid_retrieval::SearchSource::Graph => "GRP",
                    },
                    r.id, r.score, r.snippet
                );
            }
        }
    }
}
```

- [ ] **Step 2: Build and verify CLI compiles**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 3: Test CLI with sample data**

```bash
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json text --query "rust programming" --top-k 3
```
Expected: Prints ranked text search results

```bash
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json vector --query-vector "0.9,0.1,0.0,0.0" --top-k 3
```
Expected: Prints ranked vector search results

```bash
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json graph --entity "Rust" --hops 2
```
Expected: Prints graph neighbor results

```bash
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json hybrid --query "rust programming" --query-vector "0.9,0.1,0.0,0.0" --entity "Rust" --top-k 3
```
Expected: Prints fused results filtered by Rust entity

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: CLI with index and search subcommands"
```

---

### Task 10: Integration Tests

**Files:**
- Create: `tests/integration.rs`

- [ ] **Step 1: Write integration tests**

```rust
// tests/integration.rs

use std::collections::HashMap;
use mini_hybrid_retrieval::{Document, Engine, Query, SearchSource, Triple};

fn build_test_engine() -> Engine {
    let docs = vec![
        Document { id: "doc_001".into(), text: "Rust is a systems programming language focused on safety and performance".into(), vector: vec![0.9, 0.1, 0.0, 0.0] },
        Document { id: "doc_002".into(), text: "Python is a high-level scripting language popular for data science".into(), vector: vec![0.1, 0.9, 0.0, 0.1] },
        Document { id: "doc_003".into(), text: "C++ is a systems language with manual memory management and high performance".into(), vector: vec![0.8, 0.0, 0.2, 0.0] },
        Document { id: "doc_004".into(), text: "JavaScript is the language of the web used for frontend and backend development".into(), vector: vec![0.0, 0.7, 0.0, 0.8] },
        Document { id: "doc_007".into(), text: "Rust ownership model prevents memory bugs without garbage collection".into(), vector: vec![0.95, 0.0, 0.0, 0.05] },
    ];
    let triples = vec![
        Triple { subject: "Rust".into(), predicate: "influenced_by".into(), object: "C++".into() },
        Triple { subject: "C++".into(), predicate: "influenced_by".into(), object: "C".into() },
        Triple { subject: "Python".into(), predicate: "used_in".into(), object: "DataScience".into() },
    ];
    let mut entity_docs = HashMap::new();
    entity_docs.insert("Rust".into(), vec!["doc_001".into(), "doc_007".into()]);
    entity_docs.insert("C++".into(), vec!["doc_003".into()]);
    Engine::new(docs, triples, entity_docs)
}

#[test]
fn test_vector_search_finds_similar() {
    let engine = build_test_engine();
    let results = engine.search(&Query::Vector(vec![0.9, 0.1, 0.0, 0.0]), 3);
    assert!(!results.is_empty());
    assert_eq!(results[0].source, SearchSource::Vector);
    // doc_001 and doc_007 have vectors closest to [0.9, 0.1, 0.0, 0.0]
    assert!(results[0].id == "doc_001" || results[0].id == "doc_007");
}

#[test]
fn test_text_search_finds_relevant() {
    let engine = build_test_engine();
    let results = engine.search(&Query::Text("rust memory".into()), 3);
    assert!(!results.is_empty());
    assert_eq!(results[0].source, SearchSource::Text);
}

#[test]
fn test_graph_search_traverses_hops() {
    let engine = build_test_engine();
    let results = engine.search(&Query::Graph { entity: "Rust".into(), hops: 2 }, 10);
    assert!(!results.is_empty());
    // Should find C++ (1 hop) and C (2 hops)
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"C++"));
    assert!(ids.contains(&"C"));
}

#[test]
fn test_hybrid_search_without_entity() {
    let engine = build_test_engine();
    let results = engine.search(
        &Query::Hybrid {
            text: "rust".into(),
            vector: vec![0.9, 0.1, 0.0, 0.0],
            entity: None,
        },
        5,
    );
    assert!(!results.is_empty());
}

#[test]
fn test_hybrid_search_with_entity_filter() {
    let engine = build_test_engine();
    let results = engine.search(
        &Query::Hybrid {
            text: "rust".into(),
            vector: vec![0.9, 0.1, 0.0, 0.0],
            entity: Some("Rust".into()),
        },
        5,
    );
    // Only docs linked to "Rust" entity: doc_001, doc_007
    assert!(results.iter().all(|r| r.id == "doc_001" || r.id == "doc_007"));
}

#[test]
fn test_empty_query_returns_no_results() {
    let engine = build_test_engine();
    let results = engine.search(&Query::Text(String::new()), 5);
    assert!(results.is_empty());
}
```

- [ ] **Step 2: Run integration tests**

Run: `cargo test`
Expected: All tests PASS (unit + integration)

- [ ] **Step 3: Commit**

```bash
git add tests/integration.rs
git commit -m "feat: integration tests for end-to-end search pipeline"
```

---

### Task 11: Final Cleanup and Verification

**Files:**
- Modify: `src/index/mod.rs` (ensure proper re-exports)
- Review all files for consistency

- [ ] **Step 1: Verify src/index/mod.rs has proper re-exports**

```rust
// src/index/mod.rs

pub mod vector;
pub mod text;
pub mod graph;
```

- [ ] **Step 2: Run all tests**

Run: `cargo test`
Expected: All tests PASS

- [ ] **Step 3: Run clippy for lint checks**

Run: `cargo clippy -- -D warnings`
Expected: No warnings

- [ ] **Step 4: Run CLI end-to-end with all search modes**

```bash
# Text search
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json text --query "systems programming language" --top-k 3

# Vector search
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json vector --query-vector "0.9,0.1,0.0,0.0" --top-k 3

# Graph search
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json graph --entity "Rust" --hops 2

# Hybrid search
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json hybrid --query "systems programming" --query-vector "0.9,0.1,0.0,0.0" --top-k 3
```
Expected: All commands produce meaningful ranked results

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```
