# Mini Hybrid Retrieval V1 Design

## Overview

A minimal multimodal hybrid retrieval engine in Rust, implementing vector search, full-text search, and knowledge graph search from scratch (no faiss/tantivy). Designed for learning and small-data demos at hundred-scale.

## Architecture: Modular Layered

```
Engine (orchestration)
├── VectorIndex   — brute-force KNN with cosine similarity
├── TextIndex     — inverted index with BM25 scoring
├── GraphIndex    — adjacency list with BFS traversal
└── RRF Fusion    — reciprocal rank fusion of vector + text results
```

Each index is an independent module implementing a unified `Search` trait. The Engine holds all three indexes and orchestrates queries.

## Core Types

```rust
// types.rs

struct Document {
    id: String,
    text: String,
    vector: Vec<f32>,
}

struct Triple {
    subject: String,
    predicate: String,
    object: String,
}

struct SearchResult {
    id: String,
    score: f32,
    source: SearchSource, // Vector | Text | Graph
    snippet: String,
}

trait Search {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult>;
}

enum Query {
    Vector(Vec<f32>),
    Text(String),
    Graph { entity: String, hops: usize },
    Hybrid { text: String, vector: Vec<f32>, entity: Option<String> },
}
```

- `Document` holds both text and vector — acceptable at hundred-scale
- `Triple` uses string entity names for readability
- `Search` trait unifies the three indexes for Engine orchestration
- `Query` enum supports single-modal and hybrid queries

## Index Implementations

### Vector Index — Brute-force KNN

```rust
struct VectorIndex {
    docs: Vec<Document>,
    dim: usize,
}
```

- O(n) full scan, compute cosine similarity against all documents
- Sort by similarity, return top_k
- No ANN (HNSW/IVF) — that's V2 territory

### Text Index — Inverted Index + BM25

```rust
struct TextIndex {
    inverted: HashMap<String, PostingList>,
    doc_lengths: HashMap<String, usize>,
    avg_dl: f32,
    doc_count: usize,
}

struct PostingList {
    entries: Vec<Posting>,
}

struct Posting {
    doc_id: String,
    tf: usize,
    positions: Vec<usize>,
}
```

- Tokenizer: whitespace + punctuation splitting, no jieba or stemming
- Scoring: BM25 with k1=1.2, b=0.75
- Snippet: extract context around matched positions

### Graph Index — Adjacency List + BFS

```rust
struct GraphIndex {
    outgoing: HashMap<String, Vec<(String, String)>>,  // subject -> [(predicate, object)]
    incoming: HashMap<String, Vec<(String, String)>>,   // object -> [(predicate, subject)]
    entity_docs: HashMap<String, Vec<String>>,          // entity -> [doc_id]
}
```

- Bidirectional adjacency for efficient traversal in both directions
- Query: BFS from seed entity, N hops
- Scoring: decay by hop count — 1 hop = 1.0, 2 hops = 0.5, 3 hops = 0.25
- `entity_docs` maps entities to related documents, aligning graph results with vector/text

## Fusion Strategy

### RRF (Reciprocal Rank Fusion)

```
score(d) = Σ 1/(k + rank_i(d)),  k=60
```

- Uses rank only, no need to normalize raw scores across different retrieval methods
- k=60 is the standard default

### Unified ID Space Handling

Vector and text searches return `doc_id`; graph searches return entity names. They live in different ID spaces.

Resolution:
- Vector + text results undergo RRF fusion (same doc_id space)
- Graph results serve as a filter: when `entity` is specified in a hybrid query, find related doc_ids first, then restrict vector/text search to that candidate set
- Graph results are attached as supplementary information on matched documents

### Engine Orchestration

```rust
struct Engine {
    vector: VectorIndex,
    text: TextIndex,
    graph: GraphIndex,
}
```

- Single-modal queries dispatch directly to the corresponding index
- Hybrid queries: graph filters candidates (if entity specified), vector + text search, RRF fusion
- No async needed at hundred-scale — sequential calls suffice

## Data Loading

### JSON Data Format

**documents.json**
```json
[
  {"id": "doc_001", "text": "Rust is a systems programming language...", "vector": [0.12, -0.34, 0.56]}
]
```

**triples.json**
```json
[
  {"subject": "Rust", "predicate": "influenced_by", "object": "C++"}
]
```

**entity_docs.json** (optional)
```json
{"Rust": ["doc_001", "doc_003"], "C++": ["doc_002"]}
```

### Loader

```rust
fn load_documents(path: &Path) -> Result<Vec<Document>>;
fn load_triples(path: &Path) -> Result<Vec<Triple>>;
fn load_entity_docs(path: &Path) -> Result<HashMap<String, Vec<String>>>;
```

- `serde_json` for deserialization
- Vector dimension inferred from first document, validated for consistency

## CLI

```
# Build index
mini-hybrid index --docs documents.json --triples triples.json --entity-docs entity_docs.json

# Vector search
mini-hybrid search vector --query-vector [0.1,-0.2,0.3] --top-k 5

# Full-text search
mini-hybrid search text --query "rust programming" --top-k 5

# Graph search
mini-hybrid search graph --entity "Rust" --hops 2

# Hybrid search
mini-hybrid search hybrid --query "rust programming" --query-vector [0.1,-0.2,0.3] --entity "Rust" --top-k 5
```

- `clap` with derive macros for CLI parsing
- `index` loads data and builds indexes in memory; `search` executes queries
- No persistence in V1 — reload on every run

## Project Structure

```
mini-hybrid-retrieval/
├── Cargo.toml
├── data/
│   ├── documents.json
│   ├── triples.json
│   └── entity_docs.json
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── types.rs
│   ├── index/
│   │   ├── mod.rs
│   │   ├── vector.rs
│   │   ├── text.rs
│   │   └── graph.rs
│   ├── fusion.rs
│   └── loader.rs
└── tests/
    └── integration.rs
```

## Dependencies

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
```

Only 3 dependencies: serialization + CLI. No retrieval or math libraries.

## Testing

- **Unit tests**: `#[cfg(test)]` in each module
  - `vector.rs`: cosine_similarity correctness, top_k ordering
  - `text.rs`: tokenization, BM25 scoring, inverted index construction
  - `graph.rs`: BFS hop traversal, bidirectional adjacency
  - `fusion.rs`: RRF score computation and ordering
- **Integration tests**: `tests/integration.rs`, full pipeline with `data/` sample data
- **Sample data**: 10-20 documents, ~30 triples in `data/`

## V1 Scope Boundaries (What We Don't Do)

- No persistence — reload on every run
- No incremental indexing — full build only
- No Chinese tokenization — whitespace splitting only
- No ANN — brute-force scan only
- No HTTP server — CLI only
- No query caching

## Appendix: Open-Source References

| Concept | Reference | What We Borrowed |
|---------|-----------|-----------------|
| Document with text + vector | [Qdrant](https://github.com/qdrant/qdrant) | Point structure: payload + vector together |
| Triple representation | [Apache Jena](https://github.com/apache/jena) | Standard (subject, predicate, object) model |
| Unified Search trait | [Tantivy](https://github.com/quickwit-oss/tantivy) | Searcher abstraction for composable retrieval |
| Composable indexes | [LanceDB](https://github.com/lancedb/lancedb) | Vector + full-text as pluggable, composable indexes |
| RRF fusion | [Cohere Rerank](https://docs.cohere.com/docs/reranking), [Haystack](https://github.com/deepset-ai/haystack) | Reciprocal Rank Fusion as lightweight hybrid strategy |
| Multi-index orchestration | [Milvus](https://github.com/milvus-io/milvus), [Weaviate](https://github.com/weaviate/weaviate) | Modular architecture with parallel index search, simplified for single-node in-memory |
