# mini-hybrid-retrieval

A minimal multimodal hybrid retrieval engine in Rust, implementing vector search, full-text search, and knowledge graph search from scratch (no faiss/tantivy). Designed for learning and small-data demos at hundred-scale.

## Features

- **Vector Search** — Brute-force KNN with cosine similarity
- **Full-Text Search** — Inverted index with BM25 scoring
- **Graph Search** — Bidirectional adjacency list with BFS traversal and hop-decay scoring
- **Hybrid Search** — RRF (Reciprocal Rank Fusion) combining vector + text results, with optional graph-based entity filtering

## Usage

```bash
# Build
cargo build

# Run tests
cargo test

# Text search
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json text --query "rust programming" --top-k 3

# Vector search
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json vector --query-vector "0.9,0.1,0.0,0.0" --top-k 3

# Graph search
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json graph --entity "Rust" --hops 2

# Hybrid search (vector + text + optional entity filter)
cargo run -- index --docs data/documents.json --triples data/triples.json --entity-docs data/entity_docs.json hybrid --query "systems programming" --query-vector "0.9,0.1,0.0,0.0" --entity "Rust" --top-k 3
```

## Release(CLI)
```bash
# 编译 release 版本
cargo build --release

# 直接运行CLI
./target/release/mini-hybrid-retrieval index \
    --docs data/documents.json \
    --triples data/triples.json \
    --entity-docs data/entity_docs.json \
    text --query "rust" --top-k 3
```


## Data Format

**documents.json** — Documents with pre-computed embeddings:
```json
[{"id": "doc_001", "text": "Rust is a systems programming language", "vector": [0.9, 0.1, 0.0, 0.0]}]
```

**triples.json** — Knowledge graph as (subject, predicate, object) triples:
```json
[{"subject": "Rust", "predicate": "influenced_by", "object": "C++"}]
```

**entity_docs.json** — Entity-to-document mappings (optional):
```json
{"Rust": ["doc_001", "doc_007"]}
```

## Architecture

```
Engine (orchestration)
├── VectorIndex   — brute-force KNN with cosine similarity
├── TextIndex     — inverted index with BM25 scoring
├── GraphIndex    — adjacency list with BFS traversal
└── RRF Fusion    — reciprocal rank fusion
```

Each index implements a unified `Search` trait. The Engine orchestrates queries: single-modal queries dispatch directly, hybrid queries fuse vector + text results via RRF with optional graph-based entity filtering.

## V1 Scope

- Pure in-memory, no persistence
- Whitespace tokenization (no Chinese support)
- Brute-force vector search (no ANN)
- CLI only (no HTTP server)
