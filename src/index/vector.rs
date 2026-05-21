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

impl Search for VectorIndex {
    fn search(&self, _query: &Query, _top_k: usize) -> Vec<SearchResult> {
        vec![]
    }
}
