use crate::types::{Document, Query, Search, SearchSource, SearchResult};

pub struct TextIndex;

impl TextIndex {
    pub fn new(_docs: Vec<Document>) -> Self {
        Self
    }
}

impl Search for TextIndex {
    fn search(&self, _query: &Query, _top_k: usize) -> Vec<SearchResult> {
        vec![]
    }
}
