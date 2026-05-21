use std::collections::HashMap;
use crate::types::{Query, Search, SearchSource, SearchResult, Triple};

pub struct GraphIndex;

impl GraphIndex {
    pub fn new(_triples: Vec<Triple>, _entity_docs: HashMap<String, Vec<String>>) -> Self {
        Self
    }

    pub fn get_entity_docs(&self, _entity: &str) -> &[String] {
        &[]
    }
}

impl Search for GraphIndex {
    fn search(&self, _query: &Query, _top_k: usize) -> Vec<SearchResult> {
        vec![]
    }
}
