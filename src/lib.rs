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
                    ids.extend(self.graph.get_entity_docs(e).iter().cloned());
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
        assert!(results.iter().any(|r| r.id == "d3"));
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
        // Entity filter includes Rust's docs (d1, d3) + 1-hop neighbor C++'s docs (d2)
        // So d1, d2, d3 are all valid candidates
        assert!(!results.is_empty());
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"d1"), "d1 (Rust doc) should be in results");
    }
}
