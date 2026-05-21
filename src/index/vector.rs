use crate::types::{Document, Query, Search, SearchSource, SearchResult};

pub struct VectorIndex {
    docs: Vec<Document>,
    #[allow(dead_code)]
    dim: usize,
}

impl VectorIndex {
    pub fn new(docs: Vec<Document>) -> Self {
        let dim = docs.first().map(|d| d.vector.len()).unwrap_or(0);
        Self { docs, dim }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

impl Search for VectorIndex {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult> {
        let query_vec = match query {
            Query::Vector(v) => v,
            _ => return vec![],
        };

        let mut scored: Vec<(&Document, f32)> = self
            .docs
            .iter()
            .map(|doc| {
                let sim = cosine_similarity(query_vec, &doc.vector);
                (doc, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .map(|(doc, score)| SearchResult {
                id: doc.id.clone(),
                score,
                source: SearchSource::Vector,
                snippet: truncate(&doc.text, 80),
            })
            .collect()
    }
}

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
            Document {
                id: "d1".into(),
                text: "first document".into(),
                vector: vec![1.0, 0.0],
            },
            Document {
                id: "d2".into(),
                text: "second document".into(),
                vector: vec![0.0, 1.0],
            },
            Document {
                id: "d3".into(),
                text: "third document".into(),
                vector: vec![0.5, 0.5],
            },
        ];
        let index = VectorIndex::new(docs);
        let query = Query::Vector(vec![1.0, 0.0]);
        let results = index.search(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "d1");
    }

    #[test]
    fn test_vector_search_empty_index() {
        let index = VectorIndex::new(vec![]);
        let query = Query::Vector(vec![1.0, 0.0]);
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }
}
