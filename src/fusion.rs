use std::collections::HashMap;
use crate::types::{SearchResult, SearchSource};

pub fn reciprocal_rank_fusion(result_sets: &[Vec<SearchResult>], k: usize) -> Vec<SearchResult> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for result_set in result_sets {
        for (rank, result) in result_set.iter().enumerate() {
            let rrf_score = 1.0 / (k as f32 + rank as f32 + 1.0);
            *scores.entry(result.id.clone()).or_insert(0.0) += rrf_score;
        }
    }

    let mut fused: Vec<SearchResult> = scores
        .into_iter()
        .map(|(id, score)| SearchResult {
            id,
            score,
            source: SearchSource::Vector,
            snippet: String::new(),
        })
        .collect();

    fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, score: f32, source: SearchSource) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score,
            source,
            snippet: format!("snippet for {}", id),
        }
    }

    #[test]
    fn test_rrf_two_lists() {
        let list1 = vec![
            make_result("a", 0.9, SearchSource::Vector),
            make_result("b", 0.8, SearchSource::Vector),
            make_result("c", 0.7, SearchSource::Vector),
        ];
        let list2 = vec![
            make_result("b", 0.95, SearchSource::Text),
            make_result("d", 0.85, SearchSource::Text),
            make_result("a", 0.75, SearchSource::Text),
        ];

        let fused = reciprocal_rank_fusion(&[list1, list2], 60);

        // 4 unique ids: a, b, c, d
        assert_eq!(fused.len(), 4);

        // a and b appear in both lists, so they should score higher than c and d
        let top2_scores: Vec<f32> = fused[..2].iter().map(|r| r.score).collect();
        let bottom2_scores: Vec<f32> = fused[2..].iter().map(|r| r.score).collect();
        assert!(top2_scores.iter().all(|s| *s > bottom2_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max)));
    }

    #[test]
    fn test_rrf_single_list() {
        let list = vec![
            make_result("x", 0.9, SearchSource::Vector),
            make_result("y", 0.8, SearchSource::Vector),
            make_result("z", 0.7, SearchSource::Vector),
        ];

        let fused = reciprocal_rank_fusion(&[list], 60);

        assert_eq!(fused.len(), 3);
        // Single list preserves order since earlier ranks get higher RRF scores
        assert_eq!(fused[0].id, "x");
        assert_eq!(fused[1].id, "y");
        assert_eq!(fused[2].id, "z");
    }

    #[test]
    fn test_rrf_empty() {
        let fused = reciprocal_rank_fusion(&[], 60);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_deduplication() {
        let list1 = vec![make_result("a", 0.9, SearchSource::Vector)];
        let list2 = vec![make_result("a", 0.8, SearchSource::Text)];

        let k = 60;
        let fused = reciprocal_rank_fusion(&[list1, list2], k);

        // Same id "a" at rank 0 in both lists: combined score = 2/(k+1)
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].id, "a");
        let expected = 2.0 / (k as f32 + 1.0);
        assert!((fused[0].score - expected).abs() < 1e-6);
    }
}
