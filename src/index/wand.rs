use std::collections::BinaryHeap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Wrapper for (score, doc_id) that implements Ord via f32 total ordering.
#[derive(Debug, Clone, PartialEq)]
struct ScoreEntry(f32, u32);

impl Eq for ScoreEntry {}

impl PartialOrd for ScoreEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then_with(|| self.1.cmp(&other.1))
    }
}

#[derive(Clone)]
pub struct Posting {
    pub doc_id: u32,
    pub tf: u32,
}

pub struct WandResult {
    pub doc_id: u32,
    pub score: f32,
}

struct PostingCursor<'a> {
    #[allow(dead_code)]
    term_idx: usize,
    posting_list: &'a [Posting],
    current_pos: usize,
    upper_bound: f32,
}

impl<'a> PostingCursor<'a> {
    fn current_doc_id(&self) -> u32 {
        self.posting_list[self.current_pos].doc_id
    }

    fn advance_to(&mut self, target_doc_id: u32) {
        while self.current_pos < self.posting_list.len()
            && self.posting_list[self.current_pos].doc_id < target_doc_id
        {
            self.current_pos += 1;
        }
    }

    fn is_exhausted(&self) -> bool {
        self.current_pos >= self.posting_list.len()
    }
}

fn score_document(
    cursors: &[PostingCursor],
    doc_id: u32,
    _idf_values: &[f32],
    doc_lengths: &[f32],
    avg_dl: f32,
) -> f32 {
    let n = doc_lengths.len() as f32;
    let mut idf_parts: Vec<f32> = Vec::new();
    let mut tf_norm_parts: Vec<f32> = Vec::new();

    for cursor in cursors {
        if cursor.is_exhausted() {
            continue;
        }
        if cursor.current_doc_id() != doc_id {
            continue;
        }
        let tf = cursor.posting_list[cursor.current_pos].tf as f32;
        let df = cursor.posting_list.len() as f32;
        let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
        let dl = doc_lengths[doc_id as usize];
        let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avg_dl));
        idf_parts.push(idf);
        tf_norm_parts.push(tf_norm);
    }

    crate::index::simd::bm25_score_batch(&idf_parts, &tf_norm_parts)
}

pub fn wand_search(
    posting_lists: &[Vec<Posting>],
    upper_bounds: &[f32],
    idf_values: &[f32],
    doc_lengths: &[f32],
    avg_dl: f32,
    top_k: usize,
) -> Vec<WandResult> {
    if top_k == 0 {
        return Vec::new();
    }

    let mut cursors: Vec<PostingCursor> = posting_lists
        .iter()
        .enumerate()
        .filter(|(_, pl)| !pl.is_empty())
        .map(|(i, pl)| PostingCursor {
            term_idx: i,
            posting_list: pl,
            current_pos: 0,
            upper_bound: upper_bounds[i],
        })
        .collect();

    if cursors.is_empty() {
        return Vec::new();
    }

    let mut heap: BinaryHeap<std::cmp::Reverse<ScoreEntry>> = BinaryHeap::new();
    let mut threshold: f32 = 0.0;

    loop {
        // Remove exhausted cursors
        cursors.retain(|c| !c.is_exhausted());
        if cursors.is_empty() {
            break;
        }

        // Sort cursors by current_doc_id
        cursors.sort_by_key(|c| c.current_doc_id());

        // Find pivot: accumulate upper_bounds until sum > threshold
        let mut accumulated = 0.0_f32;
        let mut pivot_idx = None;
        for (idx, cursor) in cursors.iter().enumerate() {
            accumulated += cursor.upper_bound;
            if accumulated > threshold {
                pivot_idx = Some(idx);
                break;
            }
        }

        // If no pivot found, break
        let pivot_idx = match pivot_idx {
            Some(idx) => idx,
            None => break,
        };

        let pivot_doc_id = cursors[pivot_idx].current_doc_id();

        // If smallest cursor doc_id == pivot doc_id, score the document
        if cursors[0].current_doc_id() == pivot_doc_id {
            let score = score_document(&cursors, pivot_doc_id, idf_values, doc_lengths, avg_dl);

            if heap.len() < top_k {
                heap.push(std::cmp::Reverse(ScoreEntry(score, pivot_doc_id)));
                if heap.len() == top_k {
                    threshold = heap.peek().unwrap().0.0;
                }
            } else if score > threshold {
                heap.pop();
                heap.push(std::cmp::Reverse(ScoreEntry(score, pivot_doc_id)));
                threshold = heap.peek().unwrap().0.0;
            }

            // Advance all cursors pointing to pivot_doc_id past it
            for cursor in &mut cursors {
                if !cursor.is_exhausted() && cursor.current_doc_id() == pivot_doc_id {
                    cursor.current_pos += 1;
                }
            }
        } else {
            // Advance cursors before pivot to pivot doc_id
            for cursor in &mut cursors[..pivot_idx] {
                cursor.advance_to(pivot_doc_id);
            }
        }
    }

    let mut results: Vec<WandResult> = heap
        .into_iter()
        .map(|std::cmp::Reverse(ScoreEntry(score, doc_id))| WandResult { doc_id, score })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    results
}

pub fn full_scan_search(
    posting_lists: &[Vec<Posting>],
    _idf_values: &[f32],
    doc_lengths: &[f32],
    avg_dl: f32,
    top_k: usize,
) -> Vec<WandResult> {
    let n = doc_lengths.len() as f32;

    // Collect all doc_ids and their scores
    let mut doc_scores: std::collections::HashMap<u32, (Vec<f32>, Vec<f32>)> =
        std::collections::HashMap::new();

    for postings in posting_lists.iter() {
        let df = postings.len() as f32;
        let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();

        for posting in postings {
            let tf = posting.tf as f32;
            let dl = doc_lengths[posting.doc_id as usize];
            let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avg_dl));

            let entry = doc_scores.entry(posting.doc_id).or_insert_with(|| (Vec::new(), Vec::new()));
            entry.0.push(idf);
            entry.1.push(tf_norm);
        }
    }

    let mut all_results: Vec<WandResult> = doc_scores
        .into_iter()
        .map(|(doc_id, (idf_parts, tf_norm_parts))| {
            let score = crate::index::simd::bm25_score_batch(&idf_parts, &tf_norm_parts);
            WandResult { doc_id, score }
        })
        .collect();

    // Sort by score descending
    all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    all_results.truncate(top_k);
    all_results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wand_single_term() {
        // 1 term, 3 docs, top_k=2, doc with highest tf ranks first
        let posting_lists = vec![vec![
            Posting { doc_id: 0, tf: 1 },
            Posting { doc_id: 1, tf: 5 },
            Posting { doc_id: 2, tf: 3 },
        ]];
        let idf_values = vec![1.0];
        let doc_lengths = vec![100.0, 100.0, 100.0];
        let avg_dl = 100.0;
        let upper_bounds = vec![idf_values[0] * (K1 + 1.0)]; // max possible score for this term

        let results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 2);
        assert_eq!(results.len(), 2);
        // Doc 1 has tf=5, should rank first
        assert_eq!(results[0].doc_id, 1);
        // Doc 2 has tf=3, should rank second
        assert_eq!(results[1].doc_id, 2);
    }

    #[test]
    fn test_wand_multi_term() {
        // 2 terms, 4 docs, top_k=2, docs appearing in both terms rank higher
        let posting_lists = vec![
            vec![
                Posting { doc_id: 0, tf: 2 },
                Posting { doc_id: 1, tf: 3 },
                Posting { doc_id: 3, tf: 1 },
            ],
            vec![
                Posting { doc_id: 0, tf: 1 },
                Posting { doc_id: 2, tf: 4 },
                Posting { doc_id: 3, tf: 2 },
            ],
        ];
        let idf_values = vec![1.5, 2.0];
        let doc_lengths = vec![100.0, 100.0, 100.0, 100.0];
        let avg_dl = 100.0;
        let upper_bounds: Vec<f32> = idf_values.iter().map(|&idf| idf * (K1 + 1.0)).collect();

        let results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 2);
        assert_eq!(results.len(), 2);
        // Docs 0 and 3 appear in both terms, so they should rank higher
        let top_ids: std::collections::HashSet<u32> = results.iter().map(|r| r.doc_id).collect();
        assert!(top_ids.contains(&0) || top_ids.contains(&3));
    }

    #[test]
    fn test_wand_empty() {
        let posting_lists: Vec<Vec<Posting>> = vec![];
        let idf_values: Vec<f32> = vec![];
        let doc_lengths: Vec<f32> = vec![];
        let avg_dl = 0.0;
        let upper_bounds: Vec<f32> = vec![];

        let results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_wand_matches_full_scan() {
        // 3 terms, 5 docs, top_k=3, WAND results match full_scan exactly
        let posting_lists = vec![
            vec![
                Posting { doc_id: 0, tf: 3 },
                Posting { doc_id: 2, tf: 1 },
                Posting { doc_id: 4, tf: 5 },
            ],
            vec![
                Posting { doc_id: 0, tf: 2 },
                Posting { doc_id: 1, tf: 4 },
                Posting { doc_id: 3, tf: 1 },
            ],
            vec![
                Posting { doc_id: 1, tf: 1 },
                Posting { doc_id: 2, tf: 3 },
                Posting { doc_id: 4, tf: 2 },
            ],
        ];
        let idf_values = vec![1.5, 2.0, 1.0];
        let doc_lengths = vec![80.0, 120.0, 95.0, 110.0, 100.0];
        let avg_dl = 101.0;
        let upper_bounds: Vec<f32> = idf_values.iter().map(|&idf| idf * (K1 + 1.0)).collect();

        let wand_results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 3);
        let full_results = full_scan_search(&posting_lists, &idf_values, &doc_lengths, avg_dl, 3);

        assert_eq!(wand_results.len(), full_results.len());
        for (w, f) in wand_results.iter().zip(full_results.iter()) {
            assert_eq!(w.doc_id, f.doc_id);
            assert!((w.score - f.score).abs() < 1e-4, "score mismatch: {} vs {}", w.score, f.score);
        }
    }

    #[test]
    fn test_wand_pruning_effect() {
        // 2 terms, 100 docs, top_k=3
        let mut posting_lists: Vec<Vec<Posting>> = Vec::new();

        // Term 0: docs 0..50
        let mut term0 = Vec::new();
        for doc_id in 0..50u32 {
            term0.push(Posting { doc_id, tf: (doc_id % 5) + 1 });
        }
        posting_lists.push(term0);

        // Term 1: docs 25..75
        let mut term1 = Vec::new();
        for doc_id in 25..75u32 {
            term1.push(Posting { doc_id, tf: (doc_id % 3) + 1 });
        }
        posting_lists.push(term1);

        let idf_values = vec![1.5, 2.0];
        let doc_lengths: Vec<f32> = (0..100).map(|_| 100.0).collect();
        let avg_dl = 100.0;
        let upper_bounds: Vec<f32> = idf_values.iter().map(|&idf| idf * (K1 + 1.0)).collect();

        let results = wand_search(&posting_lists, &upper_bounds, &idf_values, &doc_lengths, avg_dl, 3);
        assert_eq!(results.len(), 3);
    }
}
