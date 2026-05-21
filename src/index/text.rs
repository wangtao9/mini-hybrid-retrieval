use crate::index::fst::{Fst, FstBuilder};
use crate::index::wand::{wand_search, Posting, WandResult};
use crate::types::{Document, Query, Search, SearchSource, SearchResult};
use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Bidirectional String ↔ u32 mapping for WAND-compatible doc IDs.
struct DocIdMap {
    to_internal: HashMap<String, u32>,
    to_external: Vec<String>,
}

impl DocIdMap {
    fn new() -> Self {
        DocIdMap {
            to_internal: HashMap::new(),
            to_external: Vec::new(),
        }
    }

    fn get_or_insert(&mut self, external: &str) -> u32 {
        if let Some(&id) = self.to_internal.get(external) {
            return id;
        }
        let id = self.to_external.len() as u32;
        self.to_external.push(external.to_string());
        self.to_internal.insert(external.to_string(), id);
        id
    }

    fn to_external(&self, id: u32) -> &str {
        &self.to_external[id as usize]
    }
}

struct PostingList {
    entries: Vec<Posting>,
}

pub struct TextIndex {
    fst: Fst,
    posting_lists: Vec<PostingList>,
    #[allow(dead_code)]
    term_strings: Vec<String>,
    term_upper_bounds: Vec<f32>,
    doc_id_map: DocIdMap,
    doc_lengths: Vec<f32>,
    avg_dl: f32,
    doc_count: usize,
    doc_texts: Vec<String>,
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
        let mut doc_id_map = DocIdMap::new();
        let mut doc_lengths: Vec<f32> = vec![0.0; doc_count];
        let mut doc_texts: Vec<String> = Vec::with_capacity(doc_count);

        // term -> (internal_doc_id, tf) pairs
        let mut term_postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();

        let mut total_length: usize = 0;

        for doc in &docs {
            let internal_id = doc_id_map.get_or_insert(&doc.id);
            let tokens = tokenize(&doc.text);
            doc_lengths[internal_id as usize] = tokens.len() as f32;
            total_length += tokens.len();
            doc_texts.push(doc.text.clone());

            // Count tf per term for this doc
            let mut term_counts: HashMap<String, u32> = HashMap::new();
            for token in &tokens {
                *term_counts.entry(token.clone()).or_insert(0) += 1;
            }
            for (term, tf) in term_counts {
                term_postings
                    .entry(term)
                    .or_default()
                    .push((internal_id, tf));
            }
        }

        let avg_dl = if doc_count > 0 {
            total_length as f32 / doc_count as f32
        } else {
            0.0
        };

        // Sort terms lexicographically, build posting lists and FST
        let mut sorted_terms: Vec<String> = term_postings.keys().cloned().collect();
        sorted_terms.sort();

        let n = doc_count as f32;
        let mut posting_lists: Vec<PostingList> = Vec::with_capacity(sorted_terms.len());
        let mut term_upper_bounds: Vec<f32> = Vec::with_capacity(sorted_terms.len());
        let mut fst_builder = FstBuilder::new();

        for (idx, term) in sorted_terms.iter().enumerate() {
            let mut entries = term_postings.remove(term).unwrap();
            // Sort by doc_id for WAND
            entries.sort_by_key(|(doc_id, _)| *doc_id);

            let df = entries.len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();

            // Upper bound: max possible BM25 score for this term (tf at max, shortest doc)
            let max_tf = entries.iter().map(|(_, tf)| *tf).max().unwrap_or(0) as f32;
            let min_dl = doc_lengths
                .iter()
                .copied()
                .filter(|&l| l > 0.0)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);
            let max_tf_norm = if avg_dl > 0.0 {
                (max_tf * (K1 + 1.0)) / (max_tf + K1 * (1.0 - B + B * min_dl / avg_dl))
            } else {
                0.0
            };
            let upper_bound = idf * max_tf_norm;

            posting_lists.push(PostingList {
                entries: entries
                    .into_iter()
                    .map(|(doc_id, tf)| Posting { doc_id, tf })
                    .collect(),
            });
            term_upper_bounds.push(upper_bound);
            fst_builder.insert(term.as_bytes(), idx as u64);
        }

        let fst = fst_builder.build();

        TextIndex {
            fst,
            posting_lists,
            term_strings: sorted_terms,
            term_upper_bounds,
            doc_id_map,
            doc_lengths,
            avg_dl,
            doc_count,
            doc_texts,
        }
    }

    /// Collect posting list indices for query tokens using FST exact and prefix lookup.
    fn resolve_terms(&self, tokens: &[String]) -> Vec<usize> {
        let mut indices: Vec<usize> = Vec::new();
        for token in tokens {
            // Exact match
            if let Some(idx) = self.fst.get(token.as_bytes()) {
                let i = idx as usize;
                if i < self.posting_lists.len() && !self.posting_lists[i].entries.is_empty() {
                    indices.push(i);
                }
            } else {
                // Prefix expansion: find all terms starting with this token
                for (_key, idx) in self.fst.prefix_search(token.as_bytes()) {
                    let i = idx as usize;
                    if i < self.posting_lists.len() && !self.posting_lists[i].entries.is_empty() {
                        indices.push(i);
                    }
                }
            }
        }
        indices.sort();
        indices.dedup();
        indices
    }
}

impl Search for TextIndex {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult> {
        let query_text = match query {
            Query::Text(t) => t,
            Query::Hybrid { text, .. } => text,
            _ => return vec![],
        };

        let tokens = tokenize(query_text);
        if tokens.is_empty() {
            return vec![];
        }

        let term_indices = self.resolve_terms(&tokens);
        if term_indices.is_empty() {
            return vec![];
        }

        // Collect posting lists and IDFs for WAND
        let n = self.doc_count as f32;
        let posting_lists: Vec<Vec<Posting>> = term_indices
            .iter()
            .map(|&i| self.posting_lists[i].entries.clone())
            .collect();
        let upper_bounds: Vec<f32> = term_indices.iter().map(|&i| self.term_upper_bounds[i]).collect();
        let idf_values: Vec<f32> = term_indices
            .iter()
            .map(|&i| {
                let df = self.posting_lists[i].entries.len() as f32;
                (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
            })
            .collect();

        let wand_results: Vec<WandResult> = wand_search(
            &posting_lists,
            &upper_bounds,
            &idf_values,
            &self.doc_lengths,
            self.avg_dl,
            top_k,
        );

        wand_results
            .into_iter()
            .map(|r| {
                let external_id = self.doc_id_map.to_external(r.doc_id).to_string();
                let text = &self.doc_texts[r.doc_id as usize];
                let snippet = make_snippet(text, None);
                SearchResult {
                    id: external_id,
                    score: r.score,
                    source: SearchSource::Text,
                    snippet,
                }
            })
            .collect()
    }
}

pub fn make_snippet(text: &str, positions: Option<&Vec<usize>>) -> String {
    let tokens = tokenize(text);
    let start = if let Some(pos) = positions {
        if pos.is_empty() {
            0
        } else {
            let first = pos.iter().min().copied().unwrap_or(0);
            first.saturating_sub(3)
        }
    } else {
        0
    };

    let context_window = 10;
    let end = std::cmp::min(start + context_window, tokens.len());

    if start >= tokens.len() {
        return truncate(text, 80);
    }

    let snippet = tokens[start..end].join(" ");
    let prefix = if start > 0 { "..." } else { "" };
    let suffix = if end < tokens.len() { "..." } else { "" };

    format!("{}{}{}", prefix, snippet, suffix)
}

pub fn truncate(s: &str, max_len: usize) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let result = tokenize("Hello, world! This is Rust.");
        assert_eq!(result, vec!["hello", "world", "this", "is", "rust"]);
    }

    #[test]
    fn test_tokenize_empty() {
        let result = tokenize("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_text_index_search() {
        let docs = vec![
            Document {
                id: "d1".into(),
                text: "Rust is a systems programming language.".into(),
                vector: vec![],
            },
            Document {
                id: "d2".into(),
                text: "Python is a popular scripting language.".into(),
                vector: vec![],
            },
            Document {
                id: "d3".into(),
                text: "Rust programming is fun and productive.".into(),
                vector: vec![],
            },
        ];

        let index = TextIndex::new(docs);
        let results = index.search(&Query::Text("rust programming".into()), 2);

        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"d3"), "d3 should be in top results");
        assert_eq!(results[0].source, SearchSource::Text);
    }

    #[test]
    fn test_text_index_no_match() {
        let docs = vec![
            Document {
                id: "d1".into(),
                text: "Rust is a systems language.".into(),
                vector: vec![],
            },
        ];

        let index = TextIndex::new(docs);
        let results = index.search(&Query::Text("python".into()), 5);

        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_scoring() {
        let docs = vec![
            Document {
                id: "d1".into(),
                text: "rust rust rust".into(),
                vector: vec![],
            },
            Document {
                id: "d2".into(),
                text: "rust".into(),
                vector: vec![],
            },
        ];

        let index = TextIndex::new(docs);
        let results = index.search(&Query::Text("rust".into()), 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "d1", "doc with higher tf should score higher");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_prefix_search() {
        let docs = vec![
            Document {
                id: "d1".into(),
                text: "Rust programming language".into(),
                vector: vec![],
            },
            Document {
                id: "d2".into(),
                text: "Ruby programming language".into(),
                vector: vec![],
            },
        ];

        let index = TextIndex::new(docs);
        // "ru" should prefix-expand to "rust" and "ruby"
        let results = index.search(&Query::Text("ru".into()), 5);
        assert_eq!(results.len(), 2, "prefix search should match both docs");
    }
}