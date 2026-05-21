use crate::types::{Document, Query, Search, SearchSource, SearchResult};
use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

#[derive(Debug, Clone)]
pub struct Posting {
    pub doc_id: String,
    pub tf: usize,
    pub positions: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PostingList {
    pub entries: Vec<Posting>,
}

pub struct TextIndex {
    inverted: HashMap<String, PostingList>,
    doc_lengths: HashMap<String, usize>,
    avg_dl: f32,
    doc_count: usize,
    doc_texts: HashMap<String, String>,
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
        let mut inverted: HashMap<String, PostingList> = HashMap::new();
        let mut doc_lengths: HashMap<String, usize> = HashMap::new();
        let mut doc_texts: HashMap<String, String> = HashMap::new();

        let mut total_length: usize = 0;

        for doc in &docs {
            let tokens = tokenize(&doc.text);
            doc_lengths.insert(doc.id.clone(), tokens.len());
            total_length += tokens.len();
            doc_texts.insert(doc.id.clone(), doc.text.clone());

            for (pos, token) in tokens.iter().enumerate() {
                let list = inverted.entry(token.clone()).or_insert_with(|| PostingList {
                    entries: Vec::new(),
                });

                if let Some(entry) = list.entries.iter_mut().find(|e| e.doc_id == doc.id) {
                    entry.tf += 1;
                    entry.positions.push(pos);
                } else {
                    list.entries.push(Posting {
                        doc_id: doc.id.clone(),
                        tf: 1,
                        positions: vec![pos],
                    });
                }
            }
        }

        let avg_dl = if doc_count > 0 {
            total_length as f32 / doc_count as f32
        } else {
            0.0
        };

        TextIndex {
            inverted,
            doc_lengths,
            avg_dl,
            doc_count,
            doc_texts,
        }
    }

    pub fn bm25_score(&self, tf: usize, df: usize, dl: usize) -> f32 {
        let n = self.doc_count as f32;
        let df_f = df as f32;
        let tf_f = tf as f32;
        let dl_f = dl as f32;

        let idf = (1.0 + (n - df_f + 0.5) / (df_f + 0.5)).ln();
        let tf_norm = (tf_f * (K1 + 1.0))
            / (tf_f + K1 * (1.0 - B + B * dl_f / self.avg_dl));

        idf * tf_norm
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

        let mut scores: HashMap<String, f32> = HashMap::new();
        let mut match_positions: HashMap<String, Vec<usize>> = HashMap::new();

        for token in &tokens {
            if let Some(list) = self.inverted.get(token) {
                let df = list.entries.len();
                for entry in &list.entries {
                    let dl = *self.doc_lengths.get(&entry.doc_id).unwrap_or(&0);
                    let score = self.bm25_score(entry.tf, df, dl);
                    *scores.entry(entry.doc_id.clone()).or_insert(0.0) += score;

                    let positions = match_positions.entry(entry.doc_id.clone()).or_default();
                    positions.extend(&entry.positions);
                }
            }
        }

        let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked
            .into_iter()
            .take(top_k)
            .map(|(doc_id, score)| {
                let text = self.doc_texts.get(&doc_id).unwrap();
                let positions = match_positions.get(&doc_id);
                let snippet = make_snippet(text, positions);
                SearchResult {
                    id: doc_id,
                    score,
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
            // Find the first matching token position, back up a bit for context
            let first = pos.iter().min().copied().unwrap_or(0);
            if first > 3 { first - 3 } else { 0 }
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
        // Both d1 and d3 contain both query terms; d3 is shorter and more focused
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
}
