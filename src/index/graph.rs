use std::collections::{HashMap, HashSet, VecDeque};

use crate::types::{Query, Search, SearchSource, SearchResult, Triple};

pub struct GraphIndex {
    outgoing: HashMap<String, Vec<(String, String)>>, // subject -> [(predicate, object)]
    incoming: HashMap<String, Vec<(String, String)>>, // object -> [(predicate, subject)]
    entity_docs: HashMap<String, Vec<String>>,        // entity -> [doc_id]
}

impl GraphIndex {
    pub fn new(triples: Vec<Triple>, entity_docs: HashMap<String, Vec<String>>) -> Self {
        let mut outgoing: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut incoming: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for triple in triples {
            outgoing
                .entry(triple.subject.clone())
                .or_default()
                .push((triple.predicate.clone(), triple.object.clone()));
            incoming
                .entry(triple.object.clone())
                .or_default()
                .push((triple.predicate, triple.subject));
        }

        Self {
            outgoing,
            incoming,
            entity_docs,
        }
    }

    pub fn get_entity_docs(&self, entity: &str) -> &[String] {
        self.entity_docs.get(entity).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

impl Search for GraphIndex {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult> {
        let (entity, hops) = match query {
            Query::Graph { entity, hops } => (entity.clone(), *hops),
            _ => return vec![],
        };

        // Return empty if entity not in graph at all
        if !self.outgoing.contains_key(&entity) && !self.incoming.contains_key(&entity) {
            return vec![];
        }

        let mut results: Vec<SearchResult> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(entity.clone());

        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((entity.clone(), 0));

        loop {
            let mut next_queue = VecDeque::new();
            while let Some((current, hop)) = queue.pop_front() {
                let neighbors: Vec<_> = self
                    .outgoing
                    .get(&current)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .chain(
                        self.incoming
                            .get(&current)
                            .cloned()
                            .unwrap_or_default(),
                    )
                    .collect();

                for (predicate, neighbor) in neighbors {
                    if visited.insert(neighbor.clone()) {
                        let score = 1.0_f32 / 2_f32.powi(hop as i32);

                        let mut snippet_parts = vec![format!("{} --{}--> {}", current, predicate, neighbor)];
                        let docs = self.get_entity_docs(&neighbor);
                        if !docs.is_empty() {
                            snippet_parts.push(format!("docs: {}", docs.join(", ")));
                        }

                        results.push(SearchResult {
                            id: neighbor.clone(),
                            score,
                            source: SearchSource::Graph,
                            snippet: snippet_parts.join(" | "),
                        });

                        if hop < hops {
                            next_queue.push_back((neighbor, hop + 1));
                        }
                    }
                }
            }
            if next_queue.is_empty() {
                break;
            }
            queue = next_queue;
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
        }
    }

    #[test]
    fn test_graph_bfs_1_hop() {
        let triples = vec![
            make_triple("Rust", "influences", "C++"),
            make_triple("Rust", "runs_on", "Linux"),
        ];
        let index = GraphIndex::new(triples, HashMap::new());
        let query = Query::Graph {
            entity: "Rust".to_string(),
            hops: 1,
        };
        let results = index.search(&query, 10);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| (r.score - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_graph_bfs_2_hops() {
        let triples = vec![
            make_triple("Rust", "influences", "C++"),
            make_triple("C++", "influences", "C"),
        ];
        let index = GraphIndex::new(triples, HashMap::new());
        let query = Query::Graph {
            entity: "Rust".to_string(),
            hops: 2,
        };
        let results = index.search(&query, 10);
        assert_eq!(results.len(), 2);

        let cpp = results.iter().find(|r| r.id == "C++").unwrap();
        assert!((cpp.score - 1.0).abs() < f32::EPSILON);

        let c = results.iter().find(|r| r.id == "C").unwrap();
        assert!((c.score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_graph_bidirectional() {
        let triples = vec![
            make_triple("Rust", "influences", "C++"),
        ];
        let index = GraphIndex::new(triples, HashMap::new());
        let query = Query::Graph {
            entity: "C++".to_string(),
            hops: 1,
        };
        let results = index.search(&query, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "Rust");
        assert!((results[0].score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_graph_entity_docs() {
        let triples = vec![
            make_triple("Rust", "influences", "C++"),
        ];
        let mut entity_docs = HashMap::new();
        entity_docs.insert("C++".to_string(), vec!["doc_1".to_string(), "doc_2".to_string()]);
        let index = GraphIndex::new(triples, entity_docs);
        let query = Query::Graph {
            entity: "Rust".to_string(),
            hops: 1,
        };
        let results = index.search(&query, 10);
        assert_eq!(results.len(), 1);
        assert!(results[0].snippet.contains("doc_"));
    }

    #[test]
    fn test_graph_nonexistent_entity() {
        let index = GraphIndex::new(vec![], HashMap::new());
        let query = Query::Graph {
            entity: "NonExistent".to_string(),
            hops: 1,
        };
        let results = index.search(&query, 10);
        assert!(results.is_empty());
    }
}
