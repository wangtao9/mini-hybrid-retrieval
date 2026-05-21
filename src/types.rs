use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SearchSource {
    Vector,
    Text,
    Graph,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub source: SearchSource,
    pub snippet: String,
}

#[derive(Debug, Clone)]
pub enum Query {
    Vector(Vec<f32>),
    Text(String),
    Graph { entity: String, hops: usize },
    Hybrid {
        text: String,
        vector: Vec<f32>,
        entity: Option<String>,
    },
}

pub trait Search {
    fn search(&self, query: &Query, top_k: usize) -> Vec<SearchResult>;
}
