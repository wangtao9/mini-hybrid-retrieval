use std::collections::HashMap;

use mini_hybrid_retrieval::{Document, Engine, Query, SearchSource, Triple};

fn build_test_engine() -> Engine {
    let docs = vec![
        Document { id: "doc_001".into(), text: "Rust is a systems programming language focused on safety and performance".into(), vector: vec![0.9, 0.1, 0.0, 0.0] },
        Document { id: "doc_002".into(), text: "Python is a high-level scripting language popular for data science".into(), vector: vec![0.1, 0.9, 0.0, 0.1] },
        Document { id: "doc_003".into(), text: "C++ is a systems language with manual memory management and high performance".into(), vector: vec![0.8, 0.0, 0.2, 0.0] },
        Document { id: "doc_004".into(), text: "JavaScript is the language of the web used for frontend and backend development".into(), vector: vec![0.0, 0.7, 0.0, 0.8] },
        Document { id: "doc_007".into(), text: "Rust ownership model prevents memory bugs without garbage collection".into(), vector: vec![0.95, 0.0, 0.0, 0.05] },
    ];
    let triples = vec![
        Triple { subject: "Rust".into(), predicate: "influenced_by".into(), object: "C++".into() },
        Triple { subject: "C++".into(), predicate: "influenced_by".into(), object: "C".into() },
        Triple { subject: "Python".into(), predicate: "used_in".into(), object: "DataScience".into() },
    ];
    let mut entity_docs = HashMap::new();
    entity_docs.insert("Rust".into(), vec!["doc_001".into(), "doc_007".into()]);
    entity_docs.insert("C++".into(), vec!["doc_003".into()]);
    Engine::new(docs, triples, entity_docs)
}

#[test]
fn test_vector_search_finds_similar() {
    let engine = build_test_engine();
    let results = engine.search(&Query::Vector(vec![0.9, 0.1, 0.0, 0.0]), 3);
    assert!(!results.is_empty());
    assert_eq!(results[0].source, SearchSource::Vector);
    assert!(results[0].id == "doc_001" || results[0].id == "doc_007");
}

#[test]
fn test_text_search_finds_relevant() {
    let engine = build_test_engine();
    let results = engine.search(&Query::Text("rust memory".into()), 3);
    assert!(!results.is_empty());
    assert_eq!(results[0].source, SearchSource::Text);
}

#[test]
fn test_graph_search_traverses_hops() {
    let engine = build_test_engine();
    let results = engine.search(&Query::Graph { entity: "Rust".into(), hops: 2 }, 10);
    assert!(!results.is_empty());
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"C++"));
    assert!(ids.contains(&"C"));
}

#[test]
fn test_hybrid_search_without_entity() {
    let engine = build_test_engine();
    let results = engine.search(
        &Query::Hybrid {
            text: "rust".into(),
            vector: vec![0.9, 0.1, 0.0, 0.0],
            entity: None,
        },
        5,
    );
    assert!(!results.is_empty());
}

#[test]
fn test_hybrid_search_with_entity_filter() {
    let engine = build_test_engine();
    let results = engine.search(
        &Query::Hybrid {
            text: "rust".into(),
            vector: vec![0.9, 0.1, 0.0, 0.0],
            entity: Some("Rust".into()),
        },
        5,
    );
    // Entity filter includes Rust's docs (doc_001, doc_007) + 1-hop neighbor C++'s docs (doc_003)
    assert!(!results.is_empty());
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"doc_001"), "doc_001 should be in results");
}

#[test]
fn test_empty_query_returns_no_results() {
    let engine = build_test_engine();
    let results = engine.search(&Query::Text(String::new()), 5);
    assert!(results.is_empty());
}
