use std::collections::HashMap;
use std::path::Path;

use crate::types::{Document, Triple};

pub fn load_documents(path: &Path) -> Result<Vec<Document>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    let docs: Vec<Document> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse documents: {}", e))?;

    if docs.len() > 1 {
        let dim = docs[0].vector.len();
        for doc in docs.iter().skip(1) {
            if doc.vector.len() != dim {
                return Err(format!(
                    "Vector dimension mismatch: doc {} has {} dims, expected {}",
                    doc.id,
                    doc.vector.len(),
                    dim
                ));
            }
        }
    }

    Ok(docs)
}

pub fn load_triples(path: &Path) -> Result<Vec<Triple>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse triples: {}", e))
}

pub fn load_entity_docs(path: &Path) -> Result<HashMap<String, Vec<String>>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse entity_docs: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_json(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_load_documents() {
        let json = r#"[{"id":"d1","text":"hello world","vector":[1.0,0.0]}]"#;
        let f = write_temp_json(json);
        let docs = load_documents(f.path()).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "d1");
        assert_eq!(docs[0].vector, vec![1.0, 0.0]);
    }

    #[test]
    fn test_load_documents_dimension_mismatch() {
        let json = r#"[{"id":"d1","text":"a","vector":[1.0,0.0]},{"id":"d2","text":"b","vector":[1.0]}]"#;
        let f = write_temp_json(json);
        let result = load_documents(f.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_triples() {
        let json = r#"[{"subject":"Rust","predicate":"influenced_by","object":"C++"}]"#;
        let f = write_temp_json(json);
        let triples = load_triples(f.path()).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, "Rust");
    }

    #[test]
    fn test_load_entity_docs() {
        let json = r#"{"Rust":["d1","d2"],"C++":["d3"]}"#;
        let f = write_temp_json(json);
        let map = load_entity_docs(f.path()).unwrap();
        assert_eq!(map.get("Rust").unwrap().len(), 2);
    }
}
