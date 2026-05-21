use std::collections::HashMap;
use std::path::Path;
use crate::types::{Document, Triple};

pub fn load_documents(_path: &Path) -> Result<Vec<Document>, String> {
    Ok(vec![])
}

pub fn load_triples(_path: &Path) -> Result<Vec<Triple>, String> {
    Ok(vec![])
}

pub fn load_entity_docs(_path: &Path) -> Result<HashMap<String, Vec<String>>, String> {
    Ok(HashMap::new())
}
