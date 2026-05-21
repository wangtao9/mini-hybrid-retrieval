//! Finite State Transducer with builder and exact lookup.

/// A transition in the FST, mapping an input byte to a target node with an output value.
#[derive(Debug, Clone)]
pub struct FstTransition {
    pub target: usize,
    pub output: u64,
}

/// A node in the FST, containing ordered transitions and optional final output.
#[derive(Debug, Clone)]
pub struct FstNode {
    pub transitions: Vec<(u8, FstTransition)>,
    pub is_final: bool,
    pub final_output: u64,
}

/// A Finite State Transducer supporting exact key lookup.
/// Node at index 0 is the root.
#[derive(Debug, Clone)]
pub struct Fst {
    pub nodes: Vec<FstNode>,
}

/// Builder for constructing an FST from sorted or unsorted key-value entries.
#[derive(Debug, Clone, Default)]
pub struct FstBuilder {
    entries: Vec<(Vec<u8>, u64)>,
}

impl FstBuilder {
    pub fn new() -> Self {
        FstBuilder {
            entries: Vec::new(),
        }
    }

    /// Collect a key-value pair. Keys are byte slices; values are u64.
    pub fn insert(&mut self, key: &[u8], value: u64) {
        self.entries.push((key.to_vec(), value));
    }

    /// Sort entries lexicographically by key, then build the FST with shared prefixes.
    pub fn build(mut self) -> Fst {
        // Sort entries lexicographically by key
        self.entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Start with root node at index 0
        let mut nodes = vec![FstNode {
            transitions: Vec::new(),
            is_final: false,
            final_output: 0,
        }];

        for (key, value) in &self.entries {
            let mut current = 0usize; // start at root

            for (i, &byte) in key.iter().enumerate() {
                let is_last = i == key.len() - 1;

                // Look for existing transition with this byte from current node
                if let Some(pos) = nodes[current].transitions.iter().position(|(b, _)| *b == byte) {
                    // Follow existing transition
                    let target = nodes[current].transitions[pos].1.target;
                    if is_last {
                        // Mark the target node as final
                        nodes[target].is_final = true;
                        nodes[target].final_output = *value;
                    }
                    current = target;
                } else {
                    // Create a new node
                    let new_node = FstNode {
                        transitions: Vec::new(),
                        is_final: is_last,
                        final_output: if is_last { *value } else { 0 },
                    };
                    let new_idx = nodes.len();
                    nodes.push(new_node);

                    // Add transition from current node
                    nodes[current].transitions.push((
                        byte,
                        FstTransition {
                            target: new_idx,
                            output: 0,
                        },
                    ));

                    current = new_idx;
                }
            }

            // Handle the case of an empty key: mark root as final
            if key.is_empty() {
                nodes[0].is_final = true;
                nodes[0].final_output = *value;
            }
        }

        // Sort transitions by byte in every node for deterministic binary search
        for node in &mut nodes {
            node.transitions.sort_by_key(|(byte, _)| *byte);
        }

        Fst { nodes }
    }
}

impl Fst {
    /// Exact lookup: walk the byte path using binary search on transitions.
    /// Returns the final_output if a final node is reached, or None otherwise.
    pub fn get(&self, key: &[u8]) -> Option<u64> {
        let mut current = 0usize;

        for &byte in key {
            let node = &self.nodes[current];
            match node.transitions.binary_search_by_key(&byte, |(b, _)| *b) {
                Ok(pos) => {
                    current = node.transitions[pos].1.target;
                }
                Err(_) => return None,
            }
        }

        let node = &self.nodes[current];
        if node.is_final {
            Some(node.final_output)
        } else {
            None
        }
    }

    /// Prefix search: find all keys starting with `prefix`.
    /// Returns pairs of (key_string, output_value) sorted lexicographically.
    pub fn prefix_search(&self, prefix: &[u8]) -> Vec<(String, u64)> {
        // Walk to the node at the end of the prefix
        let mut current = 0usize;
        for &byte in prefix {
            let node = &self.nodes[current];
            match node.transitions.binary_search_by_key(&byte, |(b, _)| *b) {
                Ok(pos) => {
                    current = node.transitions[pos].1.target;
                }
                Err(_) => return Vec::new(),
            }
        }

        // DFS from the prefix node, collecting all final states
        let mut results = Vec::new();
        let mut stack: Vec<(usize, Vec<u8>)> = vec![(current, Vec::new())];

        while let Some((node_idx, suffix)) = stack.pop() {
            let node = &self.nodes[node_idx];
            if node.is_final {
                let mut key = prefix.to_vec();
                key.extend_from_slice(&suffix);
                results.push((String::from_utf8_lossy(&key).into_owned(), node.final_output));
            }
            // Push transitions in reverse order so smallest byte is processed first
            for (byte, trans) in node.transitions.iter().rev() {
                let mut new_suffix = suffix.clone();
                new_suffix.push(*byte);
                stack.push((trans.target, new_suffix));
            }
        }

        results.sort_by(|a, b| a.0.cmp(&b.0));
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fst_single_key() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 42);
        let fst = builder.build();
        assert_eq!(fst.get(b"cat"), Some(42));
    }

    #[test]
    fn test_fst_multiple_keys() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        builder.insert(b"dog", 2);
        builder.insert(b"elephant", 3);
        let fst = builder.build();
        assert_eq!(fst.get(b"cat"), Some(1));
        assert_eq!(fst.get(b"dog"), Some(2));
        assert_eq!(fst.get(b"elephant"), Some(3));
    }

    #[test]
    fn test_fst_missing_key() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        let fst = builder.build();
        assert_eq!(fst.get(b"dog"), None);
    }

    #[test]
    fn test_fst_empty() {
        let builder = FstBuilder::new();
        let fst = builder.build();
        assert_eq!(fst.get(b"anything"), None);
    }

    #[test]
    fn test_fst_shared_prefix() {
        let mut builder = FstBuilder::new();
        builder.insert(b"car", 1);
        builder.insert(b"cat", 2);
        builder.insert(b"cab", 3);
        let fst = builder.build();
        assert_eq!(fst.get(b"car"), Some(1));
        assert_eq!(fst.get(b"cat"), Some(2));
        assert_eq!(fst.get(b"cab"), Some(3));
    }

    #[test]
    fn test_fst_output_accumulation() {
        let mut builder = FstBuilder::new();
        builder.insert(b"a", 10);
        builder.insert(b"ab", 20);
        let fst = builder.build();
        assert_eq!(fst.get(b"a"), Some(10));
        assert_eq!(fst.get(b"ab"), Some(20));
    }

    #[test]
    fn test_fst_prefix_search_basic() {
        let mut builder = FstBuilder::new();
        builder.insert(b"car", 1);
        builder.insert(b"cat", 2);
        builder.insert(b"cab", 3);
        builder.insert(b"dog", 4);
        let fst = builder.build();
        let results = fst.prefix_search(b"ca");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (String::from("cab"), 3));
        assert_eq!(results[1], (String::from("car"), 1));
        assert_eq!(results[2], (String::from("cat"), 2));
    }

    #[test]
    fn test_fst_prefix_search_exact_key() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 2);
        builder.insert(b"catalog", 5);
        let fst = builder.build();
        let results = fst.prefix_search(b"cat");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (String::from("cat"), 2));
        assert_eq!(results[1], (String::from("catalog"), 5));
    }

    #[test]
    fn test_fst_prefix_search_no_match() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        let fst = builder.build();
        let results = fst.prefix_search(b"do");
        assert!(results.is_empty());
    }

    #[test]
    fn test_fst_prefix_search_empty_prefix() {
        let mut builder = FstBuilder::new();
        builder.insert(b"cat", 1);
        builder.insert(b"dog", 2);
        let fst = builder.build();
        let results = fst.prefix_search(b"");
        assert_eq!(results.len(), 2);
    }
}
