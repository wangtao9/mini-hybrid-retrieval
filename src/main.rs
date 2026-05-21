use std::collections::HashMap;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use mini_hybrid_retrieval::{Engine, Query, SearchSource};

#[derive(Parser)]
#[command(name = "mini-hybrid", version = "0.1.0")]
#[command(about = "A mini multimodal hybrid retrieval engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build index and run queries
    Index {
        /// Path to documents JSON file
        #[arg(long)]
        docs: PathBuf,
        /// Path to triples JSON file
        #[arg(long)]
        triples: PathBuf,
        /// Path to entity-docs mapping JSON file
        #[arg(long)]
        entity_docs: Option<PathBuf>,
        /// Query subcommand
        #[command(subcommand)]
        query: QueryCommands,
    },
}

#[derive(Subcommand)]
enum QueryCommands {
    /// Vector similarity search
    Vector {
        /// Query vector as comma-separated floats, e.g. "0.9,0.1,0.0,0.0"
        #[arg(long)]
        query_vector: String,
        /// Number of results
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Full-text search
    Text {
        /// Query text
        #[arg(long)]
        query: String,
        /// Number of results
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Knowledge graph search
    Graph {
        /// Seed entity name
        #[arg(long)]
        entity: String,
        /// Number of hops
        #[arg(long, default_value_t = 2)]
        hops: usize,
    },
    /// Hybrid search combining vector, text, and optional graph filter
    Hybrid {
        /// Query text
        #[arg(long)]
        query: String,
        /// Query vector as comma-separated floats
        #[arg(long)]
        query_vector: String,
        /// Optional entity name for graph filtering
        #[arg(long)]
        entity: Option<String>,
        /// Number of results
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
}

fn parse_vector(s: &str) -> Vec<f32> {
    s.split(',')
        .map(|v| v.trim().parse::<f32>().expect("Invalid float in vector"))
        .collect()
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { docs, triples, entity_docs, query } => {
            let documents = mini_hybrid_retrieval::loader::load_documents(&docs)
                .unwrap_or_else(|e| { eprintln!("Error loading documents: {}", e); std::process::exit(1); });
            let triples_data = mini_hybrid_retrieval::loader::load_triples(&triples)
                .unwrap_or_else(|e| { eprintln!("Error loading triples: {}", e); std::process::exit(1); });
            let entity_docs_data: HashMap<String, Vec<String>> = match entity_docs {
                Some(path) => mini_hybrid_retrieval::loader::load_entity_docs(&path)
                    .unwrap_or_else(|e| { eprintln!("Error loading entity_docs: {}", e); std::process::exit(1); }),
                None => HashMap::new(),
            };

            let engine = Engine::new(documents, triples_data, entity_docs_data);

            let results = match query {
                QueryCommands::Vector { query_vector, top_k } => {
                    let vec = parse_vector(&query_vector);
                    engine.search(&Query::Vector(vec), top_k)
                }
                QueryCommands::Text { query, top_k } => {
                    engine.search(&Query::Text(query), top_k)
                }
                QueryCommands::Graph { entity, hops } => {
                    engine.search(&Query::Graph { entity, hops }, 100)
                }
                QueryCommands::Hybrid { query, query_vector, entity, top_k } => {
                    let vec = parse_vector(&query_vector);
                    engine.search(&Query::Hybrid { text: query, vector: vec, entity }, top_k)
                }
            };

            for r in &results {
                let source = match r.source {
                    SearchSource::Vector => "VEC",
                    SearchSource::Text => "TXT",
                    SearchSource::Graph => "GRP",
                };
                println!("[{}] {} ({:.4}): {}", source, r.id, r.score, r.snippet);
            }
        }
    }
}
