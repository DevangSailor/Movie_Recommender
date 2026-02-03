# Movie Recommender System

A hybrid knowledge graph and vector-based movie recommendation system powered by Neo4j, Ollama, and LLM-based query understanding.

## Overview

This project implements an intelligent movie recommendation system that combines:
- **Knowledge Graph (Neo4j)**: Structured movie data with relationships (directors, actors, genres, keywords)
- **Vector Embeddings**: Semantic search using plot descriptions via Ollama's `nomic-embed-text` model
- **LLM Query Understanding**: Natural language query decomposition using Qwen3 8B
- **Hybrid Search**: Graph-based connection boosting combined with semantic similarity

The system understands queries like "movies like Inception" or "dark thriller about revenge" and provides personalized recommendations based on plot similarity and graph relationships.

## Features

- **Natural Language Queries**: Ask for movies in plain English
- **Reference Movie Expansion**: Automatically finds movies similar to a given title
- **Graph-Boosted Scoring**: Recommendations prioritize shared directors, genres, actors, and keywords
- **Semantic Search**: Plot-based similarity using 768-dimensional embeddings
- **Self-Hosted LLMs**: Runs entirely offline using Ollama (no API keys required)
- **Flexible Architecture**: Modular design with separate components for graph building, embedding, and recommendation

## Architecture

```
User Query → LLM (Query Decomposition) → 
    ├─ Reference Movie Detection
    ├─ Knowledge Graph Expansion (shared connections)
    └─ Hybrid Search (semantic + graph boost) → 
        LLM (Ranking) → Top 5 Recommendations
```

## Requirements

- Python 3.8+
- Neo4j 5.15.0+
- Docker & Docker Compose
- Ollama with models:
  - `nomic-embed-text:latest` (768-dim embeddings)
  - `qwen3:8b` (query understanding and ranking)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Neo4j with Docker Compose

```bash
docker-compose up -d
```

This will start Neo4j with:
- Web interface: http://localhost:7474
- Bolt connection: bolt://localhost:7687
- APOC and Graph Data Science plugins enabled

### 4. Install Ollama Models

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai

# Pull required models
ollama pull nomic-embed-text:latest
ollama pull qwen3:8b
```

### 5. Configure Environment

The Neo4j password is set in `docker-compose.yml`. Update if needed:

```yaml
NEO4J_AUTH=neo4j/<your-password>
```

## Usage

### Building the Knowledge Graph

**Note**: To obtain the movie dataset with TMDB and Wikipedia data, please contact me at your preferred email.

Once you have the data file:

```bash
python graph_builder.py <path-to-movie-data.json> \
    --uri bolt://localhost:7687 \
    --username neo4j \
    --password <your-password>
```

This will:
1. Generate embeddings for all movies using their plot descriptions
2. Create Movie, Person, Genre, Keyword, and Language nodes
3. Establish relationships (DIRECTED_BY, HAS_CAST_MEMBER, HAS_GENRE, etc.)
4. Create vector indexes for semantic search

### Getting Recommendations

```python
from recommender import MovieRecommender

# Initialize recommender
recommender = MovieRecommender(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="<your-password>",
    ollama_url="http://localhost:11434",
    llm_model="qwen3:8b",
    embed_model="nomic-embed-text:latest"
)

# Get recommendations
query = "dark psychological thriller like Inception"
recommendations = recommender.recommend(query)
print(recommendations)

# Close connection
recommender.close()
```

### Example Queries

- `"movies like Interstellar"`
- `"funny romantic comedies with Tom Hanks"`
- `"sci-fi movies directed by Christopher Nolan"`
- `"dark thriller about revenge"`
- `"adventure movies similar to Indiana Jones"`

## Project Structure

```
.
├── docker-compose.yml       # Neo4j container configuration
├── graph_builder.py         # Knowledge graph construction
├── embedder.py             # Ollama embedding generation
├── recommender.py          # Main recommendation engine
├── prompts.py              # LLM prompt templates
├── requirements.txt        # Python dependencies
├── schema.yaml            # Knowledge graph schema definition
└── README.md              # This file
```

## How It Works

### 1. Query Decomposition

The LLM extracts structured information from natural language:
- Reference movie (e.g., "Inception")
- Genres (e.g., ["Science Fiction", "Thriller"])
- Directors, actors, keywords
- Mood/theme description

### 2. Knowledge Graph Expansion

If a reference movie is mentioned:
- Fetch all connected entities (directors, genres, actors, keywords)
- Use these connections to boost similar movies in search results

### 3. Hybrid Search

Combines two scoring approaches:

**Semantic Score**: Cosine similarity between query embedding and movie embeddings

**Graph Boost**: Additional points for shared:
- Directors (+0.25 per match)
- Genres (+0.08 per match)
- Actors (+0.08 per match)
- Keywords (+0.04 per match)

**Combined Score** = Semantic Score + Graph Boost

### 4. LLM Ranking

The top 15 candidates are passed to the LLM, which:
- Considers the original query intent
- Evaluates why each movie matches
- Returns the top 5 with explanations

## Schema

### Node Types
- **Movie**: TMDB movies with metadata and embeddings
- **Person**: Actors and directors
- **Genre**: Movie genres (Action, Drama, etc.)
- **Keyword**: Descriptive tags (e.g., "time travel", "revenge")
- **Language**: Spoken languages

### Relationships
- `(Movie)-[:DIRECTED_BY]->(Person)`
- `(Movie)-[:HAS_CAST_MEMBER]->(Person)`
- `(Movie)-[:HAS_GENRE]->(Genre)`
- `(Movie)-[:HAS_KEYWORD]->(Keyword)`
- `(Movie)-[:HAS_LANGUAGE]->(Language)`

See `schema.yaml` for complete details.

## Configuration

### Neo4j Memory Settings

Adjust in `docker-compose.yml`:

```yaml
- NEO4J_dbms_memory_heap_initial__size=1G
- NEO4J_dbms_memory_heap_max__size=2G
- NEO4J_dbms_memory_pagecache_size=1G
```

### Embedding Model

Change in `embedder.py` or pass to `MovieGraphBuilder`:

```python
embedder = OllamaEmbedder(
    base_url="http://localhost:11434",
    model="nomic-embed-text:latest"  # 768 dimensions
)
```

### LLM Model

Change in `recommender.py`:

```python
recommender = MovieRecommender(
    llm_model="qwen3:8b",  # or "llama2", "mistral", etc.
    embed_model="nomic-embed-text:latest"
)
```

## Performance

- **Embedding Generation**: ~2-5 seconds per movie (depending on plot length)
- **Query Processing**: ~1-3 seconds (includes LLM decomposition + search + ranking)
- **Vector Search**: Sub-second for 10k+ movies with proper indexing

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Verify models are installed
ollama list
```

### Neo4j Connection Issues

```bash
# Check container status
docker ps

# View Neo4j logs
docker logs knowledge_graph

# Access Neo4j browser
open http://localhost:7474
```

### Vector Index Not Created

If vector search fails, manually create the index in Neo4j Browser:

```cypher
CREATE VECTOR INDEX movie_embedding IF NOT EXISTS
FOR (n:Movie)
ON n.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, feedback, or to request the Neo4j movie dataset:
- **Email**: [Contact me for data access]

## Acknowledgments

- **TMDB**: Movie metadata and information
- **Wikipedia**: Extended plot descriptions
- **Ollama**: Self-hosted LLM infrastructure
- **Neo4j**: Graph database platform
- **nomic-embed-text**: High-quality embeddings model

## References

- [Neo4j Vector Search Documentation](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [TMDB API](https://www.themoviedb.org/documentation/api)

---

Built with love for movie lovers who want intelligent, offline recommendations
