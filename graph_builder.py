import logging
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
import json
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from embedder import OllamaEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models for Movie Knowledge Graph
# ============================================================================

@dataclass
class MovieNode:
    """Data class for Movie node"""
    tmdb_id: int
    title: str
    original_title: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None
    year: Optional[int] = None
    runtime: Optional[int] = None
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    popularity: Optional[float] = None
    budget: Optional[int] = None
    revenue: Optional[int] = None
    language: Optional[str] = None
    tagline: Optional[str] = None
    homepage: Optional[str] = None
    imdb_id: Optional[str] = None
    wikipedia_plot: Optional[str] = None
    combined_plot: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and datetime fields"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None and key not in ('created_at', 'updated_at'):
                result[key] = value
        return result


@dataclass
class PersonNode:
    """Data class for Person node (actors and directors)"""
    name: str
    tmdb_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {'name': self.name}
        if self.tmdb_id:
            result['tmdb_id'] = self.tmdb_id
        return result


@dataclass
class GenreNode:
    """Data class for Genre node"""
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name}


@dataclass
class KeywordNode:
    """Data class for Keyword node"""
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name}


@dataclass
class LanguageNode:
    """Data class for Language node"""
    code: str
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {'code': self.code, 'name': self.name}


@dataclass
class MovieRelationship:
    """Data class for relationships from Movie to other nodes"""
    rel_type: str  # DIRECTED_BY, HAS_CAST_MEMBER, HAS_GENRE, HAS_KEYWORD, HAS_LANGUAGE
    movie_tmdb_id: int
    target_name: str  # Name of the target node
    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Movie Knowledge Graph Builder
# ============================================================================

class MovieGraphBuilder:
    """Builder for Movie Knowledge Graph with vector embeddings on Movie nodes"""
    
    def __init__(self, uri: str, username: str, password: str,
                 vector_dimensions: int = 768,
                 ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text:latest"):
        """Initialize the graph builder
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            vector_dimensions: Dimensions of the embedding vectors (768 for nomic-embed-text)
            ollama_url: URL for Ollama API
            embedding_model: Name of embedding model to use
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.vector_dimensions = vector_dimensions
        
        # Initialize embedder
        self.embedder = OllamaEmbedder(base_url=ollama_url, model=embedding_model)
        
        # Initialize Neo4j schema
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes"""
        try:
            with self.driver.session() as session:
                # Create constraints for all node types
                constraints = [
                    ("Movie", "tmdb_id"),
                    ("Person", "name"),
                    ("Genre", "name"),
                    ("Keyword", "name"),
                    ("Language", "code")
                ]
                
                for label, prop in constraints:
                    try:
                        session.run(f"""
                            CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label})
                            REQUIRE n.{prop} IS UNIQUE
                        """)
                        logger.info(f"Created/verified constraint for {label}.{prop}")
                    except Exception as e:
                        logger.warning(f"Could not create constraint for {label}.{prop}: {e}")
                
                # Create vector index for Movie embeddings only
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX movie_embedding IF NOT EXISTS
                        FOR (n:Movie)
                        ON n.embedding
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {self.vector_dimensions},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    logger.info("Created vector index for Movie embeddings")
                except Exception as e:
                    logger.warning(f"Could not create vector index: {e}")
                    logger.warning("Vector search may not be available")
                
                # Create text indexes for search
                text_indexes = [
                    ("Movie", "title"),
                    ("Person", "name"),
                    ("Genre", "name"),
                    ("Keyword", "name")
                ]
                
                for label, prop in text_indexes:
                    try:
                        session.run(f"""
                            CREATE TEXT INDEX {label.lower()}_{prop}_text IF NOT EXISTS
                            FOR (n:{label})
                            ON (n.{prop})
                        """)
                        logger.debug(f"Created text index for {label}.{prop}")
                    except Exception as e:
                        logger.debug(f"Could not create text index for {label}.{prop}: {e}")
                
                logger.info("Initialized Neo4j schema for Movie Knowledge Graph")
                
        except Exception as e:
            logger.error(f"Error initializing schema: {str(e)}")
            logger.error(traceback.format_exc())
    
    # =========================================================================
    # Node Creation Methods
    # =========================================================================
    
    def create_movie_node(self, movie: MovieNode, 
                          genres: Optional[List[str]] = None,
                          keywords: Optional[List[str]] = None) -> bool:
        """Create a Movie node with embedding
        
        Args:
            movie: MovieNode instance
            genres: Optional list of genre names (for embedding context)
            keywords: Optional list of keywords (for embedding context)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding if not already present
            if not movie.embedding:
                embedding = self.embedder.embed_movie(
                    title=movie.title,
                    wikipedia_plot=movie.wikipedia_plot,
                    overview=movie.overview,
                    year=movie.year,
                    genres=genres,
                    keywords=keywords
                )
                
                if embedding:
                    movie.embedding = embedding
                    logger.info(f"Generated embedding for movie '{movie.title}'")
                else:
                    logger.warning(f"Failed to generate embedding for movie '{movie.title}'")
            
            # Prepare properties
            properties = movie.to_dict()
            
            with self.driver.session() as session:
                cypher = """
                MERGE (m:Movie {tmdb_id: $tmdb_id})
                ON CREATE SET 
                    m = $properties,
                    m.created_at = datetime($created_at),
                    m.updated_at = datetime($updated_at)
                ON MATCH SET 
                    m += $properties,
                    m.updated_at = datetime($updated_at)
                RETURN m
                """
                
                result = session.run(
                    cypher,
                    tmdb_id=movie.tmdb_id,
                    properties=properties,
                    created_at=movie.created_at.isoformat(),
                    updated_at=datetime.utcnow().isoformat()
                )
                
                record = result.single()
                if record:
                    logger.info(f"Created/updated Movie node: {movie.title} (ID: {movie.tmdb_id})")
                    return True
                else:
                    logger.warning(f"Failed to create Movie node: {movie.title}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating Movie node '{movie.title}': {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def create_person_node(self, person: PersonNode) -> bool:
        """Create a Person node (no embedding)
        
        Args:
            person: PersonNode instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                cypher = """
                MERGE (p:Person {name: $name})
                ON CREATE SET p = $properties
                ON MATCH SET p += $properties
                RETURN p
                """
                
                result = session.run(
                    cypher,
                    name=person.name,
                    properties=person.to_dict()
                )
                
                record = result.single()
                if record:
                    logger.debug(f"Created/updated Person node: {person.name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error creating Person node '{person.name}': {str(e)}")
            return False
    
    def create_genre_node(self, genre: GenreNode) -> bool:
        """Create a Genre node (no embedding)"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MERGE (g:Genre {name: $name})
                    RETURN g
                """, name=genre.name)
                
                if result.single():
                    logger.debug(f"Created/updated Genre node: {genre.name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error creating Genre node '{genre.name}': {str(e)}")
            return False
    
    def create_keyword_node(self, keyword: KeywordNode) -> bool:
        """Create a Keyword node (no embedding)"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MERGE (k:Keyword {name: $name})
                    RETURN k
                """, name=keyword.name)
                
                if result.single():
                    logger.debug(f"Created/updated Keyword node: {keyword.name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error creating Keyword node '{keyword.name}': {str(e)}")
            return False
    
    def create_language_node(self, language: LanguageNode) -> bool:
        """Create a Language node (no embedding)"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MERGE (l:Language {code: $code})
                    ON CREATE SET l.name = $name
                    ON MATCH SET l.name = $name
                    RETURN l
                """, code=language.code, name=language.name)
                
                if result.single():
                    logger.debug(f"Created/updated Language node: {language.name} ({language.code})")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error creating Language node '{language.code}': {str(e)}")
            return False
    
    # =========================================================================
    # Relationship Creation Methods
    # =========================================================================
    
    def create_relationship(self, rel: MovieRelationship) -> bool:
        """Create a relationship from a Movie to another node
        
        Args:
            rel: MovieRelationship instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Build the appropriate Cypher based on relationship type
                if rel.rel_type == "DIRECTED_BY":
                    cypher = """
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MATCH (p:Person {name: $target_name})
                    MERGE (m)-[r:DIRECTED_BY]->(p)
                    RETURN r
                    """
                elif rel.rel_type == "HAS_CAST_MEMBER":
                    cypher = """
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MATCH (p:Person {name: $target_name})
                    MERGE (m)-[r:HAS_CAST_MEMBER]->(p)
                    ON CREATE SET r = $properties
                    ON MATCH SET r += $properties
                    RETURN r
                    """
                elif rel.rel_type == "HAS_GENRE":
                    cypher = """
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MATCH (g:Genre {name: $target_name})
                    MERGE (m)-[r:HAS_GENRE]->(g)
                    RETURN r
                    """
                elif rel.rel_type == "HAS_KEYWORD":
                    cypher = """
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MATCH (k:Keyword {name: $target_name})
                    MERGE (m)-[r:HAS_KEYWORD]->(k)
                    RETURN r
                    """
                elif rel.rel_type == "HAS_LANGUAGE":
                    cypher = """
                    MATCH (m:Movie {tmdb_id: $movie_id})
                    MATCH (l:Language {code: $target_name})
                    MERGE (m)-[r:HAS_LANGUAGE]->(l)
                    ON CREATE SET r = $properties
                    ON MATCH SET r += $properties
                    RETURN r
                    """
                else:
                    logger.error(f"Unknown relationship type: {rel.rel_type}")
                    return False
                
                result = session.run(
                    cypher,
                    movie_id=rel.movie_tmdb_id,
                    target_name=rel.target_name,
                    properties=rel.properties
                )
                
                if result.single():
                    logger.debug(f"Created relationship: Movie({rel.movie_tmdb_id}) -[{rel.rel_type}]-> {rel.target_name}")
                    return True
                else:
                    logger.warning(f"Failed to create relationship: {rel.rel_type}")
                    return False
                
        except Exception as e:
            logger.error(f"Error creating relationship {rel.rel_type}: {str(e)}")
            return False
    
    # =========================================================================
    # Convenience Methods for Loading Data
    # =========================================================================
    
    def load_movie_from_dict(self, movie_data: Dict[str, Any]) -> bool:
        """Load a complete movie with all related nodes and relationships
        
        Args:
            movie_data: Dictionary containing movie data with structure:
                {
                    "tmdb_id": int,
                    "title": str,
                    "overview": str,
                    "wikipedia_plot": str,
                    ... other movie fields ...,
                    "directors": [{"name": str}, ...],
                    "cast": [{"name": str, "character": str, "order": int}, ...],
                    "genres": [{"name": str}, ...],
                    "keywords": [{"name": str}, ...],
                    "languages": [{"code": str, "name": str}, ...]
                }
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract genres and keywords for embedding context
            genres = [g.get('name') for g in movie_data.get('genres', []) if g.get('name')]
            keywords = [k.get('name') for k in movie_data.get('keywords', []) if k.get('name')]
            
            # Create Movie node
            movie = MovieNode(
                tmdb_id=movie_data['tmdb_id'],
                title=movie_data['title'],
                original_title=movie_data.get('original_title'),
                overview=movie_data.get('overview'),
                release_date=movie_data.get('release_date'),
                year=movie_data.get('year'),
                runtime=movie_data.get('runtime'),
                vote_average=movie_data.get('vote_average'),
                vote_count=movie_data.get('vote_count'),
                popularity=movie_data.get('popularity'),
                budget=movie_data.get('budget'),
                revenue=movie_data.get('revenue'),
                language=movie_data.get('language'),
                tagline=movie_data.get('tagline'),
                homepage=movie_data.get('homepage'),
                imdb_id=movie_data.get('imdb_id'),
                wikipedia_plot=movie_data.get('wikipedia_plot'),
                combined_plot=movie_data.get('combined_plot')
            )
            
            if not self.create_movie_node(movie, genres=genres, keywords=keywords):
                return False
            
            # Create Director nodes and relationships
            for director in movie_data.get('directors', []):
                person = PersonNode(name=director.get('name'), tmdb_id=director.get('tmdb_id'))
                self.create_person_node(person)
                self.create_relationship(MovieRelationship(
                    rel_type="DIRECTED_BY",
                    movie_tmdb_id=movie.tmdb_id,
                    target_name=director.get('name')
                ))
            
            # Create Cast nodes and relationships
            for cast_member in movie_data.get('cast', []):
                person = PersonNode(name=cast_member.get('name'), tmdb_id=cast_member.get('tmdb_id'))
                self.create_person_node(person)
                self.create_relationship(MovieRelationship(
                    rel_type="HAS_CAST_MEMBER",
                    movie_tmdb_id=movie.tmdb_id,
                    target_name=cast_member.get('name'),
                    properties={
                        'character': cast_member.get('character'),
                        'order': cast_member.get('order')
                    }
                ))
            
            # Create Genre nodes and relationships
            for genre in movie_data.get('genres', []):
                self.create_genre_node(GenreNode(name=genre.get('name')))
                self.create_relationship(MovieRelationship(
                    rel_type="HAS_GENRE",
                    movie_tmdb_id=movie.tmdb_id,
                    target_name=genre.get('name')
                ))
            
            # Create Keyword nodes and relationships
            for keyword in movie_data.get('keywords', []):
                self.create_keyword_node(KeywordNode(name=keyword.get('name')))
                self.create_relationship(MovieRelationship(
                    rel_type="HAS_KEYWORD",
                    movie_tmdb_id=movie.tmdb_id,
                    target_name=keyword.get('name')
                ))
            
            # Create Language nodes and relationships
            for lang in movie_data.get('languages', []):
                self.create_language_node(LanguageNode(code=lang.get('code'), name=lang.get('name')))
                self.create_relationship(MovieRelationship(
                    rel_type="HAS_LANGUAGE",
                    movie_tmdb_id=movie.tmdb_id,
                    target_name=lang.get('code'),
                    properties={'type': lang.get('type', 'spoken')}
                ))
            
            logger.info(f"Successfully loaded movie: {movie.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading movie from dict: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def load_movies_batch(self, movies: List[Dict[str, Any]], batch_size: int = 100) -> Tuple[int, int]:
        """Load multiple movies in batch
        
        Args:
            movies: List of movie dictionaries
            batch_size: Number of movies to process before logging progress
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        total = len(movies)
        
        for i, movie_data in enumerate(movies):
            try:
                if self.load_movie_from_dict(movie_data):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error processing movie at index {i}: {e}")
                failed += 1
            
            if (i + 1) % batch_size == 0:
                logger.info(f"Progress: {i + 1}/{total} movies processed ({successful} successful, {failed} failed)")
        
        logger.info(f"Batch complete: {successful}/{total} successful, {failed} failed")
        return successful, failed
    
    # =========================================================================
    # Vector Search Methods
    # =========================================================================
    
    def vector_search_movies(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for movies by semantic similarity using vector search
        
        Args:
            query_text: Query text (e.g., plot description, theme)
            limit: Maximum number of results
            
        Returns:
            List of movies with similarity scores
        """
        try:
            # Get embedding for query
            query_embedding = self.embedder.get_embedding(query_text)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            with self.driver.session() as session:
                # Check if vector index exists
                index_check = session.run("""
                    SHOW INDEXES
                    YIELD name, type
                    WHERE name = 'movie_embedding' AND type = 'VECTOR'
                    RETURN count(*) > 0 AS exists
                """)
                
                vector_index_exists = index_check.single()[0]
                
                if vector_index_exists:
                    # Use vector index for efficient search
                    cypher = """
                    CALL db.index.vector.queryNodes('movie_embedding', $limit, $embedding)
                    YIELD node, score
                    RETURN 
                        node.tmdb_id AS tmdb_id,
                        node.title AS title,
                        node.year AS year,
                        node.overview AS overview,
                        node.vote_average AS rating,
                        score AS similarity
                    ORDER BY similarity DESC
                    """
                else:
                    # Fallback to manual cosine similarity (slower)
                    logger.warning("Vector index not found, using manual similarity calculation")
                    cypher = """
                    MATCH (m:Movie)
                    WHERE m.embedding IS NOT NULL
                    WITH m, gds.similarity.cosine(m.embedding, $embedding) AS similarity
                    WHERE similarity > 0.5
                    RETURN 
                        m.tmdb_id AS tmdb_id,
                        m.title AS title,
                        m.year AS year,
                        m.overview AS overview,
                        m.vote_average AS rating,
                        similarity
                    ORDER BY similarity DESC
                    LIMIT $limit
                    """
                
                result = session.run(cypher, embedding=query_embedding, limit=limit)
                
                movies = []
                for record in result:
                    movies.append({
                        'tmdb_id': record['tmdb_id'],
                        'title': record['title'],
                        'year': record['year'],
                        'overview': record['overview'],
                        'rating': record['rating'],
                        'similarity': record['similarity']
                    })
                
                return movies
                
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def find_similar_movies(self, tmdb_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Find movies similar to a given movie based on embedding similarity
        
        Args:
            tmdb_id: TMDB ID of the source movie
            limit: Maximum number of similar movies to return
            
        Returns:
            List of similar movies with similarity scores
        """
        try:
            with self.driver.session() as session:
                # Get the embedding of the source movie
                result = session.run("""
                    MATCH (m:Movie {tmdb_id: $tmdb_id})
                    RETURN m.embedding AS embedding, m.title AS title
                """, tmdb_id=tmdb_id)
                
                record = result.single()
                if not record or not record['embedding']:
                    logger.warning(f"Movie {tmdb_id} not found or has no embedding")
                    return []
                
                source_title = record['title']
                source_embedding = record['embedding']
                
                # Find similar movies using vector index
                cypher = """
                CALL db.index.vector.queryNodes('movie_embedding', $limit_plus_one, $embedding)
                YIELD node, score
                WHERE node.tmdb_id <> $tmdb_id
                RETURN 
                    node.tmdb_id AS tmdb_id,
                    node.title AS title,
                    node.year AS year,
                    node.overview AS overview,
                    node.vote_average AS rating,
                    score AS similarity
                ORDER BY similarity DESC
                LIMIT $limit
                """
                
                result = session.run(
                    cypher,
                    embedding=source_embedding,
                    tmdb_id=tmdb_id,
                    limit_plus_one=limit + 1,
                    limit=limit
                )
                
                similar_movies = []
                for record in result:
                    similar_movies.append({
                        'tmdb_id': record['tmdb_id'],
                        'title': record['title'],
                        'year': record['year'],
                        'overview': record['overview'],
                        'rating': record['rating'],
                        'similarity': record['similarity']
                    })
                
                logger.info(f"Found {len(similar_movies)} movies similar to '{source_title}'")
                return similar_movies
                
        except Exception as e:
            logger.error(f"Error finding similar movies: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a movie including all relationships
        
        Args:
            tmdb_id: TMDB ID of the movie
            
        Returns:
            Dictionary with movie details or None if not found
        """
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (m:Movie {tmdb_id: $tmdb_id})
                OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Person)
                OPTIONAL MATCH (m)-[cast:HAS_CAST_MEMBER]->(a:Person)
                OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
                OPTIONAL MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)
                OPTIONAL MATCH (m)-[:HAS_LANGUAGE]->(l:Language)
                RETURN 
                    m.tmdb_id AS tmdb_id,
                    m.title AS title,
                    m.year AS year,
                    m.overview AS overview,
                    m.wikipedia_plot AS wikipedia_plot,
                    m.runtime AS runtime,
                    m.vote_average AS rating,
                    m.budget AS budget,
                    m.revenue AS revenue,
                    m.tagline AS tagline,
                    m.imdb_id AS imdb_id,
                    collect(DISTINCT d.name) AS directors,
                    collect(DISTINCT {name: a.name, character: cast.character, order: cast.order}) AS cast,
                    collect(DISTINCT g.name) AS genres,
                    collect(DISTINCT k.name) AS keywords,
                    collect(DISTINCT {code: l.code, name: l.name}) AS languages
                """
                
                result = session.run(cypher, tmdb_id=tmdb_id)
                record = result.single()
                
                if not record:
                    return None
                
                return dict(record)
                
        except Exception as e:
            logger.error(f"Error getting movie details: {str(e)}")
            return None
    
    def get_movies_by_person(self, person_name: str, role: str = "any") -> List[Dict[str, Any]]:
        """Get all movies associated with a person
        
        Args:
            person_name: Name of the person
            role: "director", "actor", or "any"
            
        Returns:
            List of movies
        """
        try:
            with self.driver.session() as session:
                if role == "director":
                    cypher = """
                    MATCH (m:Movie)-[:DIRECTED_BY]->(p:Person {name: $name})
                    RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.year AS year, 
                           m.vote_average AS rating, 'director' AS role
                    ORDER BY m.year DESC
                    """
                elif role == "actor":
                    cypher = """
                    MATCH (m:Movie)-[r:HAS_CAST_MEMBER]->(p:Person {name: $name})
                    RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.year AS year,
                           m.vote_average AS rating, 'actor' AS role, r.character AS character
                    ORDER BY m.year DESC
                    """
                else:  # any
                    cypher = """
                    MATCH (m:Movie)-[r]->(p:Person {name: $name})
                    WHERE type(r) IN ['DIRECTED_BY', 'HAS_CAST_MEMBER']
                    RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.year AS year,
                           m.vote_average AS rating, 
                           CASE type(r) WHEN 'DIRECTED_BY' THEN 'director' ELSE 'actor' END AS role,
                           CASE type(r) WHEN 'HAS_CAST_MEMBER' THEN r.character ELSE null END AS character
                    ORDER BY m.year DESC
                    """
                
                result = session.run(cypher, name=person_name)
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"Error getting movies by person: {str(e)}")
            return []
    
    def get_movies_by_genre(self, genre_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get movies by genre
        
        Args:
            genre_name: Name of the genre
            limit: Maximum number of results
            
        Returns:
            List of movies in the genre
        """
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {name: $genre})
                RETURN m.tmdb_id AS tmdb_id, m.title AS title, m.year AS year,
                       m.vote_average AS rating, m.overview AS overview
                ORDER BY m.vote_average DESC, m.vote_count DESC
                LIMIT $limit
                """
                
                result = session.run(cypher, genre=genre_name, limit=limit)
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"Error getting movies by genre: {str(e)}")
            return []
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the knowledge graph
        
        Returns:
            Dictionary with node and relationship counts
        """
        try:
            with self.driver.session() as session:
                stats = {}
                
                # Count nodes by type
                for label in ['Movie', 'Person', 'Genre', 'Keyword', 'Language']:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                    stats[f'{label.lower()}_count'] = result.single()['count']
                
                # Count movies with embeddings
                result = session.run("""
                    MATCH (m:Movie) WHERE m.embedding IS NOT NULL
                    RETURN count(m) AS count
                """)
                stats['movies_with_embedding'] = result.single()['count']
                
                # Count relationships by type
                for rel_type in ['DIRECTED_BY', 'HAS_CAST_MEMBER', 'HAS_GENRE', 'HAS_KEYWORD', 'HAS_LANGUAGE']:
                    result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count")
                    stats[f'{rel_type.lower()}_count'] = result.single()['count']
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {}
    
    def batch_create_nodes(self, node_type: str, nodes_data: List[Dict[str, Any]]) -> int:
        """Batch create nodes in a single transaction (much faster)
        
        Args:
            node_type: Node label (Movie, Person, Genre, Keyword, Language)
            nodes_data: List of node property dicts
            
        Returns:
            Number of nodes created
        """
        if not nodes_data:
            return 0
            
        try:
            with self.driver.session() as session:
                if node_type == "Movie":
                    cypher = """
                    UNWIND $nodes AS node
                    MERGE (n:Movie {tmdb_id: node.tmdb_id})
                    SET n += node
                    """
                elif node_type == "Person":
                    cypher = """
                    UNWIND $nodes AS node
                    MERGE (n:Person {name: node.name})
                    SET n += node
                    """
                elif node_type == "Genre":
                    cypher = """
                    UNWIND $nodes AS node
                    MERGE (n:Genre {name: node.name})
                    """
                elif node_type == "Keyword":
                    cypher = """
                    UNWIND $nodes AS node
                    MERGE (n:Keyword {name: node.name})
                    """
                elif node_type == "Language":
                    cypher = """
                    UNWIND $nodes AS node
                    MERGE (n:Language {code: node.code})
                    SET n.name = node.name
                    """
                else:
                    logger.error(f"Unknown node type: {node_type}")
                    return 0
                
                result = session.run(cypher, nodes=nodes_data)
                summary = result.consume()
                count = summary.counters.nodes_created
                logger.info(f"Batch created {count} {node_type} nodes")
                return count
                
        except Exception as e:
            logger.error(f"Error batch creating {node_type} nodes: {e}")
            return 0
    
    def batch_create_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """Batch create relationships in a single transaction (much faster)
        
        Args:
            relationships: List of dicts with keys: type, movie_tmdb_id, target_name, properties
            
        Returns:
            Number of relationships created
        """
        if not relationships:
            return 0
        
        try:
            with self.driver.session() as session:
                # Group by relationship type for efficient batch processing
                by_type = {}
                for rel in relationships:
                    rel_type = rel['type']
                    if rel_type not in by_type:
                        by_type[rel_type] = []
                    by_type[rel_type].append(rel)
                
                total_created = 0
                
                for rel_type, rels in by_type.items():
                    if rel_type == "DIRECTED_BY":
                        cypher = """
                        UNWIND $rels AS rel
                        MATCH (m:Movie {tmdb_id: rel.movie_tmdb_id})
                        MATCH (p:Person {name: rel.target_name})
                        MERGE (m)-[r:DIRECTED_BY]->(p)
                        """
                    elif rel_type == "HAS_CAST_MEMBER":
                        cypher = """
                        UNWIND $rels AS rel
                        MATCH (m:Movie {tmdb_id: rel.movie_tmdb_id})
                        MATCH (p:Person {name: rel.target_name})
                        MERGE (m)-[r:HAS_CAST_MEMBER]->(p)
                        SET r += rel.properties
                        """
                    elif rel_type == "HAS_GENRE":
                        cypher = """
                        UNWIND $rels AS rel
                        MATCH (m:Movie {tmdb_id: rel.movie_tmdb_id})
                        MATCH (g:Genre {name: rel.target_name})
                        MERGE (m)-[r:HAS_GENRE]->(g)
                        """
                    elif rel_type == "HAS_KEYWORD":
                        cypher = """
                        UNWIND $rels AS rel
                        MATCH (m:Movie {tmdb_id: rel.movie_tmdb_id})
                        MATCH (k:Keyword {name: rel.target_name})
                        MERGE (m)-[r:HAS_KEYWORD]->(k)
                        """
                    elif rel_type == "HAS_LANGUAGE":
                        cypher = """
                        UNWIND $rels AS rel
                        MATCH (m:Movie {tmdb_id: rel.movie_tmdb_id})
                        MATCH (l:Language {code: rel.target_name})
                        MERGE (m)-[r:HAS_LANGUAGE]->(l)
                        SET r += rel.properties
                        """
                    else:
                        logger.warning(f"Unknown relationship type: {rel_type}")
                        continue
                    
                    result = session.run(cypher, rels=rels)
                    summary = result.consume()
                    total_created += summary.counters.relationships_created
                
                logger.info(f"Batch created {total_created} relationships")
                return total_created
                
        except Exception as e:
            logger.error(f"Error batch creating relationships: {e}")
            return 0
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Movie Knowledge Graph from JSON data')
    parser.add_argument('json_file', type=str, help='Path to JSON file containing movie data')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--username', type=str, default='neo4j', help='Neo4j username')
    parser.add_argument('--password', type=str, default='', help='Neo4j password')
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = MovieGraphBuilder(
        uri=args.uri,
        username=args.username,
        password=args.password
    )
    
    # Load JSON file
    print(f"Loading data from: {args.json_file}")
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle JSON structure: {"nodes": {...}, "relationships": [...]}
    if 'nodes' in data:
        nodes = data['nodes']
        relationships = data.get('relationships', [])
        
        # Build movie title -> tmdb_id lookup
        movies = nodes.get('movies', [])
        title_to_id = {m['title']: m['tmdb_id'] for m in movies}
        
        # ===== STEP 1: Generate embeddings for movies =====
        print("\n--- Generating Movie Embeddings ---")
        for i, movie in enumerate(movies):
            # Get genres/keywords for this movie from relationships
            genres = [r['to'] for r in relationships 
                      if r['type'] == 'HAS_GENRE' and r['from'] == movie['title']]
            keywords = [r['to'] for r in relationships 
                        if r['type'] == 'HAS_KEYWORD' and r['from'] == movie['title']]
            
            embedding = builder.embedder.embed_movie(
                title=movie['title'],
                wikipedia_plot=movie.get('wikipedia_plot'),
                overview=movie.get('overview'),
                year=movie.get('year'),
                genres=genres,
                keywords=keywords
            )
            if embedding:
                movie['embedding'] = embedding
            
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(movies)} movies")
        
        print(f"  Completed: {len(movies)} movies embedded")
        
        # ===== STEP 2: Batch create all nodes =====
        print("\n--- Batch Loading Nodes ---")
        
        # Movies
        builder.batch_create_nodes("Movie", movies)
        
        # Persons
        persons = nodes.get('persons', [])
        builder.batch_create_nodes("Person", persons)
        
        # Genres
        genres = nodes.get('genres', [])
        builder.batch_create_nodes("Genre", genres)
        
        # Keywords
        keywords = nodes.get('keywords', [])
        builder.batch_create_nodes("Keyword", keywords)
        
        # Languages
        languages = nodes.get('languages', [])
        builder.batch_create_nodes("Language", languages)
        
        # ===== STEP 3: Batch create all relationships =====
        print("\n--- Batch Loading Relationships ---")
        
        # Prepare relationships with movie_tmdb_id
        rel_data = []
        for rel in relationships:
            movie_id = title_to_id.get(rel['from'])
            if not movie_id:
                continue
            
            target = rel['to']
            # For HAS_LANGUAGE, convert name to code
            if rel['type'] == 'HAS_LANGUAGE':
                lang_match = next((l for l in languages if l['name'] == target), None)
                if lang_match:
                    target = lang_match['code']
            
            rel_data.append({
                'type': rel['type'],
                'movie_tmdb_id': movie_id,
                'target_name': target,
                'properties': rel.get('properties', {})
            })
        
        builder.batch_create_relationships(rel_data)
        
    else:
        # Fallback: Handle flat list or single movie dict
        if isinstance(data, list):
            print(f"Found {len(data)} movies")
            successful, failed = builder.load_movies_batch(data)
            print(f"\nCompleted: {successful} successful, {failed} failed")
        else:
            print("Loading single movie...")
            builder.load_movie_from_dict(data)
    
    # Print graph stats
    print("\n--- Graph Statistics ---")
    stats = builder.get_graph_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    builder.close()