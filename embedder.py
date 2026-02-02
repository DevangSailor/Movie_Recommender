import requests
import logging
from typing import Dict, List, Any, Optional, Union
import json
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Class to generate embeddings using Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text:latest"):
        """Initialize the embedder with Ollama API settings
        
        Args:
            base_url: Base URL for Ollama API
            model: Embedding model to use (default: nomic-embed-text, 768 dimensions)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.embed_endpoint = f"{self.base_url}/api/embeddings"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Ollama API"""
        try:
            # Simple request to check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                logger.warning(f"Ollama API returned status code {response.status_code}")
            else:
                available_models = [model.get('name') for model in response.json().get('models', [])]
                if self.model not in available_models:
                    logger.warning(f"Model {self.model} not found in available models: {available_models}")
                else:
                    logger.info(f"Successfully connected to Ollama API, model {self.model} is available")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama API: {str(e)}")
            logger.error("Make sure Ollama is running and the URL is correct")
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using Ollama API
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding vector, or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
            
        try:
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(self.embed_endpoint, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.status_code} {response.text}")
                return None
            
            result = response.json()
            embedding = result.get('embedding')
            
            if not embedding:
                logger.error(f"No embedding returned from Ollama API: {result}")
                return None
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return None

    # =========================================================================
    # Movie Embedding Methods
    # =========================================================================
    
    def create_movie_embedding_text(self,
                                    title: str,
                                    wikipedia_plot: Optional[str] = None,
                                    overview: Optional[str] = None,
                                    year: Optional[int] = None,
                                    genres: Optional[List[str]] = None,
                                    keywords: Optional[List[str]] = None) -> str:
        """Create text for movie embedding from plot information
        
        The embedding focuses on the narrative content (wikipedia_plot and overview)
        to enable semantic search based on plot descriptions.
        
        Args:
            title: Movie title
            wikipedia_plot: Extended plot summary from Wikipedia
            overview: Short plot summary from TMDB
            year: Release year (optional, for context)
            genres: List of genre names (optional, for context)
            keywords: List of keywords (optional, for context)
            
        Returns:
            Text string to be embedded
        """
        text_parts = []
        
        # Add title for context
        text_parts.append(f"Title: {title}")
        
        # Add year if available
        if year:
            text_parts.append(f"Year: {year}")
        
        # Add genres if available (helps with semantic context)
        if genres and isinstance(genres, list) and len(genres) > 0:
            text_parts.append(f"Genres: {', '.join(genres)}")
        
        # Add keywords if available (helps with semantic context)
        if keywords and isinstance(keywords, list) and len(keywords) > 0:
            # Limit keywords to prevent overwhelming the embedding
            text_parts.append(f"Keywords: {', '.join(keywords[:10])}")
        
        # Primary: Use wikipedia_plot if available (more detailed)
        if wikipedia_plot and wikipedia_plot.strip():
            text_parts.append(f"Plot: {wikipedia_plot.strip()}")
        
        # Secondary: Add overview if available
        # If wikipedia_plot exists, overview provides additional context
        # If wikipedia_plot doesn't exist, overview is the primary content
        if overview and overview.strip():
            if wikipedia_plot and wikipedia_plot.strip():
                text_parts.append(f"Summary: {overview.strip()}")
            else:
                text_parts.append(f"Plot: {overview.strip()}")
        
        # Join all parts with newlines
        result = "\n".join(text_parts)
        
        # Log warning if no plot content available
        if not wikipedia_plot and not overview:
            logger.warning(f"No plot content available for movie: {title}")
        
        return result
    
    def embed_movie(self,
                    title: str,
                    wikipedia_plot: Optional[str] = None,
                    overview: Optional[str] = None,
                    year: Optional[int] = None,
                    genres: Optional[List[str]] = None,
                    keywords: Optional[List[str]] = None) -> Optional[List[float]]:
        """Create embedding for a movie based on plot information
        
        Args:
            title: Movie title
            wikipedia_plot: Extended plot summary from Wikipedia
            overview: Short plot summary from TMDB
            year: Release year (optional)
            genres: List of genre names (optional)
            keywords: List of keywords (optional)
            
        Returns:
            Embedding vector (768 dimensions for nomic-embed-text) or None if failed
        """
        # Create text for embedding
        text = self.create_movie_embedding_text(
            title=title,
            wikipedia_plot=wikipedia_plot,
            overview=overview,
            year=year,
            genres=genres,
            keywords=keywords
        )
        
        # Log the text being embedded (truncated for readability)
        logger.debug(f"Embedding text for '{title}': {text[:200]}...")
        
        # Get embedding
        embedding = self.get_embedding(text)
        
        if embedding:
            logger.info(f"Generated embedding for movie '{title}' with {len(embedding)} dimensions")
        else:
            logger.warning(f"Failed to generate embedding for movie '{title}'")
        
        return embedding
    
    def embed_movie_from_dict(self, movie_data: Dict[str, Any]) -> Optional[List[float]]:
        """Create embedding for a movie from a dictionary of movie data
        
        This is a convenience method that extracts relevant fields from a movie dict.
        
        Args:
            movie_data: Dictionary containing movie information with keys:
                - title (required)
                - wikipedia_plot (optional)
                - overview (optional)
                - year (optional)
                - genres (optional) - can be list of strings or list of dicts with 'name' key
                - keywords (optional) - can be list of strings or list of dicts with 'name' key
                
        Returns:
            Embedding vector or None if failed
        """
        # Extract title (required)
        title = movie_data.get('title')
        if not title:
            logger.error("Movie data missing required 'title' field")
            return None
        
        # Extract plot fields
        wikipedia_plot = movie_data.get('wikipedia_plot')
        overview = movie_data.get('overview')
        
        # Extract year
        year = movie_data.get('year')
        
        # Extract genres (handle both list of strings and list of dicts)
        genres = movie_data.get('genres', [])
        if genres and isinstance(genres[0], dict):
            genres = [g.get('name') for g in genres if g.get('name')]
        
        # Extract keywords (handle both list of strings and list of dicts)
        keywords = movie_data.get('keywords', [])
        if keywords and isinstance(keywords[0], dict):
            keywords = [k.get('name') for k in keywords if k.get('name')]
        
        return self.embed_movie(
            title=title,
            wikipedia_plot=wikipedia_plot,
            overview=overview,
            year=year,
            genres=genres,
            keywords=keywords
        )
    
    def batch_embed_movies(self, 
                           movies: List[Dict[str, Any]], 
                           batch_size: int = 10) -> Dict[str, Optional[List[float]]]:
        """Generate embeddings for multiple movies
        
        Args:
            movies: List of movie dictionaries (each must have 'tmdb_id' or 'title' as identifier)
            batch_size: Number of movies to process before logging progress
            
        Returns:
            Dictionary mapping movie identifier to embedding vector (or None if failed)
        """
        results = {}
        total = len(movies)
        
        for i, movie in enumerate(movies):
            # Determine identifier
            identifier = movie.get('tmdb_id') or movie.get('title')
            if not identifier:
                logger.warning(f"Movie at index {i} has no identifier, skipping")
                continue
            
            # Generate embedding
            embedding = self.embed_movie_from_dict(movie)
            results[identifier] = embedding
            
            # Log progress
            if (i + 1) % batch_size == 0:
                logger.info(f"Processed {i + 1}/{total} movies")
        
        # Final progress log
        successful = sum(1 for e in results.values() if e is not None)
        logger.info(f"Completed embedding generation: {successful}/{total} successful")
        
        return results

    # =========================================================================
    # Legacy Table/Column Embedding Methods (kept for backward compatibility)
    # =========================================================================
    
    def create_table_embedding_text(self, 
                                  table_name: str, 
                                  module: str, 
                                  submodule: str, 
                                  description: str, 
                                  primary_key: Optional[Union[Dict[str, Any], str]] = None,
                                  columns: Optional[List[Dict[str, Any]]] = None) -> str:
        """Create text for table embedding (legacy method for Oracle tables)
        
        Args:
            table_name: Name of the table
            module: Module name
            submodule: Submodule name
            description: Table description
            primary_key: Primary key info
            columns: List of important columns
            
        Returns:
            Text string to be embedded
        """
        # Start with basic table info
        text_parts = [
            f"Table: {table_name}",
            f"Module: {module}",
            f"Submodule: {submodule}",
            f"Description: {description}"
        ]
        
        # Add primary key if available
        if primary_key:
            if isinstance(primary_key, dict):
                pk_columns = primary_key.get('columns', '')
            else:
                pk_columns = primary_key
                
            text_parts.append(f"Primary Key: {pk_columns}")
        
        # Add important columns (up to 10)
        if columns and isinstance(columns, list):
            column_texts = []
            for i, col in enumerate(columns[:10]):  # Limit to 10 columns
                if isinstance(col, dict):
                    col_name = col.get('name', '')
                    col_type = col.get('datatype', '')
                    col_desc = col.get('comments', '')
                    
                    if col_name:
                        col_text = f"{col_name} ({col_type})"
                        if col_desc:
                            col_text += f": {col_desc}"
                        column_texts.append(col_text)
                        
            if column_texts:
                text_parts.append("Important Columns: " + "; ".join(column_texts))
        
        # Join all parts with newlines
        return "\n".join(text_parts)
    
    def create_column_embedding_text(self,
                                   column_name: str,
                                   datatype: str,
                                   table_name: str,
                                   description: str = "",
                                   is_primary_key: bool = False,
                                   is_foreign_key: bool = False,
                                   references_column: str = "") -> str:
        """Create text for column embedding (legacy method for Oracle tables)
        
        Args:
            column_name: Name of the column
            datatype: Data type of the column
            table_name: Name of the parent table
            description: Column description or comments
            is_primary_key: Whether the column is part of a primary key
            is_foreign_key: Whether the column is a foreign key
            references_column: Referenced column if foreign key
            
        Returns:
            Text string to be embedded
        """
        # Start with basic column info
        text_parts = [
            f"Column: {column_name}",
            f"Data Type: {datatype}",
            f"Table: {table_name}"
        ]
        
        # Add description if available
        if description:
            text_parts.append(f"Description: {description}")
        
        # Add key information
        if is_primary_key:
            text_parts.append("This is a primary key column")
        
        if is_foreign_key and references_column:
            text_parts.append(f"This is a foreign key referencing: {references_column}")
        
        # Join all parts with newlines
        return "\n".join(text_parts)
    
    def embed_table(self, 
                   table_name: str, 
                   module: str, 
                   submodule: str, 
                   description: str, 
                   primary_key: Optional[Union[Dict[str, Any], str]] = None,
                   columns: Optional[List[Dict[str, Any]]] = None) -> Optional[List[float]]:
        """Create embedding for a table (legacy method)
        
        Args:
            table_name: Name of the table
            module: Module name
            submodule: Submodule name
            description: Table description
            primary_key: Primary key info
            columns: List of important columns
            
        Returns:
            Embedding vector or None if failed
        """
        # Create text for embedding
        text = self.create_table_embedding_text(
            table_name, module, submodule, description, primary_key, columns
        )
        
        # Get embedding
        return self.get_embedding(text)
    
    def embed_column(self,
                    column_name: str,
                    datatype: str,
                    table_name: str,
                    description: str = "",
                    is_primary_key: bool = False,
                    is_foreign_key: bool = False,
                    references_column: str = "") -> Optional[List[float]]:
        """Create embedding for a column (legacy method)
        
        Args:
            column_name: Name of the column
            datatype: Data type of the column
            table_name: Name of the parent table
            description: Column description or comments
            is_primary_key: Whether the column is part of a primary key
            is_foreign_key: Whether the column is a foreign key
            references_column: Referenced column if foreign key
            
        Returns:
            Embedding vector or None if failed
        """
        # Create text for embedding
        text = self.create_column_embedding_text(
            column_name, datatype, table_name, description,
            is_primary_key, is_foreign_key, references_column
        )
        
        # Get embedding
        return self.get_embedding(text)


# Example usage
if __name__ == "__main__":
    embedder = OllamaEmbedder()
    
    # Test basic embedding
    print("Testing basic embedding...")
    sample_text = "This is a test text for embedding"
    embedding = embedder.get_embedding(sample_text)
    if embedding:
        print(f"Got embedding with {len(embedding)} dimensions")
    else:
        print("Failed to get embedding")
    
    # Test movie embedding
    print("\nTesting movie embedding...")
    movie_data = {
        "title": "Four Rooms",
        "year": 1995,
        "overview": "It's Ted the Bellhop's first night on the job and the hotel's very unusual guests are about to place him in some outrageous situations.",
        "wikipedia_plot": "The film follows Ted, a bellhop at a once-elegant Hollywood hotel, through four increasingly bizarre segments on New Year's Eve.",
        "genres": ["Comedy"],
        "keywords": ["hotel", "new year's eve", "bellhop"]
    }
    
    movie_embedding = embedder.embed_movie_from_dict(movie_data)
    if movie_embedding:
        print(f"Got movie embedding with {len(movie_embedding)} dimensions")
        print(f"First 5 values: {movie_embedding[:5]}")
    else:
        print("Failed to get movie embedding")