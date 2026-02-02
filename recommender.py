"""
Movie Recommender System
True KG + Vector Hybrid:
1. Extract reference movie from query
2. KG expansion: get reference movie's connections
3. Graph-boosted search: semantic + shared connections scoring
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
from embedder import OllamaEmbedder
from prompts import DECOMPOSITION_PROMPT, RECOMMENDATION_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MovieRecommender:
    """Movie recommender using true KG + Vector hybrid search"""
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "",
                 ollama_url: str = "http://localhost:11434",
                 llm_model: str = "qwen3:8b",
                 embed_model: str = "nomic-embed-text:latest"):
        
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.ollama_url = ollama_url.rstrip('/')
        self.llm_model = llm_model
        self.embedder = OllamaEmbedder(base_url=ollama_url, model=embed_model)
        
        logger.info(f"Initialized recommender with LLM: {llm_model}, Embedder: {embed_model}")
    
    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """Call Qwen3 8B via Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                # Clean any remaining think tags
                if '<think>' in result:
                    result = result.split('</think>')[-1].strip()
                return result
            else:
                logger.error(f"LLM error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""
    
    def _decompose_query(self, query: str) -> Dict[str, Any]:
        """Step 1: Decompose user query - extract reference movie + filters"""
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        response = self._call_llm(prompt)
        
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            result = json.loads(response)
            logger.info(f"Decomposed: reference_movie={result.get('reference_movie')}, "
                       f"genres={result.get('genres', [])}, "
                       f"directors={result.get('directors', [])}, "
                       f"keywords={result.get('keywords', [])}")
            return result
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse decomposition, using fallback")
            return {
                "reference_movie": None,
                "keywords": [],
                "genres": [],
                "directors": [],
                "actors": [],
                "mood": query
            }
    
    def _get_reference_movie_connections(self, movie_title: str) -> Optional[Dict[str, Any]]:
        """Step 2: KG Expansion - get all connections of reference movie"""
        try:
            with self.driver.session() as session:
                # Try exact match first, then fuzzy
                result = session.run("""
                    MATCH (m:Movie)
                    WHERE toLower(m.title) = toLower($title) 
                       OR toLower(m.title) CONTAINS toLower($title)
                    WITH m ORDER BY 
                        CASE WHEN toLower(m.title) = toLower($title) THEN 0 ELSE 1 END,
                        m.vote_count DESC
                    LIMIT 1
                    
                    OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Person)
                    OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
                    OPTIONAL MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)
                    OPTIONAL MATCH (m)-[:HAS_CAST_MEMBER]->(a:Person)
                    
                    RETURN m.tmdb_id AS tmdb_id,
                           m.title AS title,
                           m.embedding AS embedding,
                           collect(DISTINCT d.name) AS directors,
                           collect(DISTINCT g.name) AS genres,
                           collect(DISTINCT k.name) AS keywords,
                           collect(DISTINCT a.name) AS actors
                """, title=movie_title)
                
                record = result.single()
                if record and record['tmdb_id']:
                    connections = {
                        'tmdb_id': record['tmdb_id'],
                        'title': record['title'],
                        'embedding': record['embedding'],
                        'directors': [d for d in record['directors'] if d],
                        'genres': [g for g in record['genres'] if g],
                        'keywords': [k for k in record['keywords'] if k],
                        'actors': [a for a in record['actors'] if a]
                    }
                    logger.info(f"Found reference movie '{connections['title']}': "
                               f"{len(connections['directors'])} directors, "
                               f"{len(connections['genres'])} genres, "
                               f"{len(connections['keywords'])} keywords, "
                               f"{len(connections['actors'])} actors")
                    return connections
                else:
                    logger.warning(f"Reference movie '{movie_title}' not found in KG")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get reference movie connections: {e}")
            return None
    
    def _graph_boosted_search(self,
                              mood: str,
                              reference_movie: Optional[Dict[str, Any]] = None,
                              genres: List[str] = None,
                              directors: List[str] = None,
                              actors: List[str] = None,
                              keywords: List[str] = None,
                              limit: int = 15) -> List[Dict[str, Any]]:
        """
        Step 3: Graph-boosted hybrid search
        - Semantic similarity (from mood or reference movie embedding)
        - Graph boost (shared connections with reference movie)
        - Filter boost (matches requested genres/directors/actors)
        """
        genres = genres or []
        directors = directors or []
        actors = actors or []
        keywords = keywords or []
        
        try:
            # Get query embedding
            if reference_movie and reference_movie.get('embedding'):
                # Use reference movie's embedding for better similarity
                query_embedding = reference_movie['embedding']
                logger.info("Using reference movie embedding for search")
            else:
                query_embedding = self.embedder.get_embedding(mood)
                logger.info("Using mood embedding for search")
            
            if not query_embedding:
                logger.error("Failed to get query embedding")
                return []
            
            # Merge reference movie connections with explicit filters
            if reference_movie:
                ref_directors = reference_movie.get('directors', [])
                ref_genres = reference_movie.get('genres', [])
                ref_keywords = reference_movie.get('keywords', [])
                ref_actors = reference_movie.get('actors', [])
                ref_tmdb_id = reference_movie.get('tmdb_id')
            else:
                ref_directors = []
                ref_genres = []
                ref_keywords = []
                ref_actors = []
                ref_tmdb_id = -1  # No exclusion
            
            # Combine explicit filters with reference connections
            all_genres = list(set(genres + ref_genres))
            all_directors = list(set(directors + ref_directors))
            all_keywords = list(set(keywords + ref_keywords))
            all_actors = list(set(actors + ref_actors))
            
            with self.driver.session() as session:
                cypher = """
                // Semantic search
                CALL db.index.vector.queryNodes('movie_embedding', $search_limit, $embedding)
                YIELD node AS m, score AS semantic_score
                
                // Exclude reference movie itself
                WHERE m.tmdb_id <> $ref_tmdb_id
                
                // Count shared directors
                OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Person)
                WHERE d.name IN $all_directors
                WITH m, semantic_score, collect(DISTINCT d.name) AS matched_directors
                
                // Count shared genres
                OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
                WHERE g.name IN $all_genres
                WITH m, semantic_score, matched_directors, collect(DISTINCT g.name) AS matched_genres
                
                // Count shared keywords
                OPTIONAL MATCH (m)-[:HAS_KEYWORD]->(k:Keyword)
                WHERE k.name IN $all_keywords
                WITH m, semantic_score, matched_directors, matched_genres, 
                     collect(DISTINCT k.name) AS matched_keywords
                
                // Count shared actors
                OPTIONAL MATCH (m)-[:HAS_CAST_MEMBER]->(a:Person)
                WHERE a.name IN $all_actors
                WITH m, semantic_score, matched_directors, matched_genres, 
                     matched_keywords, collect(DISTINCT a.name) AS matched_actors
                
                // Get all genres/directors for display
                OPTIONAL MATCH (m)-[:HAS_GENRE]->(all_g:Genre)
                OPTIONAL MATCH (m)-[:DIRECTED_BY]->(all_d:Person)
                
                WITH m, semantic_score, 
                     matched_directors, matched_genres, matched_keywords, matched_actors,
                     collect(DISTINCT all_g.name) AS all_genres,
                     collect(DISTINCT all_d.name) AS all_directors
                
                // Calculate combined score
                WITH m, semantic_score,
                     matched_directors, matched_genres, matched_keywords, matched_actors,
                     all_genres, all_directors,
                     size(matched_directors) AS dir_matches,
                     size(matched_genres) AS genre_matches,
                     size(matched_keywords) AS kw_matches,
                     size(matched_actors) AS actor_matches
                
                // Combined score: semantic + graph boost
                WITH m, semantic_score,
                     matched_directors, matched_genres, matched_keywords, matched_actors,
                     all_genres, all_directors,
                     dir_matches, genre_matches, kw_matches, actor_matches,
                     (semantic_score + 
                      dir_matches * 0.25 + 
                      genre_matches * 0.08 + 
                      kw_matches * 0.04 + 
                      actor_matches * 0.08) AS combined_score
                
                RETURN m.tmdb_id AS tmdb_id,
                       m.title AS title,
                       m.year AS year,
                       m.overview AS overview,
                       m.vote_average AS rating,
                       semantic_score,
                       combined_score,
                       dir_matches,
                       genre_matches,
                       kw_matches,
                       actor_matches,
                       matched_directors,
                       matched_genres,
                       matched_keywords,
                       matched_actors,
                       all_genres,
                       all_directors
                ORDER BY combined_score DESC
                LIMIT $limit
                """
                
                result = session.run(
                    cypher,
                    embedding=query_embedding,
                    ref_tmdb_id=ref_tmdb_id,
                    all_directors=all_directors,
                    all_genres=all_genres,
                    all_keywords=all_keywords,
                    all_actors=all_actors,
                    search_limit=100,
                    limit=limit
                )
                
                movies = []
                for record in result:
                    movies.append({
                        'tmdb_id': record['tmdb_id'],
                        'title': record['title'],
                        'year': record['year'],
                        'overview': record['overview'],
                        'rating': record['rating'],
                        'semantic_score': record['semantic_score'],
                        'combined_score': record['combined_score'],
                        'matched_directors': record['matched_directors'],
                        'matched_genres': record['matched_genres'],
                        'matched_keywords': record['matched_keywords'],
                        'matched_actors': record['matched_actors'],
                        'all_genres': record['all_genres'],
                        'all_directors': record['all_directors'],
                        'graph_boost': {
                            'directors': record['dir_matches'],
                            'genres': record['genre_matches'],
                            'keywords': record['kw_matches'],
                            'actors': record['actor_matches']
                        }
                    })
                
                logger.info(f"Graph-boosted search found {len(movies)} movies")
                if movies:
                    top = movies[0]
                    logger.info(f"Top result: '{top['title']}' "
                               f"(semantic: {top['semantic_score']:.3f}, "
                               f"combined: {top['combined_score']:.3f}, "
                               f"graph boost: {top['graph_boost']})")
                
                return movies
                
        except Exception as e:
            logger.error(f"Graph-boosted search failed: {e}")
            return self._fallback_semantic_search(mood, limit)
    
    def _fallback_semantic_search(self, mood: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Fallback: Simple semantic search"""
        try:
            query_embedding = self.embedder.get_embedding(mood)
            if not query_embedding:
                return []
            
            with self.driver.session() as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('movie_embedding', $limit, $embedding)
                    YIELD node AS m, score
                    OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre)
                    OPTIONAL MATCH (m)-[:DIRECTED_BY]->(d:Person)
                    RETURN m.tmdb_id AS tmdb_id, m.title AS title,
                           m.year AS year, m.overview AS overview,
                           m.vote_average AS rating, score AS semantic_score,
                           collect(DISTINCT g.name) AS all_genres,
                           collect(DISTINCT d.name) AS all_directors
                    ORDER BY score DESC
                """, embedding=query_embedding, limit=limit)
                
                movies = [dict(r) for r in result]
                logger.info(f"Fallback search found {len(movies)} movies")
                return movies
                
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def _format_results(self, movies: List[Dict[str, Any]], 
                        reference_movie: Optional[Dict[str, Any]] = None) -> str:
        """Format results for LLM with graph boost info"""
        lines = []
        
        if reference_movie:
            lines.append(f"Reference Movie: {reference_movie['title']}")
            lines.append(f"  Directors: {', '.join(reference_movie['directors'][:3])}")
            lines.append(f"  Genres: {', '.join(reference_movie['genres'])}")
            lines.append(f"  Keywords: {', '.join(reference_movie['keywords'][:5])}")
            lines.append("")
        
        lines.append("Search Results:")
        for i, m in enumerate(movies[:15], 1):
            title = m.get('title', 'Unknown')
            year = m.get('year', 'N/A')
            rating = m.get('rating', 'N/A')
            overview = (m.get('overview', '')[:150] + "...") if m.get('overview') else ''
            
            # Show what matched
            matches = []
            if m.get('matched_directors'):
                matches.append(f"Same director: {', '.join(m['matched_directors'])}")
            if m.get('matched_genres'):
                matches.append(f"Genres: {', '.join(m['matched_genres'])}")
            if m.get('matched_actors'):
                matches.append(f"Shared cast: {', '.join(m['matched_actors'][:2])}")
            if m.get('matched_keywords'):
                matches.append(f"Keywords: {', '.join(m['matched_keywords'][:3])}")
            
            lines.append(f"{i}. {title} ({year}) - Rating: {rating}")
            if matches:
                lines.append(f"   Matches: {' | '.join(matches)}")
            if m.get('all_directors'):
                lines.append(f"   Director: {', '.join(m['all_directors'])}")
            if overview:
                lines.append(f"   {overview}")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, query: str, movies: List[Dict[str, Any]],
                                   reference_movie: Optional[Dict[str, Any]] = None) -> str:
        """LLM generates final top 5"""
        if not movies:
            return "Sorry, I couldn't find any movies matching your query."
        
        results_text = self._format_results(movies, reference_movie)
        prompt = RECOMMENDATION_PROMPT.format(query=query, results=results_text)
        
        response = self._call_llm(prompt, temperature=0.3)
        return response if response else "Unable to generate recommendations."
    
    def recommend(self, query: str) -> str:
        """Main entry point"""
        logger.info(f"Processing query: {query}")
        
        # Step 1: Decompose query
        components = self._decompose_query(query)
        
        # Step 2: Get reference movie connections (if mentioned)
        reference_movie = None
        if components.get('reference_movie'):
            reference_movie = self._get_reference_movie_connections(
                components['reference_movie']
            )
        
        # Step 3: Graph-boosted hybrid search
        movies = self._graph_boosted_search(
            mood=components.get('mood', query),
            reference_movie=reference_movie,
            genres=components.get('genres', []),
            directors=components.get('directors', []),
            actors=components.get('actors', []),
            keywords=components.get('keywords', []),
            limit=15
        )
        
        # Step 4: LLM generates recommendations
        recommendations = self._generate_recommendations(query, movies, reference_movie)
        
        return recommendations
    
    def close(self):
        """Close connections"""
        if self.driver:
            self.driver.close()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Movie Recommender')
    parser.add_argument('query', nargs='?', help='Movie query')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    recommender = MovieRecommender()
    
    if args.interactive:
        print("\n🎬 Movie Recommender (type 'quit' to exit)\n")
        while True:
            query = input("What are you looking for?\n> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                print("\n" + "="*50)
                result = recommender.recommend(query)
                print(result)
                print("="*50 + "\n")
    elif args.query:
        result = recommender.recommend(args.query)
        print(result)
    else:
        # Demo
        print("\n🎬 Movie Recommender Demo\n")
        for query in ["sci-fi movies like Interstellar", 
                      "thriller directed by Christopher Nolan"]:
            print(f"Query: {query}")
            print("-" * 40)
            result = recommender.recommend(query)
            print(result)
            print("\n" + "="*50 + "\n")
    
    recommender.close()