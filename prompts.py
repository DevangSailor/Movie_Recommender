"""
Prompts for Movie Recommender System
Using Qwen3 8B via Ollama
"""

# Query Decomposition Prompt
DECOMPOSITION_PROMPT = """/no_think
Extract movie search components from this query. Return JSON only, no explanation.

Available Genres: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western

Examples:

Query: "sci-fi movies directed by Nolan"
{{"reference_movie": null, "keywords": [], "genres": ["Science Fiction"], "directors": ["Christopher Nolan"], "actors": [], "mood": "sci-fi"}}

Query: "movies like Inception"
{{"reference_movie": "Inception", "keywords": [], "genres": [], "directors": [], "actors": [], "mood": "mind-bending thriller"}}

Query: "funny romantic movies with Tom Hanks"
{{"reference_movie": null, "keywords": [], "genres": ["Comedy", "Romance"], "directors": [], "actors": ["Tom Hanks"], "mood": "funny romantic"}}

Query: "dark thriller about revenge like Oldboy"
{{"reference_movie": "Oldboy", "keywords": ["revenge", "dark"], "genres": ["Thriller"], "directors": [], "actors": [], "mood": "dark thriller revenge"}}

Query: "sci-fi movies like Interstellar"
{{"reference_movie": "Interstellar", "keywords": [], "genres": ["Science Fiction"], "directors": [], "actors": [], "mood": "sci-fi space epic"}}

Query: "adventure movies similar to Indiana Jones"
{{"reference_movie": "Indiana Jones", "keywords": ["adventure", "treasure"], "genres": ["Adventure", "Action"], "directors": [], "actors": [], "mood": "adventure treasure hunting"}}

Query: "horror movies like The Conjuring with ghosts"
{{"reference_movie": "The Conjuring", "keywords": ["ghosts", "haunted"], "genres": ["Horror"], "directors": [], "actors": [], "mood": "supernatural horror"}}

Query: "{query}"
"""


# Cypher Generation Prompt
CYPHER_PROMPT = """/no_think
Generate a Cypher query for Neo4j based on the filters. Return only the Cypher query, nothing else.

Schema:
- (:Movie {{tmdb_id, title, year, overview, vote_average}})
- (:Person {{name}})
- (:Genre {{name}})
- (:Keyword {{name}})
- (:Movie)-[:DIRECTED_BY]->(:Person)
- (:Movie)-[:HAS_CAST_MEMBER]->(:Person)
- (:Movie)-[:HAS_GENRE]->(:Genre)
- (:Movie)-[:HAS_KEYWORD]->(:Keyword)

Rules:
- Always return: m.tmdb_id, m.title, m.year, m.overview, m.vote_average
- Use OPTIONAL MATCH for flexibility
- LIMIT 15
- If no filters provided, return empty string

Examples:

Filters: {{"genres": ["Science Fiction"], "directors": ["Christopher Nolan"], "actors": []}}
MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {{name: "Science Fiction"}}), (m)-[:DIRECTED_BY]->(d:Person {{name: "Christopher Nolan"}}) RETURN m.tmdb_id, m.title, m.year, m.overview, m.vote_average ORDER BY m.vote_average DESC LIMIT 15

Filters: {{"genres": ["Action"], "directors": [], "actors": ["Tom Cruise"]}}
MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {{name: "Action"}}), (m)-[:HAS_CAST_MEMBER]->(a:Person {{name: "Tom Cruise"}}) RETURN m.tmdb_id, m.title, m.year, m.overview, m.vote_average ORDER BY m.vote_average DESC LIMIT 15

Filters: {{"genres": ["Comedy", "Romance"], "directors": [], "actors": []}}
MATCH (m:Movie)-[:HAS_GENRE]->(g1:Genre {{name: "Comedy"}}), (m)-[:HAS_GENRE]->(g2:Genre {{name: "Romance"}}) RETURN m.tmdb_id, m.title, m.year, m.overview, m.vote_average ORDER BY m.vote_average DESC LIMIT 15

Filters: {{"genres": [], "directors": ["Steven Spielberg"], "actors": []}}
MATCH (m:Movie)-[:DIRECTED_BY]->(d:Person {{name: "Steven Spielberg"}}) RETURN m.tmdb_id, m.title, m.year, m.overview, m.vote_average ORDER BY m.vote_average DESC LIMIT 15

Filters: {{"genres": [], "directors": [], "actors": []}}


Filters: {filters}
"""


# Final Recommendation Prompt
RECOMMENDATION_PROMPT = """/no_think
You are a movie recommender. Based on the user's query and search results, recommend the top 5 best matching movies.

User Query: "{query}"

Search Results:
{results}

Instructions:
- Pick the 5 movies that best match what the user is looking for
- Give a brief reason (1 sentence) for each recommendation
- Consider the mood, themes, and preferences expressed in the query
- Format as a numbered list

Recommendations:
"""


# Keyword Matching Prompt (optional, if needed for better keyword extraction)
KEYWORD_EXTRACTION_PROMPT = """/no_think
Extract relevant movie keywords from this query. Return as JSON array only.

Think about: themes, settings, plot elements, emotions, visual style.

Examples:

Query: "dark psychological thriller about revenge"
["revenge", "psychological", "dark", "suspense"]

Query: "feel-good adventure in space"
["space", "adventure", "feel-good", "epic"]

Query: "romantic comedy in New York"
["romantic", "love", "new york", "comedy"]

Query: "{query}"
"""