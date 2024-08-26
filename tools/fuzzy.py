import streamlit as st
from llm import llm
from graph import graph

# Create the Cypher QA chain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

FUZZY_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Fine Tuning:

For movie titles that begin with "The", move "the" to the end. For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".

Example Cypher Statements:

1. To find a person with partial name:
```
MATCH (p:Person)-[rel]->(m:Movie)
WHERE toLower(p.name) CONTAINS toLower("Partial Name")
RETURN p.name AS person_name, type(rel) AS related, collect(m.title) AS movie_title
ORDER BY size(movie_title) DESC
LIMIT 5
```

2. To find a movie with partial title:
```
MATCH (p:Person)-[rel]->(m:Movie)
WHERE toLower(m.title) CONTAINS toLower("Partial Title")
RETURN m.title AS movie_title, collect(p.name) AS cast
ORDER BY size(cast) DESC
LIMIT 10
```

Schema:
{schema}

Question:
{question}
"""

fuzzy_prompt = PromptTemplate.from_template(FUZZY_GENERATION_TEMPLATE)

fuzzy_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=fuzzy_prompt
)