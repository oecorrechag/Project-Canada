# 1. Library imports
from pinecone import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from secretitos import pincone_api
from typing import List, Dict, Any

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

index_name = "movies"
pc = Pinecone(api_key=pincone_api)
index_2 = pc.Index(index_name, dimension=384)

def get_shared_keywords(query: str, document: str, top_n: int = 5) -> List[str]:
    """
    Encuentra las palabras clave más relevantes compartidas entre una consulta y un documento.
    
    Args:
        query: Texto de la consulta del usuario
        document: Texto del documento a comparar
        top_n: Número máximo de palabras clave a devolver
        
    Returns:
        Lista de las top_n palabras clave compartidas, ordenadas por relevancia
    """
    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query, document])
    
    # Obtener pesos y palabras
    query_weights, doc_weights = tfidf_matrix.toarray()
    vocabulary = vectorizer.get_feature_names_out()
    
    # Seleccionar palabras relevantes (presentes en ambos textos)
    relevant_words = [
        (word, query_weight + doc_weight)
        for word, query_weight, doc_weight in zip(vocabulary, query_weights, doc_weights)
        if query_weight > 0 and doc_weight > 0
    ]
    
    # Ordenar y seleccionar las mejores
    relevant_words.sort(key=lambda x: x[1], reverse=True)
    return [word.capitalize() for word, _ in relevant_words[:top_n]]

def format_search_result(result: Dict[str, Any], keywords: List[str], number_of_desition: int) -> str:
    """
    Formatea los resultados de búsqueda en un string legible.

    Args:
        result: Diccionario con los resultados de la búsqueda
        keywords: Lista de palabras clave relevantes

    Returns:
        String formateado con la información de la película
    """
    metadata = result['matches'][number_of_desition]['metadata']
    return (
        f"Título: {metadata['Series_Title']}\n"
        f"Resumen: {metadata['Overview']}\n"
        f"Palabras clave: {', '.join(keywords)}"
    )


def search_movies(query: str, n_results: int = 2) -> str:
    """
    Busca películas relevantes basadas en una consulta y devuelve el mejor resultado.

    Args:
        query: Consulta de búsqueda del usuario
        n_results: Número de resultados a considerar

    Returns:
        String formateado con la información de la película más relevante
    """
    # Realizar búsqueda semántica
    query_vector = model.encode(query).tolist()
    results = index_2.query(vector=query_vector, top_k=n_results, include_metadata=True)

    # Obtener palabras clave para el primer resultado
    first_result_text = results['matches'][0]['metadata']['text']
    keywords = get_shared_keywords(query, first_result_text)
    number_of_desition = 0

    # Si no hay palabras clave compartidas, intentar con el segundo resultado
    if not keywords and n_results > 1:
        second_result_text = results['matches'][1]['metadata']['text']
        keywords = get_shared_keywords(query, second_result_text)
        number_of_desition = 1

    return format_search_result(results, keywords, number_of_desition)
