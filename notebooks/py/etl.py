import re
import unicodedata
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from secret import pincone_api
from mypackage import dir

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index_name = "movies"
batch_size = 64 # Configuración del tamaño de lote a subir a la base de datos

# Environment variables
project = 'canada'
data = dir.make_dir(project) 
raw = data('raw')
pc = Pinecone(api_key=pincone_api)

# Función para cargar datos
def cargar_datos(table_name: str) -> pd.DataFrame:
    df = pd.read_csv(raw / f'{table_name}.csv', sep = ',', decimal = '.', header = 0, encoding = 'latin1')
    print(f'Loaded table: {table_name}')
    return df

def quitar_tildes(texto):
    return ''.join(c for c in unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def limpiar_texto_para_embeddings(dataframe,
                                  columna: str,
                                  mantener_numeros: bool = False,
                                  quitar_acentos: bool = True,
                                  quitar_stopwords: bool = False) -> str:
    """
    Limpia y normaliza texto para prepararlo para la creación de embeddings.

    Args:
        texto: Cadena de texto a limpiar
        mantener_numeros: Conservar números en el texto (por defecto False)
        quitar_acentos: Convertir caracteres acentuados a su versión sin acento
        quitar_stopwords: Eliminar palabras comunes (requiere nltk si True)

    Returns:
        Texto limpio y normalizado
    """
    # Normalización básica
    dataframe[columna] = dataframe[columna].str.lower()

    # Eliminar URLs
    dataframe[columna] = dataframe[columna].apply(lambda t: re.sub(r'https?://\S+|www\.\S+', '', str(t)) if pd.notna(t) else t)

    # Eliminar menciones y hashtags
    dataframe[columna] = dataframe[columna].apply(lambda t: re.sub(r'[@#]\w+', '', str(t)) if pd.notna(t) else t)

    # Eliminar caracteres especiales (conservando números si se indica)
    dataframe[columna] = dataframe[columna].apply(lambda t: re.sub(r'[^\w\s]' if mantener_numeros else r'[^a-záéíóúñ\s]', '', str(t)) if pd.notna(t) else t)

    # Quitar acentos si se especifica
    if quitar_acentos:
        dataframe[columna] = dataframe[columna].apply(quitar_tildes)

    # Eliminar espacios adicionales
    dataframe[columna] = dataframe[columna].str.strip()

    # Opcional: eliminar stopwords (palabras comunes)
    if quitar_stopwords:
        dataframe[columna] = dataframe[columna].apply(remove_stopwords)

    return dataframe

# Función para transformar los documentos
def transformar_documentos(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_combine = ['Overview', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']
    df['text'] = df[columns_to_combine].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    # Crear el id
    df['ids'] = df.index.astype('str')
    df = limpiar_texto_para_embeddings(df, columna='text', quitar_stopwords=True)
    df = df.fillna(' ')

    # Crea los embeddings de la columna text
    embeddings = model.encode(df['text'], batch_size=64, show_progress_bar=True)
    df['embeddings'] = embeddings.tolist()
    return df

# Función para cargar los documentos en la base de datos Pinecone
def cargar_en_db(df: pd.DataFrame, index_name: str, 
                 dimension_embeddings: int, batch_size: int) -> None:

    existing_indexes = pc.list_indexes()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension_embeddings, 
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    else:
        print(f"Index {index_name} already exists")

    index = pc.Index(index_name)
    ids_array = df['ids'].values
    embeddings_array = np.array(df['embeddings'].tolist())  # Asume que embeddings son listas
    metadata_records = df.drop(['ids', 'embeddings'], axis=1).to_dict('records')


    # Procesamiento por lotes con manejo de errores
    with tqdm(total=len(df), desc="Processing batches") as pbar:
        for i in range(0, len(df), batch_size):
            try:
                i_end = i + batch_size
                batch_ids = ids_array[i:i_end]
                batch_emb = embeddings_array[i:i_end]
                batch_metadata = metadata_records[i:i_end]
                to_upsert = list(zip(batch_ids, batch_emb, batch_metadata))
                index.upsert(vectors=to_upsert)
                pbar.update(len(batch_ids))

            except Exception as e:
                print(f"Error en lote {i}-{i_end}: {str(e)}")

    # Verificar estadísticas del índice
    stats = index.describe_index_stats()
    print(f"Estadísticas del índice: {stats}")

if __name__ == '__main__':
    # ETL Documentos
    df = cargar_datos('imdb_top_1000_fixed')
    df = transformar_documentos(df)

    dimension_embeddings = len(df['embeddings'][0])
    cargar_en_db(df=df, index_name=index_name, 
                 dimension_embeddings=dimension_embeddings, batch_size=batch_size)
