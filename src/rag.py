from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from functools import lru_cache
from src.config import RAW_DATA_PATH,CHROMA_DB_PATH,COLLECTION_NAME,CHUNK_SIZE,CHUNK_OVERLAP,EMBEDDING_MODEL,TOP_K

## loading embedding model if needed
@lru_cache(maxsize=1)
def get_embed_model():
    return SentenceTransformer(EMBEDDING_MODEL)

#### Vector Data Base
vector_dt=chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection=vector_dt.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

if collection.count() > 0:

    print("Collection already has data, skipping ingestion.")  #token saver
else:

    ## loading embedding model
    embed_model = get_embed_model()

    #### PDF loading 
    loader=PyPDFDirectoryLoader(RAW_DATA_PATH)
    raw_data=loader.load()

    #### Splitting

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    chunks=text_splitter.split_documents(raw_data)
    if not chunks:

        print("......No chunks. Check PDFs....")
    else:
        
        documents=[]
        metadata=[]
        ids=[]
        

        for i,chunk in enumerate(chunks):
            documents.append(chunk.page_content)
            metadata.append(chunk.metadata)
            src = chunk.metadata["source"].split("/")[-1].replace(".pdf", "")
            ids.append(f"{src}_{i}")

        embeddings = embed_model.encode(documents, normalize_embeddings=True)

        collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadata,
        ids=ids,
        )

        #### Checks
        print(f"Loaded {len(raw_data)} pages")
        print(f"Created {len(chunks)} chunks")
        print(f"First chunk preview: {chunks[0].page_content[:200]}")
        print(f"First chunk metadata: {chunks[0].metadata}")


### Retrieval part 
def retrieval(query: str, k: int=TOP_K):

    embed_model=get_embed_model()
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    results=collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
    )
    return (
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    )

### debugging
if __name__ == "__main__":
    docs, dists, metas = retrieval("What is the ML project about?", k=3)
    for i, (d, s, m) in enumerate(zip(docs, dists, metas)):
        print(f"\n--- {i+1} (distance={s:.4f}) ---")
        print(d[:300])
        print(f"source: {m['source']}, page: {m['page']}")