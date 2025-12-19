from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter

def faiss(chunks):
    
    embedder = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    vectorstore = FAISS.from_texts(chunks, embedder)
    vectorstore.save_local("retrievers/vectorstore")

def bm25(chunks):
    tokens = [chunk.lower().split() for chunk in chunks]

    bm25 = BM25Okapi(tokens)

    with open("retrievers/bm25/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open("retrievers/bm25/bm25.pkl", "wb") as f:
        pickle.dump(chunks, f)

def ingest_data(context, chunk_size, chunk_overlap, retriever):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = splitter.split_text(context)

    if retriever == "faiss":
        faiss(chunks)        

    if retriever == "bm25":
        bm25(chunks)

    if retriever == "hybrid":
        faiss(chunks)
        bm25(chunks)