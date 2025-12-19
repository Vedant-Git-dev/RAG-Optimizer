import pickle
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

def faiss_retrieval(k, question, embedding):
    vectorstore = FAISS.load_local("retrievers/vectorstore", embedding, allow_dangerous_deserialization = True)
    retriever = vectorstore.as_retriever(
        search_kwargs = {"k" : k}
    )

    return retriever.invoke(question)

def bm25_retrieval(k, question):
    with open("retrievers/bm25/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open("retrievers/bm25/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    question = question.lower().split()
    scores = bm25.get_scores(question)

    top_k = sorted(
        range(len(scores)),
        key = lambda i: scores[i],
        reverse = True
    )[:k]

    return [chunks[i] for i in top_k]

def normalize_doc(doc, source = "bm25"):
    normalized = []
    
    for d in doc:
        if isinstance(d, Document):
            normalized.append(d)

        else :
            normalized.append(
                Document(
                    page_content = d,
                    metadata = {"source" : source}
                )
            )

    return normalized

def merge_and_dedupe(doc1, doc2):
    seen = set()
    merged = []

    for d in doc1 + doc2:
        text = d.page_content.strip().lower()

        if text not in seen:
            merged.append(d)
            seen.add(text)

    return merged

def hybrid_retrieval(k, question, embedding):
    faiss_doc = faiss_retrieval(k = k, question = question, embedding = embedding)
    bm25_doc = bm25_retrieval(k = k, question = question)

    faiss_doc = normalize_doc(faiss_doc, "faiss")
    bm25_doc = normalize_doc(bm25_doc, "bm25")

    final_docs = merge_and_dedupe(faiss_doc, bm25_doc)

    return final_docs

# Main function
def run_rag_pipeline(retriever, k, answer_model, reranker, question, need_rerank, embedding):
    
    retriever_k = max(10, k * 2) if reranker else k

    if retriever == "faiss": 
        docs = faiss_retrieval(k = retriever_k, question = question, embedding = embedding)
    
    elif retriever == "bm25":
        docs = bm25_retrieval(k = retriever_k, question = question)

    else:
        docs = hybrid_retrieval(k = retriever_k, question = question, embedding = embedding)

    if need_rerank:

        pairs = [(question, chunk.page_content) for chunk in docs]
        scores = reranker.predict(pairs)

        
        top_k = sorted(
            zip(scores, docs),
            key = lambda x: x[0],
            reverse = True
        )[:k]

        docs = [doc for _, doc in top_k]

    retrieved_context = "\n\n".join(d.page_content for d in docs)

    try:
        ans = answer_model(question = question, context = retrieved_context)["answer"]

    except Exception as e:

        prompt = f"""
            You are an assistant answering questions using retrieved document context.

            INSTRUCTIONS:
            - Use ONLY the information provided in the context.
            - Do NOT use external knowledge.
            - Do NOT copy sentences verbatim from the context.
            - If information is repeated across chunks, combine it into a single coherent answer.
            - If the answer is not present in the context, say: "The information is not available in the provided context."

            ANSWER STYLE:
            - Be concise but complete.
            - Use full sentences.
            - If the question asks for a list, return a clean, comma-separated list.
            - If the question asks for an explanation, write a short structured paragraph.
            - Do not mention section names, chunk numbers, or document metadata.

            CONTEXT:
            {retrieved_context}

            QUESTION:
            {question}

            ANSWER:
            """
        
        ans = answer_model.invoke(prompt)
    
    return ans
    
    