from huggingface_hub import InferenceClient
from src.config import HF_TOKEN,LLM_MODEL,TOP_K
from src.rag import retrieval


client = InferenceClient(model=LLM_MODEL,token=HF_TOKEN)

def answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. You answer questions about data science projects.
You only answer based on the context provided. Do not use your internal knowledge.
If the context does not contain the answer, say: "I don't know."
Context:
{context}

Question: {question}
Answer:"""
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    q = input("Question: ")
    docs, _, _ = retrieval(q, k=TOP_K)
    print(answer(q, docs))