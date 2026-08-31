from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from retrieval import retrieve, format_results


def load_llm(model_name="google/flan-t5-small", max_new_tokens=256):
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        max_new_tokens=max_new_tokens,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def generate_answer(query, llm, store_path="vector_store", k=3):
    results = retrieve(query, store_path=store_path, k=k)
    context = "\n\n".join([doc.page_content for doc, _ in results])

    prompt = f"""Answer the following question based on the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    answer = llm.invoke(prompt)
    return answer, format_results(results)


if __name__ == "__main__":
    print("Loading LLM...")
    llm = load_llm()

    while True:
        query = input("\nEnter your query (or 'quit'): ")
        if query.lower() == "quit":
            break

        answer, sources = generate_answer(query, llm)
        print(f"\nAnswer: {answer}")
        print("\nSources:")
        for s in sources:
            print(f"  Page {s['metadata'].get('page', 'N/A')} (Score: {s['score']})")
