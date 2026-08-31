from vector_store import load_vector_store


def retrieve(query, store_path="vector_store", k=3):
    vector_store, _ = load_vector_store(store_path)
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def format_results(results):
    output = []
    for i, (doc, score) in enumerate(results):
        output.append({
            "rank": i + 1,
            "score": round(float(score), 4),
            "content": doc.page_content,
            "metadata": doc.metadata,
        })
    return output


if __name__ == "__main__":
    query = input("Enter your query: ")
    results = retrieve(query, k=3)
    formatted = format_results(results)

    for r in formatted:
        print(f"\n--- Rank {r['rank']} (Score: {r['score']}) ---")
        print(f"Page: {r['metadata'].get('page', 'N/A')}")
        print(r["content"][:300])
