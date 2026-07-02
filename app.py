import os
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents
from src.search import RAGSearch

# ── Build or load the FAISS index ──────────────────────────────────────────
store = FaissVectorStore("faiss_store")
faiss_path = "faiss_store/faiss.index"

if not os.path.exists(faiss_path):
    print("[INFO] No index found — building from documents...")
    docs = load_all_documents("data")
    store.build_from_documents(docs)
else:
    store.load()

# ── Interactive query loop ──────────────────────────────────────────────────
if __name__ == "__main__":
    rag_search = RAGSearch()

    print("\n" + "=" * 55)
    print("  🤖  RAG Document Q&A  —  type 'exit' to quit")
    print("=" * 55)

    while True:
        query = input("\n❓ Your question: ").strip()

        if not query:
            print("  ⚠️  Please enter a question.")
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("\n👋 Goodbye!\n")
            break

        print("\n⏳ Searching your documents...\n")
        result = rag_search.search_and_summarize(query, top_k=3)
        print("─" * 55)
        print("✅ Answer:\n")
        print(result["summary"])
        print("\n📚 Sources / Citations:")
        if result["citations"]:
            for idx, citation in enumerate(result["citations"], 1):
                print(f"  [{idx}] {citation}")
        else:
            print("  No citations available (please rebuild your FAISS index to capture metadata).")
        print("─" * 55)
