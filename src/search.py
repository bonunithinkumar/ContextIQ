import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

def find_txt_line_number(file_path: str, chunk_text: str) -> int:
    """Find the line number of a text chunk in a .txt file."""
    if not os.path.exists(file_path):
        return None
    try:
        # Get first non-empty line of the chunk
        lines = [line.strip() for line in chunk_text.split('\n') if line.strip()]
        if not lines:
            return None
        target = lines[0]
        # Avoid matching extremely short/common strings
        if len(target) < 10 and len(lines) > 1:
            target = target + " " + lines[1]
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f, 1):
                if target in line:
                    return idx
    except Exception:
        pass
    return None

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemma2-9b-it"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key)  # Update from "llama3-8b-8192" to the working model


    def search_and_summarize(self, query: str, top_k: int = 5) -> dict:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = []
        citations = []
        seen_citations = set()
        
        for r in results:
            meta = r.get("metadata")
            if not meta:
                continue
                
            text = meta.get("text", "")
            texts.append(text)
            
            source = meta.get("source", "")
            doc_name = os.path.basename(source) if source else "Unknown Document"
            
            citation_parts = [f"Doc: {doc_name}"]
            
            if "page" in meta:
                citation_parts.append(f"Page: {meta['page'] + 1}")
                
            if source and source.endswith('.txt'):
                line_no = find_txt_line_number(source, text)
                if line_no:
                    citation_parts.append(f"Line: {line_no}")
            elif "row" in meta:
                citation_parts.append(f"Row: {meta['row']}")
            elif "seq_num" in meta:
                citation_parts.append(f"Sequence: {meta['seq_num']}")
                
            citation_str = ", ".join(citation_parts)
            if citation_str not in seen_citations:
                seen_citations.add(citation_str)
                citations.append(citation_str)
                
        context = "\n\n".join(texts)
        if not context:
            return {
                "summary": "No relevant documents found.",
                "citations": []
            }
            
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\n\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return {
            "summary": response.content,
            "citations": citations
        }

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    result = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", result["summary"])
    print("Citations:", result["citations"])