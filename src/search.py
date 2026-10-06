import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="openai/gpt-oss-20b"
    ):
        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()

        else:
            from data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file"
            )

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=llm_model
        )

        print("[INFO] Groq LLM initialized")


    def search_and_summarize(
        self,
        query,
        top_k=3
    ):

        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        context = "\n\n".join(
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        )

        if not context:
            return "No relevant documents found."

        prompt = f"""
Answer the query using only the context below.

Query:
{query}

Context:
{context}

Give a simple and clear answer.
"""

        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":

    rag = RAGSearch()

    query = "What is NovaTech Innovations?"

    answer = rag.search_and_summarize(
        query,
        top_k=3
    )

    print("\nAnswer:")
    print(answer)


