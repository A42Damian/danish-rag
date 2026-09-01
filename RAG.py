from knowledge_base import Index
from retriever import Retriever
from generator import Generator

class RetrievalAugmentedGenerator:
    def __init__(self,
                 faiss_path:str,
                 model_id,
                 num_sources,
                 ) -> None:
        self.model_id = model_id
        self.num_sources = num_sources

        self.knowledge_base = Index.from_path(faiss_path)
        self.retriever = Retriever(top_k=self.num_sources, kb_index=self.knowledge_base)
        self.generator = Generator(model_id=self.model_id)

    def answer_q(self, question):
        retrieved = self.retriever.retrieve(question)
        answer = self.generator.generate(question, retrieved)

        sources = ""
        for i, result in enumerate(retrieved, start=1):
            sources += (
                f"Source {i}:\n"
                f"{result['content']}\n\n"
            )

        return answer, sources


def main():
    print(f"Starting test of Retrieval Arugmented Generator")
    kb_dir = "test"
    model_id = "danish-foundation-models/DFM-Mimir"
    num_sources = 3
    rag = RetrievalAugmentedGenerator(faiss_path=kb_dir, model_id=model_id, num_sources=num_sources)
    question = "Hvordan beskrives AI til offentlige myndigheder?"
    print(f"Asking question: {question}")
    answer, sources = rag.answer_q(question)
    print(f"Answer: \n {answer}")
    print(f"Sources: \n {sources}")

if __name__ == "__main__":
    main()