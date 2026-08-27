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
        return answer