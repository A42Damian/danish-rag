from knowledge_base import Index

class Retriever:
    def __init__(self,
                top_k:int,
                kb_index:Index
                ) -> None:

        self.top_k = top_k
        self.kb_index = kb_index

    def retrieve(self, question:str):
        found = self.kb_index.search_index(question, self.top_k)
        additional_context = found

        return additional_context