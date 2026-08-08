from knowledge_base import Index

class Retriever:
    def __init__(self,
                top_k:int,
                kb_index:Index
                ) -> None:

        self.top_k = top_k
        self.kb_index = kb_index

    def retrieve(self, question:str):
        found = self.kb_index.search_index(question)
        additional_context = ""
        for find in found:
            try:
                content = find['content']
                print(f"[Debug] Retrieved content: {content}")
                additional_context = f"{additional_context} + {content} + \n"
            except:
                print(f"[Error] Problem with retrieved content... Skipping")
                continue

        return additional_context