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


def main():
    print("Starting Retriever testing")
    test_dir = "test"
    print(f"Loading index from {test_dir}")
    kb_index = Index.from_path(test_dir)

    top_k = 3
    test_question = "Hvordan beskrives AI til offentlige myndigheder?"
    print(f"Creating Retriever. Top k = {top_k}")

    retriever = Retriever(top_k=top_k, kb_index=kb_index)

    print(f"Finding context for question: {test_question}")

    found_context = retriever.retrieve(test_question)

    print(f"Found context: {found_context}")

if __name__ == "__main__":
    main()