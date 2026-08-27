from sentence_transformers import SentenceTransformer


class Generator:
    def __init__(self,
                 model_id,
                 ) -> None:
        
        self.model_id = model_id
        self.model = SentenceTransformer(
                    self.model_id
                )

    def generate(self, question, retrieved):

        retrieved_context = ""
        #TODO! Later collapse finds into one if they share source. (Sort based on index?)
        for i, find in enumerate(retrieved, start=1):
            retrieved_context += f"""
                                Source {i} \n
                                {find["content"]} \n \n
                                """

        prompt = f"""
                SYSTEM:
                Answer the question using the provided context.
                \n
                CONTEXT:
                {retrieved_context}
                \n
                QUESTION:
                {question}
                """

        answer = self.model(prompt)
        return answer