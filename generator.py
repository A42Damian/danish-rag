from transformers import AutoTokenizer, AutoModelForCausalLM


class Generator:
    def __init__(self,
                 model_id,
                 ) -> None:
        
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained("danish-foundation-models/DFM-Mimir", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def generate(self, question, retrieved):
        # Build context from retrieved chunks
        retrieved_context = ""

        for i, result in enumerate(retrieved, start=1):
            retrieved_context += (
                f"Source {i}:\n"
                f"{result['content']}\n\n"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using the provided context. "
                    "If the context does not contain enough information to answer, "
                    "say that you do not have enough information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{retrieved_context}\n\n"
                    f"Question:\n{question}"
                )
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200
        )

        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return answer