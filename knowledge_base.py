from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import faiss
import json

"""

"""
model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)
tokenizer = AutoTokenizer.from_pretrained(
    "intfloat/multilingual-e5-base"
)

class index:
    def __init__(self, text_size, text_overlap, faiss_index=None, meta=None) -> None:
        self.text_size = text_size
        self.text_overlap = text_overlap

        self.faiss_index = faiss_index

        self.meta:list[dict] = meta or []

    def add_pdf(self, pdf_path):
        text = load_pdf(pdf_path)
        tokens = tokenizer(text)
        chunks = chunk_tokens(tokens, self.text_size, self.text_overlap)

        embed_list = []
        # iterate over chunks and embed
            # Save embed and metadata
        for idx, chunk in enumerate(chunks):
            vec = embed_text(chunk)
            embed_list.append(np.array(vec, dtype=np.float32).reshape(1, -1))

            self.meta.append({
                "source": pdf_path,
                "chunk_id": idx,
                "text": tokenizer.decode(chunk)
            })
        
        xb = np.vstack(embed_list)

        # Creates a faiss if there is not already one
        if self.faiss_index is None:
            d = xb.shape[1]
            self.faiss_index = faiss.IndexFlatL2(d)

        # Adds embeds to faiss
        self.faiss_index.add(xb)

        # Write faiss to storage
        faiss.write_index(self.faiss_index, "faiss_index.faiss")

        # Write metadata to storage
        with open("faiss_meta.json", 'w') as f:
            for item in self.meta:
                f.write(json.dumps(item) + "\n")

    def search_faiss(self,search_text:str):
        # embed text and formate to float32
        q = embed_text(search_text)
        q = np.array(q, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        # if a faiss exists return distances and indeces for the 5 nearest neighbours
        if self.faiss_index is not None:
            dist, index = self.faiss_index.search(q, k=5)
            return dist, index
        else:
            raise RuntimeError("No Faiss index to search in")


def load_pdf(pdf_path:str):
    from pypdf import PdfReader
    #Loads PDF as a string of text
    reader = PdfReader(pdf_path)
    pdf_text:str = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

# TODO: This should be rewritten for token-level chunk use instead of character level.
def gen_split_overlap(seq, size, overlap):
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')

    # Yields a generator with a len and overlap
    for i in range(0, len(seq) - overlap, size - overlap):
        yield seq[i:i + size]

def chunk_text(text:str, size:int, overlap:int):
    chunks:list[str] = []
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')
    
    for i in range(0, len(text) - overlap, size-overlap):
        chunks.append(text[i:i+size])

    return chunks

def chunk_tokens(tokens, size:int, overlap:int):
    chunks:list=[]
    if size > 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')
    
    for i in range(0, len(tokens) - overlap, size-overlap):
        chunks.append(tokens[i:i+size])

    return chunks

def embed_text(text:str):
    embedding = model.encode(text)
    return embedding


def main():
    print("Starting Knowledgebase testing \n")


if __name__ == "__main__":
    main()


