from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

"""

"""
model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

meta:list[dict] = []
faiss_index = None

class index:
    def __init__(self, text_size, text_overlap, faiss_index, meta) -> None:
        self.text_size = text_size
        self.text_overlap = text_overlap

        self.faiss_index = faiss_index

        self.meta:list[dict] = meta or []

    def add_pdf(self, pdf_path):
        text = load_pdf(pdf_path)
        chunks = chunk_text(text, self.text_size, self.text_overlap)
        embed_list = []
        for idx, chunk in enumerate(chunks):
            vec = embed_text(chunk)
            embed_list.append(np.array(vec, dtype=np.float32).reshape(1, -1))

            self.meta.append({
                "id": idx,
                "text": chunk
            })

        xb = np.vstack(embed_list)

        if self.faiss_index is None:
            d = xb.shape[1]
            self.faiss_index = faiss.IndexFlatL2(d)

        self.faiss_index.add(xb)

        faiss.write_index(self.faiss_index, "faiss_index.faiss")

        with open("faiss_meta.jsonl", 'w') as f:
            for item in self.meta:
                f.write(json.dumps(item) + "\n")

    def search_faiss(self,search_text:str):
        q = embed_text(search_text)
        q = np.array(q, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        if self.faiss_index is not None:
            dist, index = self.faiss_index.search(q, k=5)
            return dist, index
        else:
            raise RuntimeError("No Faiss index to search in")


def load_pdf(pdf_path:str):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    pdf_text:str = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

def gen_split_overlap(seq, size, overlap):        
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')

    for i in range(0, len(seq) - overlap, size - overlap):            
        yield seq[i:i + size]

def chunk_text(text:str, size:int, overlap:int):
    chunks:list[str] = []
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')
    
    for i in range(0, len(text) - overlap, size-overlap):
        chunks.append(text[i:i+size])

    return chunks

def embed_text(text:str):
    embedding = model.encode(text)
    return embedding


def main():
    sample_text = "This is a sample text for testing the knowledge base. " * 10
    test_index = index(text_size=50, text_overlap=10, faiss_index=None, meta=[])

    chunks = chunk_text(sample_text, test_index.text_size, test_index.text_overlap)
    print(f"Generated {len(chunks)} chunks.")

    if chunks:
        embedding = embed_text(chunks[0])
        print(f"First chunk embedding length: {len(embedding)}")

        xb = np.vstack([
            np.array(embed_text(chunk), dtype=np.float32).reshape(1, -1)
            for chunk in chunks
        ])

        test_index.faiss_index = faiss.IndexFlatL2(xb.shape[1])
        test_index.faiss_index.add(xb)
        print(f"Added {len(chunks)} vectors to Faiss index.")
        print(f"Index size: {test_index.faiss_index.ntotal}")

        query = "testing the knowledge base"
        distances, indices = test_index.search_faiss(query)
        print(f"Search query: {query}")
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")

        if indices.size and indices[0][0] != -1:
            hit_meta = test_index.meta[indices[0][0]]
            print(f"Top hit text: {hit_meta['text']}")
    else:
        print("No chunks generated.")


if __name__ == "__main__":
    main()


