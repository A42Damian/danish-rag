from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import faiss
import json
import os

from document_loaders import pdf_loaders

"""
Knowledge base based on faiss index for a simple RAG system.
"""

class Index:
    def __init__(self,
                 text_size:int,
                 text_overlap:int,
                 model_id:str,
                 ) -> None:
        
        
        self.text_size = text_size
        self.text_overlap = text_overlap

        self.model_id = model_id
        self.model = SentenceTransformer(
            self.model_id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )

        self.faiss_index = None

        self.meta:dict = {"text_size": self.text_size,
                          "text_overlap": self.text_overlap,
                          "model_id": self.model_id,
                          "chunks": []}


    def save(self, path:str):

        os.makedirs(path, exist_ok=True)

        # Write faiss to storage
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, os.path.join(path, "faiss_index.faiss"))

        # Write metadata to storage
        with open(os.path.join(path,"faiss_meta.json"), 'w') as f:
            f.write(json.dumps(self.meta))

    @classmethod
    def from_path(cls, path:str):
        # Load meta info
        with open(os.path.join(path,"faiss_meta.json"), 'r') as f:
            meta = json.load(f)

        # Create class object
        cls = Index(meta["text_size"], meta["text_overlap"], meta["model_id"])

        # Set object meta dictionary
        cls.meta = meta

        # Load faiss index from storage
        cls.faiss_index = faiss.read_index(os.path.join(path, "faiss_index.faiss"))
        
        return cls

    def embed_text(self, text:str):
        embedding = self.model.encode(text)
        return embedding

    def add_pdf(self, pdf_path):
        print(f"Adding PDF from {pdf_path}")
        text = load_pdf(pdf_path)
        tokens = self.tokenizer(text)["input_ids"]
        embed_list = []
        # iterate over chunks and embed
            # Save embed and metadata
        for idx, chunk in enumerate(gen_split_overlap(tokens, self.text_size, self.text_overlap)):
            chunk_text = self.tokenizer.decode(chunk)
            
            vec = self.embed_text(chunk_text)
            vec = np.array(vec, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(vec)
            embed_list.append(vec)

            self.meta["chunks"].append({
                "source": pdf_path,
                "chunk_id": idx,
                "text": chunk_text
            })
        
        xb = np.vstack(embed_list)

        # Creates a faiss if there is not already one
        if self.faiss_index is None:
            d = xb.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)

        # Adds embeds to faiss
        self.faiss_index.add(xb)

    def add_directory(self, dir:str, recursive:bool=False):
        
        print(f"Adding directory:{dir} | Recursive:{recursive}")
        directory = os.fsencode(dir)

        for file in os.listdir(directory):
            print(f"Adding file: {file}")
            file = os.fsdecode(file)
            path = os.path.join(dir, file)
            if recursive and os.path.isdir(path):
                self.add_directory(path, recursive=True)
            else:
                if file.endswith(".pdf"):
                    self.add_pdf(os.path.join(dir, file))
                else: 
                    print(f"Unkown File Format... Skipping {file}")
                    continue
            print(f"Finished adding directory: {directory}")

    def search_index(self, search_text:str, top_k:int):
        # embed text and formate to float32
        q = self.embed_text(search_text)
        q = np.array(q, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        # if a faiss exists return distances and indeces for the 5 nearest neighbours
        if self.faiss_index is not None:
            found_list = []
            dists, indices = self.faiss_index.search(q, k=top_k)
            for dist, index in zip(dists[0], indices[0]):
                found = {"index": index,
                         "distance": dist,
                         "source":self.meta["chunks"][index]["source"],
                         "content":self.meta["chunks"][index]["text"]
                         }
                found_list.append(found)
            return found_list
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

def gen_split_overlap(seq, size, overlap):
    if size < 1 or overlap < 0:
        raise ValueError('size must be >= 1 and overlap >= 0')

    # Yields a generator with a len and overlap
    for i in range(0, len(seq) - overlap, size - overlap):
        yield seq[i:i + size]

def recursive_split(text: str, separators: list[str], max_size: int, tokenizer=None) -> list[str]:

    def size_of(s: str) -> int:
        if tokenizer is not None:
            return len(tokenizer(s)["input_ids"])
        return len(s.split(" "))

    # Base case 1: already fits, nothing more to do
    if size_of(text) <= max_size:
        return [text]

    # Base case 2: out of separators, fall back to a hard cut
    if not separators:
        words = text.split(" ")
        return [" ".join(words[i:i + max_size]) for i in range(0, len(words), max_size)]

    # Recursive case: split on the first separator, recurse on remainder for oversized pieces
    sep, rest = separators[0], separators[1:]
    chunks = []
    for piece in text.split(sep):
        if not piece:
            continue
        if size_of(piece) <= max_size:
            chunks.append(piece)
        else:
            chunks.extend(recursive_split(piece, rest, max_size, tokenizer))
    return chunks
                
def main():
    print("Starting Knowledgebase testing \n")
    pdf_dir = r"PDFs\Digitaliseringsstyrelsen"
    save_dir = "test"

    print("Creating index with text_size 500 and text_overlap of 100")
    index = Index(text_size=400,
                  text_overlap=80,
                  model_id="intfloat/multilingual-e5-base"
                  )
    
    print(f"Adding directory {pdf_dir} to index")
    index.add_directory(pdf_dir)

    print(f"Saving index to {save_dir}")
    index.save(save_dir)

    found_files = os.listdir(os.fsencode(save_dir))
    print(f"Files found in {save_dir}:")
    if found_files:
        for file in found_files:
            filename = os.fsdecode(file)
            print(f"File found: {filename}")
    else:
        print(f"No files found in {save_dir}")

    print(f"Loading index from {save_dir}")
    del index
    index = Index.from_path(save_dir)


    query = "Hvordan beskrives AI til offentlige myndigheder?"
    print(f"searching faiss with query: {query}")
    results = index.search_index(query, 1)
    for id, result in enumerate(results):
        print(f"Result: Rank {id} | {result}")

    print(f"Test Complete")

if __name__ == "__main__":
    main()


