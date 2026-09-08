from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import faiss
import json
import os
from typing import Any, Dict

from document_loaders import pdf_loaders

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker



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
        self.converter = DocumentConverter()

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

    def extract_chunk_metadata(self, chunk) -> Dict[str, Any]:
        metadata = {
            "text": chunk.text,
            "headings": [],
            "page_info": None,
            "content_type": None
        }
        return metadata

    def add_pdf(self, pdf_path):
        print(f"Adding PDF from {pdf_path}")
        result = self.converter.convert(pdf_path)
        doc = result.document

        chunker = HybridChunker(tokenizer=self.tokenizer)

        chunks = list(chunker.chunk(doc))

        embed_list = []

        for idx, chunk in enumerate(chunks):
            chunk_md = self.extract_chunk_metadata(chunk)

            vec = self.embed_text(chunk_md["text"])
            vec = np.array(vec, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(vec)
            embed_list.append(vec)

            self.meta["chunks"].append({
                "source": pdf_path,
                "chunk_id": idx,
                "text": chunk_md["text"],
                "headings": json.dumps(chunk_md['headings']),
                "page_info": chunk_md["page_info"],
                "content_type": chunk_md["content_type"]
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


