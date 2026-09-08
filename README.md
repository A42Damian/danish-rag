# danish-rag
Attempting to understand and create a Retrieval Augmented Generation (RAG) system

At this point it includes:
## A Knowledge Base:
Can take a directory and will then load, chunk, and vectorize PDFs found in it.
The vectors are used in a FAISS index for semantic search based on a given string (question).

### Document Loading and Chunking
#### V1 (Number of Tokens)
The first version used the pypdf PDFReader. Which just returns a whole PDF as a single string.
For chunking of the PDF it was just naively chunked based on number of tokens in a section. The embedding model had a token limit of around 512, so each chunk was 400 and then 80 tokens of overlap.

#### V1.5 (Paragraphs, Lines, Punctuation, Size)
It started with trying to create a function that would separate the text into chunks if the text exceeded a maximum size.
The separation would be based on a ordered list, so that it would separate paragraphs, then lineshifts, then periods, and so on. With a default case of just being the size.
In the end this idea was dropped because it added a lot of complexity and many possible problems with inconsistent PDF text structuring.

#### V2 (Docling)
Since the project was not to understand complicated heuristics for PDF reading, pre-existing libraries were explored.
The Docling library was chosen because of its robustness and high evaluation scores.
Docling is able to handle many document types and is a deep-learning model based approach to extracting data from varied sources both in structure and content.
Since this will also be replacing the pydf PDFReader, more meta data can also be included in the finale sourcing for the user. Having page numbers and even location on the page as possible information the user can be given.


## A Retriever
Takes a question, passes it to the knowledge base and returns the found chunks.
Currently it just passes what the Index found directly. It is abstracted as its own class for possible future expansion.

## A generator
Takes the question and additional context found by the reciever to generate an answer based on the given context.

The model in use is the very recently published Mimir. (https://huggingface.co/danish-foundation-models/DFM-Mimir)
It is chosen because of its high performance on few parameters, meaning it can run locally even on limited hardware.
It is also a model specialized for danish and english making it an obvious choice for a *danish*-RAG system.

The model is also licensed under the Apachee 2.0 license.
