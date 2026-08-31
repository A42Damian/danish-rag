# danish-rag
Attempting to understand and create a Retrieval Augmented Generation (RAG) system

At this point it includes:
## A Knowledge Base:
Can take a directory and will then load, chunk, and vectorize PDFs found in it.
The vectors are used in a FAISS index for semantic search based on a given string (question).

## A Retriever
Takes a question, passes it to the knowledge base and returns the found chunks.
Currently it just passes what the Index found directly. It is abstracted as its own class for possible future expansion.

## A generator
Takes the question and additional context found by the reciever to generate an answer based on the given context.

The model in use is the very recently published Mimir. (https://huggingface.co/danish-foundation-models/DFM-Mimir)
It is chosen because of its high performance on few parameters, meaning it can run locally even on limited hardware.
It is also a model specialized for danish and english making it an obvious chouive for a *danish*-RAG system.

The model is also licensed under the Apachee 2.0 license.
