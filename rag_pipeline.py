from chunking import chunk_text
from embeddings import create_embeddings, model
from retriever import build_vector_store, retrieve_chunks
from llm import generate_answer


def process_document(text):

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    index = build_vector_store(embeddings)

    return chunks, index


def ask_question(question, chunks, index):

    query_embedding = model.encode([question])

    relevant_chunks = retrieve_chunks(query_embedding, index, chunks)

    context = "\n".join(relevant_chunks)

    answer = generate_answer(context, question)

    return answer, relevant_chunks