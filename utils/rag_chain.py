from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from utils.vectorstore import load_vector_store


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():

    db = load_vector_store()

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    llm = ChatOllama(
        model="llama3",
        temperature=0.1,
        num_ctx=4096
    )

    prompt = ChatPromptTemplate.from_template("""
You are an intelligent PDF document assistant.

Your task is to answer questions ONLY from the provided context.

STRICT RULES:
- When confident give detailed answers
- Use ONLY the retrieved context.
- If the answer is not present, say:
  "I could not find this information in the document."
- DO NOT guess.
- Keep answers accurate and professional.

Context:
{context}

Question:
{input}

Answer:
""")

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
