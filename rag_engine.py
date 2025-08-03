from docx import Document as DocxDocument
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import pickle, os
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

# prompt template for the question-answering chain
template = """
You are a helpful assistant. Use the following context to answer the question.
If the answer is not contained in the context, say: "I’m sorry, I don’t have enough information to answer that.
be friendly and concise in your response.
if the user greets you, greet them back.
find the best answer to the question based on the context provided even if it is not a direct answer.
be proffessional and concise in your response.
do not be ambiguous, do not say "I don't know" or "I am not sure", instead provide the best possible answer based on the context.
in a question like "who are you?" or "what is your name?" answer with "I am a helpful assistant." or "I am a language model created by Lxera." or similar.
search deeply in the context for the answer, do not make up answers.
the user may ask you to "search" or "find" something, do not search the web, only use the context provided.
the user may ask you to "summarize" or "explain" something, do not summarize or explain the context, only answer the question.
the user may ask you to "translate" something, do not translate the context, only answer the question.
the user may ask you to "paraphrase" something, do not paraphrase the context, only answer the question.
the user may ask you to "compare" something, do not compare the context, only answer the question.
the user may ask you to "list" something, do not list the context, only answer the question.
the user may ask you to "describe" something, do not describe the context, only answer the question.
the user may ask you to "define" something, do not define the context, only answer the question.
the user may ask you to "explain" something, do not explain the context, only answer the question.
the user may ask you to "give an example" of something, do not give an example of the context, only answer the question.
the user may ask you to "give a summary" of something, do not give a summary of the context, only answer the question.
the uer language may be different, detect it and respond in the same language.

Context:
{context}

Question:
{question}

Answer:
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# urls = [
#     "https://acadimacollege.com/",
#     "https://anasacademy.uk",
#     "https://lxera.net/"
# ]
# def load_url(url: str) -> list[Document]:
#     headers = {
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/115.0.0.0 Safari/537.36"
#         ),
#         "Accept-Language": "en-US,en;q=0.9",
#         "Referer": "https://google.com",
#     }

#     response = requests.get(url, headers=headers)
#     if response.status_code != 200:
#         raise Exception(f"Failed to load URL: {url} (status {response.status_code})")

#     soup = BeautifulSoup(response.text, "html.parser")

#     for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
#         tag.decompose()

#     text = soup.get_text(separator="\n")
#     clean_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

#     meta = {"source": f"{url} @ {datetime.now():%Y-%m-%d %H:%M:%S}"}
#     return [Document(page_content=clean_text, metadata=meta)]




def get_qa_chain():
    db = FAISS.load_local("faiss.idx", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 20})
    

    llm = ChatOpenAI(
        model="gpt-4o",        
        temperature=0.2,       
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",     
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_PROMPT
        }  
    )
    return qa_chain

# def add_new_doc(file_path: str):
#     print("Adding", file_path)          # simple sanity log
#     new_chunks = splitter.split_documents(load_word(file_path))

#     # 🧹  remove previous versions of that same file (optional)
#     db = FAISS.load_local("faiss.idx", embeddings, allow_dangerous_deserialization=True)
#     db.delete(where={"source": {"$contains": os.path.basename(file_path)}})

#     db.add_documents(new_chunks)
#     db.save_local("faiss.idx")