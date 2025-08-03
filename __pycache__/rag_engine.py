from docx import Document as DocxDocument
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

# prompt template for the question-answering chain
template = """
You are a helpful assistant. Use the following context to answer the question.
If the answer is not contained in the context, say: "I’m sorry, I don’t have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

def get_qa_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 
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