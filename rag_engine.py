from docx import Document as DocxDocument
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle, os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

def load_word(path: str) -> list[Document]:
    docx = DocxDocument(path)
    full_text = "\n".join(p.text for p in docx.paragraphs if p.text.strip())
    # add timestamp so every upload is unique
    meta = {"source": f"{os.path.basename(path)} @ {datetime.now():%Y-%m-%d %H:%M:%S}"}
    return [Document(page_content=full_text, metadata=meta)]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # characters, not tokens
    chunk_overlap=10,     # keep some context between chunks
)

training = "training"
paths = os.listdir(training)
chunks = []
for doc in paths:
    docs = load_word(os.path.join('training\\',doc))
    chunks += splitter.split_documents(docs)
print(f"{len(chunks)} chunks")


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # replace with your OpenAI API key

db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss.idx")          # creates faiss.idx & index.pkl

def get_qa_chain():
    db = FAISS.load_local("faiss.idx", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o",        
        temperature=0.2,       
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",     
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain

def add_new_doc(file_path: str):
    print("Adding", file_path)          # simple sanity log
    new_chunks = splitter.split_documents(load_word(file_path))

    # 🧹  remove previous versions of that same file (optional)
    db = FAISS.load_local("faiss.idx", embeddings, allow_dangerous_deserialization=True)
    db.delete(where={"source": {"$contains": os.path.basename(file_path)}})

    db.add_documents(new_chunks)
    db.save_local("faiss.idx")