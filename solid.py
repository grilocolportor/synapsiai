from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Protocol

import os
import glob
import logging
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DB_NAME = "vector_db"

# --- SRP: Configuração dos Embeddings ---
class EmbeddingConfig:
    @staticmethod
    def get_embedding_model():
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# --- SRP: Configuração da Vector Store ---
class VectorStoreManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.vectorstore = None

    def setup_vectorstore(self, documents: List[Document]):
        embedding_model = EmbeddingConfig.get_embedding_model()
        if os.path.exists(self.db_name):
            Chroma(persist_directory=self.db_name, embedding_function=embedding_model).delete_collection()

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=self.db_name,
        )
        return self.vectorstore

# --- SRP: Carregamento de Documentos ---
class DocumentLoader:
    @staticmethod
    def load_documents(folder_path: str) -> List[Document]:
        folders = glob.glob(f"{folder_path}/*")
        documents = []
        for folder in folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(folder, glob="*.md", loader_cls=TextLoader)
            folder_docs = loader.load()
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        return documents

# --- SRP: Divisão de Texto ---
class TextSplitter:
    @staticmethod
    def split_documents(documents: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

# --- SRP: Configuração dos Modelos ---
class ModelConfig:
    @staticmethod
    def get_model(model_class: str):
        from langchain_ollama import ChatOllama
        from langchain_openai import ChatOpenAI
        from langchain_community.llms import HuggingFaceHub

        if model_class == "hf_hub":
            return HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", model_kwargs={"temperature": 0.1, "max_new_tokens": 512})
        elif model_class == "openai":
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        elif model_class == "ollama":
            return ChatOllama(model="phi3", temperature=0.1)
        else:
            raise ValueError("Modelo desconhecido")

# --- DIP: Interface para Cadeia de Conversa ---
class ConversationChainProtocol(Protocol):
    def generate_response(self, user_query: str, chat_history: List[HumanMessage | AIMessage]) -> str:
        ...

# --- Implementação da Cadeia de Conversa ---
class ConversationChain(ConversationChainProtocol):
    def __init__(self, llm, retriever, memory):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory

    def generate_response(self, user_query: str, chat_history: List[HumanMessage | AIMessage]) -> str:
        try:
            retrieved_context = self.retriever.get_relevant_documents(user_query)
            full_context = "\n".join([msg.content for msg in chat_history] + [doc.page_content for doc in retrieved_context])
            formatted_input = f"{full_context}\nUser: {user_query}\nAssistant:"
            response = self.llm.invoke(formatted_input)
            response_text = response["output"] if isinstance(response, dict) else response
            self.memory.save_context({"input": user_query}, {"output": response_text})
            return response_text
        except Exception as e:
            logger.error(f"Erro na cadeia de conversa: {e}")
            raise HTTPException(status_code=500, detail="Erro ao processar resposta")

# --- FastAPI Models ---
class ChatHistoryItem(BaseModel):
    message: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryItem]

class ChatResponse(BaseModel):
    response: str
    updated_history: List[Dict]

# --- Endpoint FastAPI ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        chat_history = [
            HumanMessage(content=msg.message) if idx % 2 == 0 else AIMessage(content=msg.content)
            for idx, msg in enumerate(request.history)
        ]

        llm = ModelConfig.get_model(model_class="ollama")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        vectorstore = VectorStoreManager(DB_NAME).setup_vectorstore(TextSplitter.split_documents(DocumentLoader.load_documents("knowledge-base")))

        chain = ConversationChain(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
        response = chain.generate_response(request.message, chat_history)

        updated_history = [{"message": item.message, "content": item.content} for item in request.history]
        updated_history.append({"message": request.message, "content": response})

        return ChatResponse(response=response, updated_history=updated_history)
    except Exception as e:
        logger.error(f"Erro no endpoint: {e}")
        raise HTTPException(status_code=500, detail="Erro no processamento da requisição")
