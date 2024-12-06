from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.memory.buffer import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

import logging
import glob
import os

load_dotenv()

db_name = "vector_db"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ou especifique os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

folders = glob.glob("knowledge-base/*")

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="*.md", loader_cls=TextLoader)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

doc_types = set(chunk.metadata["doc_type"] for chunk in chunks)

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model selector
model_class = "ollama"  # "hf_hub", "openai", "ollama"

# Models setup
def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    return HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "max_new_tokens": 512,
        },
    )

def model_openai(model="gpt-4o-mini", temperature=0.1):
    return ChatOpenAI(model=model, temperature=temperature)

def model_ollama(model="phi3", temperature=0.1):
    return ChatOllama(model=model, temperature=temperature)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embedding_model).delete_collection()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=db_name,
)

# Exemplo de recuperação de embedding
sample_embedding = vectorstore.similarity_search("Exemplo de consulta", k=1)[0]
print(sample_embedding)

# Generate response
def model_response(user_query, chat_history, model_class):
    try:
        if model_class == "hf_hub":
            llm = model_hf_hub()
        elif model_class == "openai":
            llm = model_openai()
        elif model_class == "ollama":
            llm = model_ollama()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retriever = vectorstore.as_retriever()

        # Cadeia de recuperação de conversa customizada
        class CustomConversationChain:
            def __init__(self, llm, retriever, memory):
                self.llm = llm
                self.retriever = retriever
                self.memory = memory

            def __call__(self, user_query, chat_history):
                # Recuperar contexto do histórico de conversas
                retrieved_context = self.retriever.get_relevant_documents(user_query)
                # Formatar a entrada para o modelo
                full_context = "\n".join([msg.content for msg in chat_history] + [doc.page_content for doc in retrieved_context])
                formatted_input = f"{full_context}\nUser: {user_query}\nAssistant:"
                # Obter resposta do modelo
                response = self.llm.invoke(formatted_input)  # Passar a entrada como string
                # Verificar e garantir que a resposta seja uma string
                response_text = response["output"] if isinstance(response, dict) else response
                # Armazenar na memória
                self.memory.save_context({"input": user_query}, {"output": str(response_text)})
                return response_text

        chain = CustomConversationChain(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        response = chain(user_query, chat_history)

        logger.info(f"Modelo respondeu com: {response}")
        return response
    except Exception as e:
        logger.error(f"Erro ao gerar resposta do modelo: {e}")
        return "Desculpe, houve um erro ao gerar a resposta."

# Modelo para o item de histórico (representando um único chat)
class ChatHistoryItem(BaseModel):
    message: str
    content: str

# FastAPI models
class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryItem]

class ChatResponse(BaseModel):
    response: str
    updated_history: List[Dict]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Recebendo requisição: {request.json()}")

        chat_history = [
            HumanMessage(content=msg.message) if idx % 2 == 0 else AIMessage(content=msg.content)
            for idx, msg in enumerate(request.history)
        ]

        response = model_response(
            user_query=request.message,
            chat_history=chat_history,
            model_class=model_class,
        )

        # Garantir que `response` seja uma string
        if isinstance(response, AIMessage):
            response = response.content  # Extrai o conteúdo de texto

        # Construir a lista updated_history como dicionários
        updated_history = [{"message": item.message, "content": item.content} for item in request.history]
        updated_history.append({"message": request.message, "content": response})

        logger.info(f"Respondendo com: {response}")

        return ChatResponse(response=response, updated_history=updated_history)
    except Exception as e:
        logger.error(f"Erro no endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.middleware("http")
async def add_charset_to_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response
