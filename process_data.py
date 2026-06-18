import os
import logging
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("unstructured").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, "documentos") 
DB_DIR = os.path.join(SCRIPT_DIR, "chroma_db")

def processar_e_salvar_documentos():
    logging.info("--- Início do Processamento Híbrido ---")
    if not os.path.exists(DOCS_DIR): os.makedirs(DOCS_DIR)
    
    # Bypass de Ingestão (Relatório: reduz inicialização)
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        logging.info("Banco de dados já detectado. Pulando a leitura dos PDFs para inicialização rápida!")
        return
    
    logging.info("Lendo PDFs com OCR de alta resolução...")
    loader = DirectoryLoader(
        DOCS_DIR, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader,
        loader_kwargs={"strategy": "hi_res"}
    )
    documentos = loader.load()
    if not documentos: return

    # Chunking refinado (Relatório: 500 caracteres)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")

    logging.info("Salvando no banco vetorial...")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    logging.info("--- Concluído com Sucesso ---")

if __name__ == "__main__":
    processar_e_salvar_documentos()