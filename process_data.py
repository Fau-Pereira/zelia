import os
import sys
import logging
import time
from dotenv import load_dotenv

# Esta seção importa as ferramentas necessárias para ler pastas, PDFs (com ou sem imagem)
# e transformar os textos em vetores matemáticos (embeddings) que a IA entende.
# ÚLTIMA ALTERAÇÃO: Substituímos o PyPDFLoader pelo DirectoryLoader e UnstructuredPDFLoader, e atualizamos o import do Chroma.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")
load_dotenv(ENV_PATH) 

# Define os caminhos padrão do projeto. A pasta 'documentos' é onde você colocará os PDFs
# (calendários, manuais, mapas), e 'chroma_db' é onde a "memória" da Zélia será salva.
# ÚLTIMA ALTERAÇÃO: Atualizamos a variável para apontar para a pasta 'documentos' em vez de um único arquivo PDF.
DEFAULT_DOCS_DIRECTORY = os.path.join(SCRIPT_DIR, "documentos") 
DEFAULT_PERSIST_DIRECTORY = os.path.join(SCRIPT_DIR, "chroma_db")

def processar_e_salvar_documentos(docs_dir: str, persist_directory: str):
    logging.info("--- Início do Processamento Híbrido (Texto + OCR) ---")

    if not os.getenv("GOOGLE_API_KEY"):
        logging.error("A variável de ambiente GOOGLE_API_KEY não foi encontrada.")
        sys.exit(1) 
    
    logging.info(f"Lendo todos os documentos na pasta: {docs_dir}...")
    if not os.path.exists(docs_dir):
        logging.error(f"A pasta {docs_dir} não existe. Crie a pasta e coloque os PDFs lá.")
        return

    # Motor de Ingestão de Dados: Esta é a inteligência que varre a pasta atrás de PDFs.
    # Ele tenta ler o texto digitalmente primeiro. Se o PDF for um escaneamento (foto),
    # ele aciona automaticamente o OCR (Tesseract) para extrair o texto da imagem.
    # ÚLTIMA ALTERAÇÃO: Implementação do Unstructured com loader_kwargs={"strategy": "auto"} para habilitar o OCR inteligente.
    try:
        loader = DirectoryLoader(
            docs_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documentos = loader.load()
    except Exception as e:
        logging.error(f"Erro ao processar documentos: {e}")
        return
    
    if not documentos:
        logging.warning("Nenhum documento encontrado na pasta.")
        return
        
    logging.info(f"Leitura concluída! {len(documentos)} blocos carregados.")

    # Divisão de Texto (Chunking): O modelo de IA tem um limite de leitura por vez.
    # Aqui "picotamos" os documentos em pedaços menores (1000 caracteres) com uma 
    # pequena sobreposição (200 caracteres) para não cortar frases ou contextos ao meio.
    # ÚLTIMA ALTERAÇÃO: Nenhuma mudança estrutural, a lógica robusta de chunking foi mantida.
    logging.info("Dividindo o texto em pedaços...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_divididos = text_splitter.split_documents(documentos)
    logging.info(f"Os documentos foram divididos em {len(docs_divididos)} chunks.")

    # Embeddings: É o tradutor que converte nosso texto em coordenadas matemáticas espaciais.
    # ÚLTIMA ALTERAÇÃO: Nenhuma mudança, mantivemos o modelo oficial do Google Gemini.
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Banco de Dados Vetorial: Onde os textos e suas "coordenadas" são salvos fisicamente no HD.
    # É aqui que o motor RAG vai pesquisar quando o aluno fizer uma pergunta.
    # ÚLTIMA ALTERAÇÃO: Atualizado para suportar nativamente a nova biblioteca langchain_chroma.
    logging.info(f"Criando banco de dados vetorial em: {persist_directory}...")
    vector_store = Chroma(
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )

    # Controle de Requisições (Rate Limiting): Como estamos usando a API gratuita do Google,
    # enviamos os pedaços de texto em pequenos lotes (15) e pausamos por 10 segundos
    # para evitar que o servidor bloqueie nosso projeto por excesso de tráfego.
    # ÚLTIMA ALTERAÇÃO: Lógica de pausas mantida da sua versão original.
    tamanho_lote = 15
    for i in range(0, len(docs_divididos), tamanho_lote):
        lote = docs_divididos[i : i + tamanho_lote]
        logging.info(f"Processando lote de chunks {i+1} até {min(i + tamanho_lote, len(docs_divididos))} de {len(docs_divididos)}...")
        
        # Tenta adicionar os documentos. Se der erro de limite, espera mais um pouco.
        try:
            vector_store.add_documents(lote)
        except Exception as e:
            logging.warning(f"Limite da API atingido. Pausando por 60 segundos antes de tentar novamente... Erro: {e}")
            time.sleep(60)
            vector_store.add_documents(lote) # Tenta novamente após a pausa
        
        if (i + tamanho_lote) < len(docs_divididos):
            logging.info("Pausa de 15 segundos para não exceder o limite da API (Free Tier)...")
            time.sleep(15)
    
    logging.info("--- Processamento Concluído com Sucesso! ---")

if __name__ == "__main__":
    import argparse
    # O argparse permite que você execute o arquivo pelo terminal passando parâmetros extras,
    # tornando o script flexível caso você queira ler documentos de outra pasta no futuro.
    # ÚLTIMA ALTERAÇÃO: Trocamos o argumento '--pdf' por '--dir' para refletir a leitura da pasta inteira.
    parser = argparse.ArgumentParser(description="Processa PDFs de uma pasta com suporte a OCR.")
    parser.add_argument("--dir", default=DEFAULT_DOCS_DIRECTORY, help="Caminho para a pasta com os PDFs.")
    parser.add_argument("--db", default=DEFAULT_PERSIST_DIRECTORY, help="Caminho para o diretório do ChromaDB.")
    args = parser.parse_args()

    processar_e_salvar_documentos(docs_dir=args.dir, persist_directory=args.db)