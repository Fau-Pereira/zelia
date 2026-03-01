import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Caminho do banco de dados
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(SCRIPT_DIR, "chroma_db")

# 1. Configurar Modelos (Exatamente os mesmos que já estavam a funcionar!)
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 2. Conectar ao Banco de Dados
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings_model
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 3. O NOVO PROMPT (Agora com espaço para o histórico)
template = """Você é a Zélia, a assistente virtual da universidade. 
Sua função é responder a dúvidas dos alunos baseando-se APENAS nos trechos do manual fornecidos.

Histórico recente da conversa:
{historico_conversa}

Trechos do manual encontrados:
{context}

Pergunta atual do aluno: {question}

Responda de forma clara, educada e direta. Se a resposta não estiver nos trechos do manual, diga que não tem essa informação no momento.
Resposta:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["historico_conversa", "context", "question"]
)

# 4. A Função RAG Atualizada
def get_rag_response(query: str, history: list = None) -> str:
    if history is None:
        history = []
        
    # Formata o histórico de uma lista para um texto legível para a IA
    historico_texto = ""
    # Pegamos apenas as últimas 4 mensagens para não sobrecarregar a memória
    for msg in history[-4:]: 
        remetente = "Aluno" if msg["role"] == "user" else "Zélia"
        historico_texto += f"{remetente}: {msg['content']}\n"
        
    if not historico_texto:
        historico_texto = "Nenhuma conversa anterior."

    # Busca documentos relevantes
    relevant_docs = retriever.invoke(query)
    contexto = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Preenche o prompt com o histórico, contexto e a pergunta
    final_prompt = QA_PROMPT.format(
        historico_conversa=historico_texto,
        context=contexto, 
        question=query
    )
    
    # Chama o Gemini
    response = llm.invoke(final_prompt)
    return response.content