import os
from datetime import datetime
from dotenv import load_dotenv

# Importações atualizadas
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent 

import asyncio
from get_webdata import main

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(SCRIPT_DIR, "chroma_db")

# 1. Configurar Modelos
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 2. Conectar ao Banco de Dados Vetorial
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings_model
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# ==========================================
# 🛠️ CRIAÇÃO DAS FERRAMENTAS DO AGENTE
# ==========================================

@tool
def pesquisa_base_conhecimento(query: str) -> str:
    """Pesquisa informações oficiais nos documentos da universidade. Use SEMPRE esta ferramenta para responder a dúvidas institucionais ou regras."""
    docs = retriever.invoke(query)
    
    contextos_formatados = []
    for doc in docs:
        fonte = os.path.basename(doc.metadata.get('source', 'Documento Desconhecido'))
        pagina = doc.metadata.get('page', 'N/A')
        bloco = f"[ARQUIVO: {fonte} | PÁGINA: {pagina}]\n{doc.page_content}"
        contextos_formatados.append(bloco)
        
    return "\n\n---\n\n".join(contextos_formatados)

@tool
def consultar_data_atual() -> str:
    """Consulta a data e hora atual do sistema em tempo real. Use esta ferramenta para saber o dia de hoje ao calcular prazos."""
    agora = datetime.now()
    return agora.strftime("%d/%m/%Y, %H:%M")

@tool
def consultar_eventos() -> str:
    """Consulta os dados informando sobre os eventos institucionais. De primeira vista, os dados estarão como se acabassem de ser colados, você organizará os dados de acordo com cada evento, e retornará com precisão de acordo com o que o usuário pedir. Use sempre que o usuário pedir informação sobre algum evento. Em caso de interesse em algum evento específicom ou quando a lista não for tão grande, mostre também o link de inscrição, caso haja."""
    return asyncio.run(main())

@tool
def consultar_evento_especifico(query:str) -> str:
    """Consulta os dados detalhados sobre um determinado evento. 
     na {query}, quero que você pegue apenas o link sobre o evento que ele quer saber mais, e retorne normalmente.
       Use sempre que o usuário pedir informação detalhadasobre algum evento e use SOMENTE após a ferramenta consultar eventos já estiver sido consultada. Lembre-se de retornar o link para inscrição também."""
    return asyncio.run(main(query))

ferramentas = [pesquisa_base_conhecimento, consultar_data_atual, consultar_eventos, consultar_evento_especifico]


# ==========================================
# 🧠 CONFIGURAÇÃO DO AGENTE LANGGRAPH
# ==========================================

INSTRUCOES_SISTEMA = """Você é a Zélia, uma assistente virtual autónoma da universidade.
Você tem ferramentas ao seu dispor. Sempre que não souber algo, pare e use a ferramenta apropriada.
Se usar a 'pesquisa_base_conhecimento', lembre-se OBRIGATORIAMENTE de citar a fonte da informação no final da sua resposta.

1. Ao responder com base na ferramenta, você DEVE separar a resposta da fonte usando uma linha horizontal (---).  

2. Formate a citação exatamente assim:  

👉 **Fonte:** [Nome do Arquivo] (Pág. [Número])  

EXEMPLO DE RESPOSTA:  

A matrícula para o próximo semestre começa no dia 15 de maio.  
---
 👉 **Fonte:** calendario_academico_2026.pdf (Pág. 4) 

Seja amigável, clara e direta."""

# Cria o agente autónomo BÁSICO (Sem modificadores que causam erro de versão)
agente = create_react_agent(llm, ferramentas)


# ==========================================
# 🚀 FUNÇÃO DE COMUNICAÇÃO (Chamada pela API)
# ==========================================
def get_rag_response(query: str, history: list = None) -> str:
    if history is None:
        history = []
        
    # Colocamos a instrução do sistema diretamente como a PRIMEIRA mensagem
    mensagens = [
        ("system", INSTRUCOES_SISTEMA)
    ]
    
    # 1. Injeta o histórico da conversa
    for msg in history[-4:]: 
        if msg["role"] == "user":
            mensagens.append(("human", msg["content"]))
        else:
            mensagens.append(("ai", msg["content"]))
            
    # 2. Injeta a pergunta nova do aluno
    mensagens.append(("human", query))

   # 3. O Agente entra em ação, escolhe as ferramentas e responde
    resposta = agente.invoke({"messages": mensagens})
    
    conteudo_bruto = resposta["messages"][-1].content
    
    # Se a resposta vier como um "pacote de dados" (lista com assinatura), extraímos só o texto
    if isinstance(conteudo_bruto, list):
        for bloco in conteudo_bruto:
            if isinstance(bloco, dict) and 'text' in bloco:
                return bloco['text']
        return str(conteudo_bruto) # Fallback de segurança
        
    # Se já vier como texto simples, devolvemos direto
    return conteudo_bruto