import streamlit as st
import requests

# 1. CONFIGURAÇÃO DA PÁGINA (Aba do browser)
# Tem de ser o primeiro comando do Streamlit!
st.set_page_config(
    page_title="Zélia - Assistente Académica",
    page_icon="🎓",
    layout="centered"
)

# 2. BARRA LATERAL (Sidebar)
with st.sidebar:
    # Pode substituir o link abaixo por um URL do logótipo da sua universidade
    st.image("https://cdn-icons-png.flaticon.com/512/8066/8066119.png", width=120) 
    st.title("Sobre a Zélia")
    st.info(
        "Olá, sou a Zélia, a assistente virtual da Unijorge, criada para ajudar com dúvidas sobre o "
        "**Manual do Aluno** e processos académicos."
    )
    st.divider() # Linha de separação
    
    # Botão para limpar a memória da conversa
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun() # Atualiza o ecrã instantaneamente

# 3. CABEÇALHO PRINCIPAL
st.title("🎓 Zélia - Atendimento ao Aluno")
st.markdown("Olá! Pergunte-me o que precisar sobre regras, matrículas, faltas e muito mais.")

API_URL = "http://127.0.0.1:8000/perguntar"

# Inicializa o histórico se estiver vazio
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. EXIBIR MENSAGENS COM AVATARES PERSONALIZADOS
for message in st.session_state.messages:
    # Define o ícone com base em quem está a falar
    avatar_icon = "🧑‍🎓" if message["role"] == "user" else "👩‍🏫"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# 5. CAIXA DE TEXTO E ENVIO
if prompt := st.chat_input("Ex: Qual o prazo para trancamento de matrícula?"):
    
    # Mostra a pergunta do aluno no ecrã
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    # Chama a API e mostra a resposta da Zélia
    with st.chat_message("assistant", avatar="👩‍🏫"):
        message_placeholder = st.empty()
        message_placeholder.markdown("A consultar o manual... ⏳")
        
        try:
            # Envia a pergunta E o histórico
            payload = {
                "query": prompt,
                "history": st.session_state.messages
            }
            
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                resposta_ia = response.json().get("answer", "Desculpe, não consegui obter a resposta.")
                message_placeholder.markdown(resposta_ia)
                st.session_state.messages.append({"role": "assistant", "content": resposta_ia})
            else:
                message_placeholder.markdown(f"❌ Erro na API: Status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            message_placeholder.markdown("❌ Erro de ligação ao servidor FastAPI!")