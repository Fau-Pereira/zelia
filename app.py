import streamlit as st
import requests
import os

st.set_page_config(page_title="Zélia - Assistente Acadêmica", page_icon="🎓", layout="centered")

API_URL = os.getenv("API_URL", "http://localhost:8000/perguntar")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/7407/7407117.png", width=80) 
    st.title("Sobre a Zélia")
    st.info("Olá, sou a Zélia, a assistente virtual da Unijorge, criada para ajudar com dúvidas sobre o **Manual do Aluno** e processos académicos.")
    st.divider()
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🎓 Zélia - Atendimento ao Aluno")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar_icon = "🧑‍🎓" if message["role"] == "user" else "👩‍🏫"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Qual o prazo para trancamento de matrícula?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="👩‍🏫"):
        message_placeholder = st.empty()
        message_placeholder.markdown("A consultar o manual... ⏳")
        try:
            payload = {"query": prompt, "history": st.session_state.messages}
            response = requests.post(API_URL, json=payload, timeout=180)
            if response.status_code == 200:
                resposta_ia = response.json().get("answer", "Desculpe, erro.")
                message_placeholder.markdown(resposta_ia)
                st.session_state.messages.append({"role": "assistant", "content": resposta_ia})
            else:
                message_placeholder.markdown(f"❌ Erro na API: Status {response.status_code}")
        except Exception:
            message_placeholder.markdown("❌ Erro de ligação ao servidor FastAPI!")