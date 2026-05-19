from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader
from langchain_core.documents import Document
import requests
import asyncio
from playwright.async_api import async_playwright
import os



def loading():
    #Utilizando WebBaseLoader

    # loader = WebBaseLoader('https://eventos.unijorge.com.br/eventos/')
    # dados = loader.load()
    # for i in dados:
    #     print(i.page_content[:500])

    #utilizando PlaywrightURLLoader pip install playwright
    url = PlaywrightURLLoader('https://eventos.unijorge.com.br/eventos/')
    loader = PlaywrightURLLoader(urls=url, remove_selectors=['header', 'footer'])
    dados = loader.load()
    print(dados)

    loader = PlaywrightURLLoader(urls=['https://eventos.unijorge.com.br/eventos/'])
    data = loader.load()

    # Para ver se funcionou
    for i in data:
        print(i)


def api():
    url = requests.get('https://www.even3.com.br/api/v1/customclients/events/unijorge')
    dados = url.json()
    print(dados)


# loading()


async def extrair_com_scroll_infinito(url):
    """Esta função usa o Playwright para acessar a página de eventos da Unijorge, rolar a página para baixo várias vezes para garantir que todos os eventos sejam carregados (mesmo aqueles que aparecem apenas com o scroll), e então extrai o texto completo da página. O resultado é um documento LangChain contendo o texto extraído, que pode ser usado como fonte de conhecimento para a Zélia responder perguntas sobre os eventos."""
    
    async with async_playwright() as p:
        # headless=True roda em segundo plano. Mude para False se quiser ver a mágica acontecendo.
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = await context.new_page()
        
        print(f"Conectando a {url}...")
        await page.goto(url, wait_until="networkidle")
        
        # Espera inicial para os primeiros cards aparecerem
        await page.wait_for_timeout(3000)

        print("Iniciando scroll para carregar todos os eventos...")
        
        # Vamos rolar a página 5 vezes
        for i in range(2):
            # Rola para o fim da página
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            print(f"Rolagem {i+1} executada...")
            
            # Espera o 'loader' do site trabalhar e trazer novos eventos
            await page.wait_for_timeout(2000) 

        # Captura o texto final
        #Código em JavaScript para extrair o link e adicioanr ao texto
        print("Extraindo texto completo...")
        content = await page.evaluate("""() => {
        // Transforma todos os links da página em formato Markdown [Texto](URL)
        const links = document.querySelectorAll('a');
        links.forEach(link => {
            const href = link.href;
            const text = link.innerText.trim();
            if (href && text) {
                const markdownLink = ` [${text}](${href}) `;
                link.parentNode.replaceChild(document.createTextNode(markdownLink), link);
            }
        });
        return document.body.innerText;
    }""")
        
        await browser.close()
        return content

#@tool # Decorador do LangChain para transformar esta função em uma ferramenta utilizável pelo agente
async def main(url: str ='https://eventos.unijorge.com.br/eventos/'):
    """Extrai o texto da página de eventos da Unijorge, rolando a página para garantir que todos os eventos sejam carregados. Retorna os primeiros 2000 caracteres do conteúdo extraído para validação. Use esta ferramenta para obter informações atualizadas sobre os eventos que ocorrerão na universidade."""
    texto = await extrair_com_scroll_infinito(url)
    
    # Criando o documento LangChain
    doc = Document(page_content=texto, metadata={"source": url})
    
    # Validação rápida
    #eventos_detectados = doc.page_content.count("Inscrição") # Geralmente cada card tem esse botão
    # print(f"\n--- RELATÓRIO ---")
    # print(f"Total aproximado de eventos encontrados: {eventos_detectados}")
    # print(f"Prévia do conteúdo:\n{doc.page_content[:2000]}") # Mostra os primeiros 2000 caracteres para validação
    return doc.page_content[300:] # Retorna só os primeiros 2000 caracteres para evitar sobrecarga


os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

if __name__ == "__main__":
      print(asyncio.run(extrair_com_scroll_infinito("https://eventos.unijorge.com.br/eventos/")))