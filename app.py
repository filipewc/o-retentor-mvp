"""
O Retentor — Pilar 1: Motor de Congruencia · Dashboard
========================================================
Dashboard Streamlit para pitch de vendas executivo.
Analisa congruencia textual e visual entre anuncios e artigos
de blog via Gemini (multimodal).

Executar: streamlit run app.py
"""

import json
import time
import textwrap
from io import StringIO, BytesIO
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image
import google.generativeai as genai


# ====================================================================
#  Constantes do Motor
# ====================================================================

HEADERS_NAVEGADOR = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
}

SELETORES_CONTEUDO = [
    "article",
    '[role="main"]',
    ".post-content",
    ".entry-content",
    ".article-body",
    ".blog-post",
    "main",
]

LIMITE_PALAVRAS = 5_000

# Dimensao maxima do thumbnail exibido na UI
THUMBNAIL_MAX = (180, 180)

PROMPT_SISTEMA = textwrap.dedent("""\
    Voce e um especialista senior em retencao de trafego e copywriting
    para blogs monetizados com publicidade.

    Voce recebera:
    - TEXTO DO ANUNCIO: a copy textual que o leitor viu antes de clicar.
    - TEXTO DO ARTIGO: o conteudo completo (ou parcial) do blog de destino.
    - IMAGEM DO CRIATIVO (opcional): a peca visual do anuncio que gerou o clique.

    Sua funcao e avaliar se o conteudo do artigo CUMPRE a promessa feita
    no anuncio — tanto a promessa textual quanto a promessa visual, quando
    a imagem do criativo for fornecida.

    Quando a imagem estiver presente, avalie:
    - Se o tom visual (cores, estilo, tema) do criativo e coerente com o
      conteudo e o tom do artigo.
    - Se elementos visuais do criativo (produtos, cenarios, pessoas, graficos)
      sao mencionados ou entregues nos primeiros paragrafos do artigo.
    - Se existe quebra de expectativa visual: o criativo promete uma coisa
      visualmente e o artigo entrega outra.

    Quando apenas o texto do anuncio for fornecido (sem imagem), avalie
    exclusivamente a congruencia textual.

    Criterios para o score:
    - 90-100: Congruencia perfeita. O artigo entrega exatamente o que o
              anuncio promete (visual e textualmente), logo nos primeiros
              paragrafos.
    - 70-89:  Boa congruencia. A promessa e cumprida, mas poderia ser mais
              direta no inicio.
    - 50-69:  Congruencia parcial. O tema e o mesmo, mas a promessa
              especifica do anuncio demora ou nao aparece com clareza.
    - 30-49:  Congruencia fraca. O artigo tangencia o tema, mas o leitor
              provavelmente se sentira enganado.
    - 0-29:   Incongruente. O artigo nao tem relacao relevante com a
              promessa do anuncio.

    Seja direto, pratico e acionavel na sugestao.
""")

SCHEMA_RESPOSTA = {
    "type": "object",
    "properties": {
        "score_congruencia": {
            "type": "integer",
            "description": "Score de 0 a 100 indicando alinhamento entre anuncio e artigo",
        },
        "diagnostico": {
            "type": "string",
            "description": "Diagnostico de 2 a 4 frases sobre a congruencia encontrada",
        },
        "promessa_entregue_no_inicio": {
            "type": "boolean",
            "description": "True se a promessa do anuncio aparece nos primeiros paragrafos",
        },
        "sugestao_primeiro_paragrafo": {
            "type": "string",
            "description": "Sugestao pratica de reescrita do primeiro paragrafo para reter o leitor",
        },
    },
    "required": [
        "score_congruencia",
        "diagnostico",
        "promessa_entregue_no_inicio",
        "sugestao_primeiro_paragrafo",
    ],
}


# ====================================================================
#  Dataclass de Resultado
# ====================================================================

@dataclass
class ResultadoAnalise:
    url_artigo: str
    texto_anuncio: str
    score_congruencia: Optional[int] = None
    promessa_entregue_no_inicio: Optional[bool] = None
    diagnostico: Optional[str] = None
    sugestao_primeiro_paragrafo: Optional[str] = None
    status_erro: Optional[str] = None
    com_imagem: bool = False

    @property
    def sucesso(self) -> bool:
        return not self.status_erro

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status_erro"] = d["status_erro"] or ""
        return d


# ====================================================================
#  Motor de Scraping
# ====================================================================

def extrair_texto_blog(url: str, timeout: int = 15) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        resposta = requests.get(url, headers=HEADERS_NAVEGADOR, timeout=timeout)
        resposta.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Falha de conexao: servidor inacessivel")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout apos {timeout}s")
    except requests.exceptions.HTTPError:
        raise RuntimeError(f"HTTP {resposta.status_code}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro de requisicao: {type(e).__name__}")

    soup = BeautifulSoup(resposta.text, "html.parser")

    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    conteudo = None
    for seletor in SELETORES_CONTEUDO:
        conteudo = soup.select_one(seletor)
        if conteudo and len(conteudo.get_text(strip=True)) > 200:
            break
        conteudo = None

    if conteudo is None:
        conteudo = soup.find("body")

    if conteudo is None:
        raise RuntimeError("Nenhum conteudo textual extraido da pagina")

    texto = conteudo.get_text(separator="\n", strip=True)

    if len(texto.split()) < 50:
        raise RuntimeError(f"Conteudo muito curto ({len(texto.split())} palavras)")

    palavras = texto.split()
    if len(palavras) > LIMITE_PALAVRAS:
        texto = " ".join(palavras[:LIMITE_PALAVRAS]) + "\n[...texto truncado...]"

    return texto


# ====================================================================
#  Processamento de Imagem
# ====================================================================

def processar_imagem_upload(arquivo_upload) -> Image.Image:
    """Abre e valida a imagem enviada pelo usuario."""
    try:
        imagem = Image.open(arquivo_upload)
        # Converte para RGB se necessario (ex: RGBA, P)
        if imagem.mode not in ("RGB", "L"):
            imagem = imagem.convert("RGB")
        return imagem
    except Exception as e:
        raise RuntimeError(f"Erro ao processar imagem: {e}")


def gerar_thumbnail(imagem: Image.Image, max_size: tuple = THUMBNAIL_MAX) -> Image.Image:
    """Gera uma copia reduzida para exibicao na interface."""
    thumb = imagem.copy()
    thumb.thumbnail(max_size, Image.LANCZOS)
    return thumb


# ====================================================================
#  Motor de Analise (Gemini) — Multimodal
# ====================================================================

def criar_modelo(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        system_instruction=PROMPT_SISTEMA,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA_RESPOSTA,
            temperature=0.3,
        ),
    )


def analisar_congruencia(
    modelo: genai.GenerativeModel,
    texto_anuncio: str,
    texto_artigo: str,
    imagem: Optional[Image.Image] = None,
) -> dict:
    """
    Envia os textos (e opcionalmente a imagem) ao Gemini.
    Se a imagem estiver presente, envia como conteudo multimodal.
    """
    modo = "textual e visual" if imagem else "textual"

    prompt = (
        f"MODO DE ANALISE: {modo}\n\n"
        f"TEXTO DO ANUNCIO:\n"
        f"---\n{texto_anuncio.strip()}\n---\n\n"
        f"TEXTO DO ARTIGO (primeiros paragrafos sao os mais importantes):\n"
        f"---\n{texto_artigo}\n---"
    )

    if imagem:
        prompt += (
            "\n\nA imagem do criativo do anuncio esta anexada. "
            "Analise a congruencia visual entre o criativo e o conteudo do artigo."
        )

    # Monta o conteudo: multimodal (lista) ou texto puro (string)
    if imagem:
        conteudo = [prompt, imagem]
    else:
        conteudo = prompt

    try:
        resposta = modelo.generate_content(conteudo)
    except Exception as e:
        raise RuntimeError(f"Erro na API Gemini: {type(e).__name__}: {e}")

    try:
        resultado = json.loads(resposta.text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"JSON invalido retornado pela IA: {e}")

    for campo in SCHEMA_RESPOSTA["required"]:
        if campo not in resultado:
            raise RuntimeError(f"Campo ausente na resposta: '{campo}'")

    return resultado


# ====================================================================
#  Pipeline completo (scraping + analise)
# ====================================================================

def executar_analise(
    modelo,
    url: str,
    texto_anuncio: str,
    imagem: Optional[Image.Image] = None,
) -> ResultadoAnalise:
    resultado = ResultadoAnalise(
        url_artigo=url,
        texto_anuncio=texto_anuncio,
        com_imagem=imagem is not None,
    )

    try:
        texto_artigo = extrair_texto_blog(url)
    except RuntimeError as e:
        resultado.status_erro = f"SCRAPING: {e}"
        return resultado

    try:
        analise = analisar_congruencia(modelo, texto_anuncio, texto_artigo, imagem)
        resultado.score_congruencia = analise["score_congruencia"]
        resultado.promessa_entregue_no_inicio = analise["promessa_entregue_no_inicio"]
        resultado.diagnostico = analise["diagnostico"]
        resultado.sugestao_primeiro_paragrafo = analise["sugestao_primeiro_paragrafo"]
    except RuntimeError as e:
        resultado.status_erro = f"IA: {e}"

    return resultado


# ====================================================================
#  CSS — SaaS Premium, Tema Claro Forcado
# ====================================================================

def injetar_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=DM+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #FFFFFF;
        --bg-secondary: #F8F9FA;
        --bg-tertiary: #F1F3F5;
        --border-light: #E9ECEF;
        --border-medium: #DEE2E6;
        --text-primary: #212529;
        --text-secondary: #495057;
        --text-tertiary: #868E96;
        --text-muted: #ADB5BD;
        --accent: #1A73E8;
        --accent-hover: #1557B0;
        --accent-light: #E8F0FE;
        --score-high: #0B8043;
        --score-high-bg: #E6F4EA;
        --score-high-border: #B7E1CD;
        --score-mid: #B06000;
        --score-mid-bg: #FEF7E0;
        --score-mid-border: #FDD663;
        --score-low: #C5221F;
        --score-low-bg: #FCE8E6;
        --score-low-border: #F5B7B1;
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
        --shadow-card: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    }

    .stApp,
    .stApp > header,
    .main,
    .main .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    .main .block-container {
        max-width: 960px;
        padding-top: 2.5rem;
    }

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-light);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: var(--text-tertiary) !important;
        line-height: 1.6;
    }
    section[data-testid="stSidebar"] label {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown li,
    label, .stTextInput label, .stTextArea label {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-primary) !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Source Serif 4', serif !important;
        color: var(--text-primary) !important;
    }

    .hero-wrapper {
        padding-bottom: 1.75rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-light);
    }
    .hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-family: 'Source Serif 4', serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.015em;
        line-height: 1.15;
        margin: 0 0 0.5rem 0;
    }
    .hero-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 400;
        color: var(--text-tertiary);
        line-height: 1.55;
        max-width: 620px;
    }

    .score-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1.75rem 2rem;
        text-align: center;
        box-shadow: var(--shadow-card);
        position: relative;
    }
    .score-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }
    .score-card.level-high::after { background: var(--score-high); }
    .score-card.level-mid::after  { background: var(--score-mid); }
    .score-card.level-low::after  { background: var(--score-low); }

    .score-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }
    .score-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3.5rem;
        font-weight: 600;
        line-height: 1;
        margin: 0.2rem 0;
    }
    .level-high .score-value { color: var(--score-high); }
    .level-mid  .score-value { color: var(--score-mid); }
    .level-low  .score-value { color: var(--score-low); }

    .score-suffix {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-muted);
    }
    .score-classification {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    .level-high .score-classification { color: var(--score-high); }
    .level-mid  .score-classification { color: var(--score-mid); }
    .level-low  .score-classification { color: var(--score-low); }

    .score-mode-tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 3px;
        margin-top: 0.6rem;
    }
    .mode-textual {
        background: var(--accent-light);
        color: var(--accent);
    }
    .mode-multimodal {
        background: #F3E8FD;
        color: #7C3AED;
    }

    .promise-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.9rem;
        border-radius: 100px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .promise-delivered {
        background: var(--score-high-bg);
        border: 1px solid var(--score-high-border);
        color: var(--score-high);
    }
    .promise-broken {
        background: var(--score-low-bg);
        border: 1px solid var(--score-low-border);
        color: var(--score-low);
    }

    .info-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-sm);
        height: 100%;
    }
    .info-card-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.6rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid var(--border-light);
    }
    .info-card-body {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 400;
        color: var(--text-secondary);
        line-height: 1.65;
    }

    .thumbnail-frame {
        display: inline-block;
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-sm);
        padding: 6px;
        line-height: 0;
    }
    .thumbnail-frame img {
        border-radius: 3px;
        max-height: 140px;
        width: auto;
    }
    .thumbnail-caption {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-top: 0.4rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        padding: 3px;
        border: 1px solid var(--border-light);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        color: var(--text-tertiary);
        border-radius: calc(var(--radius-md) - 2px);
        padding: 0.45rem 1.25rem;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-sm);
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.55rem 1.75rem !important;
        letter-spacing: 0.01em;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: var(--accent-hover) !important;
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.2);
    }

    .stTextInput input, .stTextArea textarea {
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-light) !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }

    [data-testid="stFileUploader"] {
        background: var(--bg-secondary);
        border: 1px dashed var(--border-medium);
        border-radius: var(--radius-md);
        padding: 0.75rem;
    }

    .stDownloadButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        background: var(--bg-primary) !important;
        color: var(--accent) !important;
        border: 1px solid var(--accent) !important;
        border-radius: var(--radius-sm) !important;
    }
    .stDownloadButton > button:hover {
        background: var(--accent-light) !important;
    }

    .stProgress > div > div {
        background: var(--accent) !important;
    }

    .batch-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.65rem 1rem;
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-sm);
        margin-bottom: 0.3rem;
        font-family: 'DM Sans', sans-serif;
    }
    .batch-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .dot-high { background: var(--score-high); }
    .dot-mid  { background: var(--score-mid); }
    .dot-low  { background: var(--score-low); }
    .dot-err  { background: var(--text-muted); }

    .batch-url {
        flex: 1;
        font-size: 0.8rem;
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .batch-score-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.88rem;
        font-weight: 600;
        min-width: 32px;
        text-align: right;
    }
    .val-high { color: var(--score-high); }
    .val-mid  { color: var(--score-mid); }
    .val-low  { color: var(--score-low); }
    .val-err  { color: var(--text-muted); }

    .batch-bar-track {
        width: 90px;
        height: 4px;
        background: var(--bg-tertiary);
        border-radius: 2px;
        overflow: hidden;
    }
    .batch-bar-fill {
        height: 100%;
        border-radius: 2px;
    }

    .metric-row {
        display: flex;
        gap: 0.75rem;
        margin: 0.5rem 0 1.25rem 0;
    }
    .metric-card {
        flex: 1;
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        padding: 0.9rem 1.1rem;
        box-shadow: var(--shadow-sm);
    }
    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1;
    }

    .sidebar-brand {
        font-family: 'Source Serif 4', serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        margin-bottom: 0.15rem;
    }
    .sidebar-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        font-weight: 500;
        color: var(--text-muted) !important;
        letter-spacing: 0.08em;
        background: var(--bg-tertiary);
        padding: 2px 7px;
        border-radius: 3px;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .sidebar-hr {
        height: 1px;
        background: var(--border-light);
        border: none;
        margin: 1.15rem 0;
    }
    .sidebar-section {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-muted) !important;
        margin-bottom: 0.5rem;
    }

    .lock-card {
        text-align: center;
        padding: 2.75rem 2.5rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
    }
    .lock-icon {
        width: 44px;
        height: 44px;
        margin: 0 auto 1.1rem auto;
        background: var(--bg-tertiary);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .lock-icon svg {
        width: 20px;
        height: 20px;
        stroke: var(--text-muted);
        fill: none;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .lock-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
    }
    .lock-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: var(--text-tertiary);
        line-height: 1.55;
    }

    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
        background: var(--bg-secondary) !important;
        border-radius: var(--radius-sm) !important;
    }

    [data-testid="stAlert"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem;
        border-radius: var(--radius-sm);
    }

    [data-testid="stStatusWidget"] {
        font-family: 'DM Sans', sans-serif !important;
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-md) !important;
    }

    [data-testid="stJson"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-sm) !important;
    }

    .stCodeBlock, code, pre {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-light) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ====================================================================
#  Componentes de UI
# ====================================================================

def render_hero():
    st.markdown("""
        <div class="hero-wrapper">
            <div class="hero-eyebrow">Pilar 1 &mdash; Motor de Congruencia</div>
            <h1 class="hero-title">O Retentor</h1>
            <p class="hero-subtitle">
                Identifica desalinhamentos entre a promessa do anuncio (textual e visual)
                e o conteudo do artigo que geram abandono de pagina e perda de receita
                publicitaria.
            </p>
        </div>
    """, unsafe_allow_html=True)


def _score_level(score: int) -> tuple[str, str]:
    if score >= 70:
        return "level-high", "Congruencia forte"
    elif score >= 50:
        return "level-mid", "Congruencia parcial &mdash; risco de abandono"
    else:
        return "level-low", "Congruencia fraca &mdash; alta probabilidade de rejeicao"


def render_score_card(score: int, promessa_entregue: bool, com_imagem: bool = False):
    nivel, classificacao = _score_level(score)

    modo_cls = "mode-multimodal" if com_imagem else "mode-textual"
    modo_txt = "ANALISE MULTIMODAL" if com_imagem else "ANALISE TEXTUAL"

    st.markdown(f"""
        <div class="score-card {nivel}">
            <div class="score-eyebrow">Score de Congruencia</div>
            <div class="score-value">{score}<span class="score-suffix"> / 100</span></div>
            <div class="score-classification">{classificacao}</div>
            <div class="score-mode-tag {modo_cls}">{modo_txt}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    if promessa_entregue:
        st.markdown("""
            <div style="text-align:center;">
                <span class="promise-indicator promise-delivered">
                    Promessa do anuncio entregue nos primeiros paragrafos
                </span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align:center;">
                <span class="promise-indicator promise-broken">
                    Promessa do anuncio nao identificada no inicio do artigo
                </span>
            </div>
        """, unsafe_allow_html=True)


def render_info_card(label: str, body: str):
    st.markdown(f"""
        <div class="info-card">
            <div class="info-card-label">{label}</div>
            <div class="info-card-body">{body}</div>
        </div>
    """, unsafe_allow_html=True)


def render_resultado_completo(resultado: ResultadoAnalise):
    if not resultado.sucesso:
        st.error(f"Erro na analise: {resultado.status_erro}")
        return

    render_score_card(
        resultado.score_congruencia,
        resultado.promessa_entregue_no_inicio,
        resultado.com_imagem,
    )

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        render_info_card("Diagnostico", resultado.diagnostico)
    with col2:
        render_info_card("Sugestao de Reescrita", resultado.sugestao_primeiro_paragrafo)

    with st.expander("Ver dados brutos (JSON)"):
        st.json({
            "score_congruencia": resultado.score_congruencia,
            "promessa_entregue_no_inicio": resultado.promessa_entregue_no_inicio,
            "diagnostico": resultado.diagnostico,
            "sugestao_primeiro_paragrafo": resultado.sugestao_primeiro_paragrafo,
            "modo_analise": "multimodal" if resultado.com_imagem else "textual",
        })


def render_thumbnail(imagem: Image.Image):
    """Exibe miniatura da imagem com moldura elegante."""
    thumb = gerar_thumbnail(imagem)
    buf = BytesIO()
    thumb.save(buf, format="PNG")
    buf.seek(0)

    import base64
    b64 = base64.b64encode(buf.read()).decode()

    st.markdown(f"""
        <div class="thumbnail-frame">
            <img src="data:image/png;base64,{b64}" alt="Criativo do anuncio" />
        </div>
        <div class="thumbnail-caption">Criativo anexado</div>
    """, unsafe_allow_html=True)


def render_batch_row(resultado: ResultadoAnalise):
    url_curta = resultado.url_artigo
    if len(url_curta) > 58:
        url_curta = url_curta[:58] + "..."

    if resultado.sucesso:
        score = resultado.score_congruencia
        if score >= 70:
            dot_cls, val_cls, bar_color = "dot-high", "val-high", "var(--score-high)"
        elif score >= 50:
            dot_cls, val_cls, bar_color = "dot-mid", "val-mid", "var(--score-mid)"
        else:
            dot_cls, val_cls, bar_color = "dot-low", "val-low", "var(--score-low)"
        score_text = str(score)
        pct = score
    else:
        dot_cls, val_cls, bar_color = "dot-err", "val-err", "var(--text-muted)"
        score_text = "ERR"
        pct = 0

    st.markdown(f"""
        <div class="batch-row">
            <div class="batch-dot {dot_cls}"></div>
            <div class="batch-url">{url_curta}</div>
            <div class="batch-bar-track">
                <div class="batch-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
            </div>
            <div class="batch-score-val {val_cls}">{score_text}</div>
        </div>
    """, unsafe_allow_html=True)


def render_batch_summary(resultados: list[ResultadoAnalise]):
    sucessos = [r for r in resultados if r.sucesso]
    scores = [r.score_congruencia for r in sucessos if r.score_congruencia is not None]

    if not scores:
        st.warning("Nenhuma analise concluida com sucesso.")
        return

    media = sum(scores) / len(scores)
    erros = len(resultados) - len(sucessos)

    st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Media</div>
                <div class="metric-value">{media:.0f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Maior Score</div>
                <div class="metric-value">{max(scores)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Menor Score</div>
                <div class="metric-value">{min(scores)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Erros</div>
                <div class="metric-value">{erros}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ====================================================================
#  Sidebar
# ====================================================================

def render_sidebar() -> Optional[str]:
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-brand">O Retentor</div>
            <div class="sidebar-tag">PILAR 1 &middot; v3.0 MULTIMODAL</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Autenticacao</div>', unsafe_allow_html=True)

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Cole sua chave aqui",
            help="Obtenha em aistudio.google.com/apikey",
        )

        if api_key:
            st.success("Chave inserida com sucesso.")
        else:
            st.info("Insira sua API Key para acessar o painel.")

        st.markdown('<div class="sidebar-hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Sobre a Ferramenta</div>', unsafe_allow_html=True)

        st.markdown("""
            O Motor de Congruencia compara a promessa do
            anuncio — textual e visual — com o conteudo
            real do artigo de destino.

            Na versao multimodal, a imagem do criativo do
            anuncio e analisada pela IA para detectar
            quebras de expectativa visual que causam
            abandono de pagina.
        """)

        st.markdown('<div class="sidebar-hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Capacidades</div>', unsafe_allow_html=True)
        st.markdown("""
            - Analise textual (anuncio vs artigo)
            - Analise visual (criativo vs artigo)
            - Processamento em lote via CSV
            - Relatorio exportavel
        """)

        st.markdown('<div class="sidebar-hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Stack Tecnologica</div>', unsafe_allow_html=True)
        st.code("Gemini 2.5 Flash Lite\nBeautifulSoup4\nStreamlit\nPillow\nPandas", language=None)

    return api_key if api_key else None


# ====================================================================
#  Tela de Bloqueio
# ====================================================================

def render_lock_screen():
    st.markdown("<div style='height:4rem'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="lock-card">
                <div class="lock-icon">
                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                    </svg>
                </div>
                <div class="lock-title">Acesso restrito</div>
                <div class="lock-desc">
                    Insira sua Gemini API Key na barra lateral
                    para desbloquear o painel de analise.
                </div>
            </div>
        """, unsafe_allow_html=True)


# ====================================================================
#  Aba 1 — Teste ao Vivo (com upload de imagem)
# ====================================================================

def render_tab_ao_vivo(api_key: str):
    col_url, col_ad = st.columns([3, 2])

    with col_url:
        url = st.text_input(
            "URL do Artigo",
            placeholder="https://seu-blog.com/artigo",
        )

    with col_ad:
        anuncio = st.text_area(
            "Texto do Anuncio",
            placeholder="Cole aqui a copy do anuncio que direciona o leitor ao artigo.",
            height=100,
        )

    # Upload de criativo (opcional)
    col_upload, col_preview = st.columns([2, 1])

    with col_upload:
        arquivo_imagem = st.file_uploader(
            "Faca o upload da imagem do Criativo (Opcional)",
            type=["png", "jpg", "jpeg"],
            help="Formatos aceitos: PNG, JPG, JPEG. A imagem sera analisada pela IA junto com o texto.",
        )

    imagem = None
    with col_preview:
        if arquivo_imagem is not None:
            try:
                imagem = processar_imagem_upload(arquivo_imagem)
                render_thumbnail(imagem)
            except RuntimeError as e:
                st.error(f"Erro ao processar imagem: {e}")
                imagem = None

    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

    if st.button("Gerar Diagnostico", use_container_width=True, type="primary"):
        if not url or not anuncio:
            st.warning("Preencha a URL e o texto do anuncio para continuar.")
            return

        modelo = criar_modelo(api_key)
        modo_label = "multimodal (texto + imagem)" if imagem else "textual"

        with st.status(f"Analisando congruencia ({modo_label})...", expanded=True) as status:
            st.write("Extraindo texto do artigo...")
            try:
                texto_artigo = extrair_texto_blog(url)
                palavras = len(texto_artigo.split())
                st.write(f"Scraping concluido: {palavras} palavras extraidas.")
            except RuntimeError as e:
                status.update(label="Falha no scraping", state="error")
                st.error(f"Erro no scraping: {e}")
                return

            if imagem:
                st.write("Enviando texto e imagem para analise via Gemini...")
            else:
                st.write("Enviando texto para analise via Gemini...")

            try:
                analise = analisar_congruencia(modelo, anuncio, texto_artigo, imagem)
            except RuntimeError as e:
                status.update(label="Falha na analise", state="error")
                st.error(f"Erro na API: {e}")
                return

            status.update(label="Analise concluida", state="complete")

        resultado = ResultadoAnalise(
            url_artigo=url,
            texto_anuncio=anuncio,
            score_congruencia=analise["score_congruencia"],
            promessa_entregue_no_inicio=analise["promessa_entregue_no_inicio"],
            diagnostico=analise["diagnostico"],
            sugestao_primeiro_paragrafo=analise["sugestao_primeiro_paragrafo"],
            com_imagem=imagem is not None,
        )

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
        render_resultado_completo(resultado)


# ====================================================================
#  Aba 2 — Processamento em Lote
# ====================================================================

def render_tab_lote(api_key: str):
    st.markdown("""
        Faca upload do arquivo CSV contendo as colunas
        `url_artigo` e `texto_anuncio`.
    """)

    st.markdown("""
        *Nota: o processamento em lote utiliza analise textual.
        Para analise multimodal com imagem, utilize a aba Teste ao Vivo.*
    """)

    arquivo = st.file_uploader(
        "Selecione o arquivo CSV",
        type=["csv"],
        help="Colunas obrigatorias: url_artigo, texto_anuncio",
    )

    if arquivo is None:
        with st.expander("Formato esperado do CSV"):
            st.code(
                'url_artigo,texto_anuncio\n'
                'https://blog.exemplo.com/artigo-1,"Texto do anuncio 1"\n'
                'https://blog.exemplo.com/artigo-2,"Texto do anuncio 2"',
                language="csv",
            )
        return

    try:
        conteudo = arquivo.getvalue().decode("utf-8")
        df = pd.read_csv(StringIO(conteudo), sep=None, engine="python", dtype=str)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        return

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    colunas_necessarias = {"url_artigo", "texto_anuncio"}
    faltando = colunas_necessarias - set(df.columns)
    if faltando:
        st.error(f"Colunas ausentes: {faltando}. Encontradas: {list(df.columns)}")
        return

    df = df.dropna(subset=["url_artigo", "texto_anuncio"], how="all").reset_index(drop=True)
    total = len(df)

    st.markdown(f"""
        <div class="info-card" style="text-align:center; margin:1rem 0;">
            <div class="info-card-label" style="border:none; text-align:center;">
                Arquivo carregado
            </div>
            <div class="info-card-body">
                <strong>{total}</strong> pares identificados para analise
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button(f"Analisar {total} itens", use_container_width=True, type="primary"):
        modelo = criar_modelo(api_key)
        resultados: list[ResultadoAnalise] = []

        progress_bar = st.progress(0, text="Iniciando processamento...")
        results_container = st.container()

        for idx, row in df.iterrows():
            numero = idx + 1
            url = str(row["url_artigo"]).strip()
            anuncio_texto = str(row["texto_anuncio"]).strip()

            url_curta = url[:50] + "..." if len(url) > 50 else url
            progress_bar.progress(
                numero / total,
                text=f"Analisando {numero} de {total}: {url_curta}",
            )

            resultado = executar_analise(modelo, url, anuncio_texto)
            resultados.append(resultado)

            with results_container:
                render_batch_row(resultado)

            if numero < total:
                time.sleep(1.5)

        progress_bar.progress(1.0, text="Processamento concluido.")

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
        render_batch_summary(resultados)

        for r in resultados:
            if r.sucesso:
                label = f"{r.url_artigo[:60]} | Score: {r.score_congruencia}"
                with st.expander(label):
                    render_resultado_completo(r)
            else:
                label = f"{r.url_artigo[:60]} | Erro"
                with st.expander(label):
                    st.error(r.status_erro)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        registros = [r.to_dict() for r in resultados]
        df_saida = pd.DataFrame(registros, columns=[
            "url_artigo", "texto_anuncio", "score_congruencia",
            "promessa_entregue_no_inicio", "diagnostico",
            "sugestao_primeiro_paragrafo", "status_erro",
        ])

        csv_bytes = df_saida.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        st.download_button(
            label="Baixar Relatorio CSV",
            data=csv_bytes,
            file_name="relatorio_congruencia.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ====================================================================
#  App Principal
# ====================================================================

def main():
    st.set_page_config(
        page_title="O Retentor - Motor de Congruencia",
        page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' rx='3' fill='%231A73E8'/><text x='3' y='12.5' font-size='11' fill='white' font-family='serif' font-weight='bold'>R</text></svg>",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    injetar_css()

    api_key = render_sidebar()

    render_hero()

    if not api_key:
        render_lock_screen()
        return

    tab_vivo, tab_lote = st.tabs(["Teste ao Vivo", "Processamento em Lote"])

    with tab_vivo:
        render_tab_ao_vivo(api_key)

    with tab_lote:
        render_tab_lote(api_key)


if __name__ == "__main__":
    main()
