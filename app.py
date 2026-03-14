"""
O Retentor — Pilar 1: Motor de Congruência · Dashboard
========================================================
Dashboard Streamlit para pitch de vendas.
Analisa congruência entre anúncios e artigos de blog via Gemini.

Executar: streamlit run app.py
"""

import json
import time
import textwrap
from io import StringIO
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
import google.generativeai as genai


# ══════════════════════════════════════════════
#  Constantes do Motor
# ══════════════════════════════════════════════

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

PROMPT_SISTEMA = textwrap.dedent("""\
    Você é um copywriter sênior especialista em retenção de leitores para blogs.
    Sua função é avaliar se o conteúdo de um artigo CUMPRE a promessa feita no
    anúncio que trouxe o leitor até a página.

    Um leitor que clica em um anúncio cria uma expectativa imediata. Se os
    primeiros parágrafos do artigo não entregam essa expectativa, ele abandona
    a página — e a receita cai.

    Você receberá:
    - TEXTO DO ANÚNCIO: a copy que o leitor viu antes de clicar.
    - TEXTO DO ARTIGO: o conteúdo completo (ou parcial) do blog.

    Critérios para o score:
    - 90-100: Congruência perfeita. O artigo entrega exatamente o que o anúncio
              promete, logo nos primeiros parágrafos.
    - 70-89:  Boa congruência. A promessa é cumprida, mas poderia ser mais
              direta no início.
    - 50-69:  Congruência parcial. O tema é o mesmo, mas a promessa específica
              do anúncio demora ou não aparece com clareza.
    - 30-49:  Congruência fraca. O artigo tangencia o tema, mas o leitor
              provavelmente se sentirá enganado.
    - 0-29:   Incongruente. O artigo não tem relação relevante com a promessa.

    Seja direto, prático e acionável na sugestão.
""")

SCHEMA_RESPOSTA = {
    "type": "object",
    "properties": {
        "score_congruencia": {
            "type": "integer",
            "description": "Score de 0 a 100 indicando alinhamento entre anúncio e artigo",
        },
        "diagnostico": {
            "type": "string",
            "description": "Diagnóstico de 2 a 4 frases sobre a congruência encontrada",
        },
        "promessa_entregue_no_inicio": {
            "type": "boolean",
            "description": "True se a promessa do anúncio aparece nos primeiros parágrafos",
        },
        "sugestao_primeiro_paragrafo": {
            "type": "string",
            "description": "Sugestão prática de reescrita do primeiro parágrafo para reter o leitor",
        },
    },
    "required": [
        "score_congruencia",
        "diagnostico",
        "promessa_entregue_no_inicio",
        "sugestao_primeiro_paragrafo",
    ],
}


# ══════════════════════════════════════════════
#  Dataclass de Resultado
# ══════════════════════════════════════════════

@dataclass
class ResultadoAnalise:
    url_artigo: str
    texto_anuncio: str
    score_congruencia: Optional[int] = None
    promessa_entregue_no_inicio: Optional[bool] = None
    diagnostico: Optional[str] = None
    sugestao_primeiro_paragrafo: Optional[str] = None
    status_erro: Optional[str] = None

    @property
    def sucesso(self) -> bool:
        return not self.status_erro

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status_erro"] = d["status_erro"] or ""
        return d


# ══════════════════════════════════════════════
#  Motor de Scraping
# ══════════════════════════════════════════════

def extrair_texto_blog(url: str, timeout: int = 15) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        resposta = requests.get(url, headers=HEADERS_NAVEGADOR, timeout=timeout)
        resposta.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Falha de conexão: servidor inacessível")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout após {timeout}s")
    except requests.exceptions.HTTPError:
        raise RuntimeError(f"HTTP {resposta.status_code}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro de requisição: {type(e).__name__}")

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
        raise RuntimeError("Nenhum conteúdo textual extraído da página")

    texto = conteudo.get_text(separator="\n", strip=True)

    if len(texto.split()) < 50:
        raise RuntimeError(f"Conteúdo muito curto ({len(texto.split())} palavras)")

    palavras = texto.split()
    if len(palavras) > LIMITE_PALAVRAS:
        texto = " ".join(palavras[:LIMITE_PALAVRAS]) + "\n[...texto truncado...]"

    return texto


# ══════════════════════════════════════════════
#  Motor de Análise (Gemini)
# ══════════════════════════════════════════════

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
) -> dict:
    prompt = (
        f"TEXTO DO ANÚNCIO:\n"
        f"---\n{texto_anuncio.strip()}\n---\n\n"
        f"TEXTO DO ARTIGO (primeiros parágrafos são os mais importantes):\n"
        f"---\n{texto_artigo}\n---"
    )

    try:
        resposta = modelo.generate_content(prompt)
    except Exception as e:
        raise RuntimeError(f"Erro na API Gemini: {type(e).__name__}: {e}")

    try:
        resultado = json.loads(resposta.text)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"JSON inválido retornado pela IA: {e}")

    for campo in SCHEMA_RESPOSTA["required"]:
        if campo not in resultado:
            raise RuntimeError(f"Campo ausente na resposta: '{campo}'")

    return resultado


# ══════════════════════════════════════════════
#  Pipeline completo (scraping + análise)
# ══════════════════════════════════════════════

def executar_analise(modelo, url: str, texto_anuncio: str) -> ResultadoAnalise:
    resultado = ResultadoAnalise(url_artigo=url, texto_anuncio=texto_anuncio)

    try:
        texto_artigo = extrair_texto_blog(url)
    except RuntimeError as e:
        resultado.status_erro = f"SCRAPING: {e}"
        return resultado

    try:
        analise = analisar_congruencia(modelo, texto_anuncio, texto_artigo)
        resultado.score_congruencia = analise["score_congruencia"]
        resultado.promessa_entregue_no_inicio = analise["promessa_entregue_no_inicio"]
        resultado.diagnostico = analise["diagnostico"]
        resultado.sugestao_primeiro_paragrafo = analise["sugestao_primeiro_paragrafo"]
    except RuntimeError as e:
        resultado.status_erro = f"IA: {e}"

    return resultado


# ══════════════════════════════════════════════
#  CSS Customizado — tema escuro editorial
# ══════════════════════════════════════════════

def injetar_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;500;600;700&display=swap');

    /* ── Reset e base ────────────────────────── */
    .stApp {
        background: #0a0e17;
        color: #c8cdd5;
    }

    /* ── Sidebar ─────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1520 0%, #0a0e17 100%);
        border-right: 1px solid #1a2236;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8892a4;
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
    }

    /* ── Headers ─────────────────────────────── */
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .hero-accent {
        background: linear-gradient(135deg, #f97316, #fb923c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 300;
        color: #6b7590;
        margin-top: 0.5rem;
        line-height: 1.5;
        max-width: 680px;
    }
    .hero-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #f97316, transparent);
        border: none;
        margin: 1.5rem 0 2rem 0;
        border-radius: 2px;
    }

    /* ── Score card ───────────────────────────── */
    .score-card {
        background: linear-gradient(145deg, #111827 0%, #0d1117 100%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .score-card.score-high::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
    .score-card.score-mid::before  { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .score-card.score-low::before  { background: linear-gradient(90deg, #ef4444, #dc2626); }

    .score-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 4.5rem;
        font-weight: 600;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .score-high .score-number { color: #4ade80; }
    .score-mid  .score-number { color: #fbbf24; }
    .score-low  .score-number { color: #f87171; }

    .score-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #475569;
    }
    .score-verdict {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 0.75rem;
    }
    .score-high .score-verdict { color: #86efac; }
    .score-mid  .score-verdict { color: #fde68a; }
    .score-low  .score-verdict { color: #fca5a5; }

    /* ── Info panels ─────────────────────────── */
    .info-panel {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-panel-header {
        font-family: 'Outfit', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .info-panel-body {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 400;
        color: #cbd5e1;
        line-height: 1.65;
    }

    /* ── Promise badge ───────────────────────── */
    .promise-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1.2rem;
        border-radius: 100px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .promise-yes {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #86efac;
    }
    .promise-no {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.25);
        color: #fca5a5;
    }

    /* ── Tabs ─────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #111827;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        color: #64748b;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #f8fafc !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* ── Buttons ──────────────────────────────── */
    .stButton > button {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #fb923c, #f97316);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(249, 115, 22, 0.3);
    }

    /* ── Inputs ───────────────────────────────── */
    .stTextInput input, .stTextArea textarea {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #f97316 !important;
        box-shadow: 0 0 0 1px #f97316 !important;
    }

    /* ── File uploader ───────────────────────── */
    [data-testid="stFileUploader"] {
        background: #111827;
        border: 2px dashed #1e293b;
        border-radius: 12px;
        padding: 1rem;
    }

    /* ── Progress bar override ───────────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #f97316, #fb923c) !important;
    }

    /* ── Batch table ─────────────────────────── */
    .batch-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.75rem 1rem;
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-family: 'Outfit', sans-serif;
    }
    .batch-url {
        flex: 1;
        font-size: 0.85rem;
        color: #94a3b8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .batch-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        min-width: 48px;
        text-align: right;
    }
    .batch-score.s-high { color: #4ade80; }
    .batch-score.s-mid  { color: #fbbf24; }
    .batch-score.s-low  { color: #f87171; }
    .batch-score.s-err  { color: #64748b; }

    .batch-bar {
        width: 120px;
        height: 6px;
        background: #1e293b;
        border-radius: 3px;
        overflow: hidden;
    }
    .batch-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }

    /* ── Sidebar branding ────────────────────── */
    .sidebar-brand {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .sidebar-version {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #475569;
        letter-spacing: 0.08em;
        background: #1e293b;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 1.5rem;
    }
    .sidebar-divider {
        height: 1px;
        background: #1e293b;
        margin: 1.25rem 0;
    }
    .sidebar-section-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.75rem;
    }

    /* ── Expander ─────────────────────────────── */
    .streamlit-expanderHeader {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
    }

    /* ── Hide default Streamlit chrome ────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Status animation ────────────────────── */
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .processing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #f97316;
        animation: pulse-glow 1.2s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  Componentes de UI
# ══════════════════════════════════════════════

def render_hero():
    st.markdown("""
        <div class="hero-title">
            🛡️ O <span class="hero-accent">Retentor</span>
        </div>
        <p class="hero-subtitle">
            Motor de Congruência — Detecta quando a promessa do anúncio
            não é cumprida pelo artigo, antes que o leitor abandone a página
            e a receita vá embora.
        </p>
        <div class="hero-divider"></div>
    """, unsafe_allow_html=True)


def render_score_card(score: int, promessa_entregue: bool):
    if score >= 70:
        classe = "score-high"
        veredicto = "Congruência forte — o leitor encontra o que foi prometido"
    elif score >= 50:
        classe = "score-mid"
        veredicto = "Congruência parcial — há risco de abandono nos primeiros segundos"
    else:
        classe = "score-low"
        veredicto = "Congruência fraca — o leitor se sentirá enganado e vai sair"

    st.markdown(f"""
        <div class="score-card {classe}">
            <div class="score-label">Score de Congruência</div>
            <div class="score-number">{score}</div>
            <div class="score-verdict">{veredicto}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if promessa_entregue:
        st.markdown("""
            <div class="promise-badge promise-yes">
                ✅ Promessa do anúncio entregue nos primeiros parágrafos
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="promise-badge promise-no">
                ⚠️ Promessa do anúncio NÃO encontrada no início do artigo
            </div>
        """, unsafe_allow_html=True)


def render_info_panel(icon: str, header: str, body: str):
    st.markdown(f"""
        <div class="info-panel">
            <div class="info-panel-header">{icon} {header}</div>
            <div class="info-panel-body">{body}</div>
        </div>
    """, unsafe_allow_html=True)


def render_resultado_completo(resultado: ResultadoAnalise):
    if not resultado.sucesso:
        st.error(f"**Erro na análise:** {resultado.status_erro}")
        return

    render_score_card(resultado.score_congruencia, resultado.promessa_entregue_no_inicio)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        render_info_panel("🔬", "Diagnóstico", resultado.diagnostico)

    with col2:
        render_info_panel("💡", "Sugestão de Reescrita", resultado.sugestao_primeiro_paragrafo)

    with st.expander("📎 Dados da análise (JSON bruto)"):
        st.json({
            "score_congruencia": resultado.score_congruencia,
            "promessa_entregue_no_inicio": resultado.promessa_entregue_no_inicio,
            "diagnostico": resultado.diagnostico,
            "sugestao_primeiro_paragrafo": resultado.sugestao_primeiro_paragrafo,
        })


def render_batch_row(resultado: ResultadoAnalise):
    url_curta = resultado.url_artigo
    if len(url_curta) > 55:
        url_curta = url_curta[:55] + "…"

    if resultado.sucesso:
        score = resultado.score_congruencia
        pct = score
        if score >= 70:
            s_class = "s-high"
            bar_color = "#4ade80"
        elif score >= 50:
            s_class = "s-mid"
            bar_color = "#fbbf24"
        else:
            s_class = "s-low"
            bar_color = "#f87171"
        score_text = str(score)
    else:
        s_class = "s-err"
        bar_color = "#334155"
        pct = 0
        score_text = "ERR"

    st.markdown(f"""
        <div class="batch-row">
            <div class="batch-url">{url_curta}</div>
            <div class="batch-bar">
                <div class="batch-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
            </div>
            <div class="batch-score {s_class}">{score_text}</div>
        </div>
    """, unsafe_allow_html=True)


def render_batch_summary(resultados: list[ResultadoAnalise]):
    sucessos = [r for r in resultados if r.sucesso]
    scores = [r.score_congruencia for r in sucessos if r.score_congruencia is not None]

    if not scores:
        st.warning("Nenhuma análise concluída com sucesso.")
        return

    media = sum(scores) / len(scores)
    erros = len(resultados) - len(sucessos)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Média", f"{media:.0f}")
    col2.metric("Maior Score", f"{max(scores)}")
    col3.metric("Menor Score", f"{min(scores)}")
    col4.metric("Erros", f"{erros}")


# ══════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════

def render_sidebar() -> Optional[str]:
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-brand">🛡️ O Retentor</div>
            <div class="sidebar-version">PILAR 1 · v2.0</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">🔑 Autenticação</div>', unsafe_allow_html=True)

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Cole sua chave aqui…",
            help="Obtenha em aistudio.google.com/apikey",
        )

        if api_key:
            st.success("Chave inserida", icon="✅")
        else:
            st.info("Insira a chave para liberar o painel.", icon="🔒")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">ℹ️ Sobre</div>', unsafe_allow_html=True)

        st.markdown("""
            O **Motor de Congruência** compara a promessa
            do anúncio com o conteúdo real do artigo.

            Se o leitor não encontra o que esperava nos
            primeiros parágrafos, ele abandona — e a
            receita de ads cai.

            Este painel diagnostica o problema e sugere
            a correção antes que o dinheiro vá embora.
        """)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">📐 Stack</div>', unsafe_allow_html=True)
        st.code("Gemini 2.5 Flash Lite\nBeautifulSoup4\nStreamlit\nPandas", language=None)

    return api_key if api_key else None


# ══════════════════════════════════════════════
#  Tela de bloqueio
# ══════════════════════════════════════════════

def render_lock_screen():
    st.markdown("<div style='height:4rem'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align:center; padding:3rem 2rem; background:#111827;
                        border:1px solid #1e293b; border-radius:16px;">
                <div style="font-size:3rem; margin-bottom:1rem;">🔒</div>
                <div style="font-family:'DM Serif Display',serif; font-size:1.5rem;
                            color:#f8fafc; margin-bottom:0.75rem;">
                    Acesso Restrito
                </div>
                <div style="font-family:'Outfit',sans-serif; font-size:0.9rem;
                            color:#64748b; line-height:1.6;">
                    Insira sua <strong style="color:#f97316;">Gemini API Key</strong>
                    na barra lateral para desbloquear o painel de análise.
                </div>
            </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  Aba 1 — Teste ao Vivo
# ══════════════════════════════════════════════

def render_tab_ao_vivo(api_key: str):
    col_url, col_ad = st.columns([3, 2])

    with col_url:
        url = st.text_input(
            "URL do Artigo",
            placeholder="https://seu-blog.com/artigo-aqui",
        )

    with col_ad:
        anuncio = st.text_area(
            "Texto do Anúncio",
            placeholder="Cole aqui a copy do anúncio que leva o leitor ao artigo…",
            height=100,
        )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if st.button("🔍  Gerar Diagnóstico", use_container_width=True, type="primary"):
        if not url or not anuncio:
            st.warning("Preencha a URL e o texto do anúncio para continuar.")
            return

        modelo = criar_modelo(api_key)

        with st.status("Analisando congruência…", expanded=True) as status:
            st.write("⏳ Extraindo texto do artigo…")
            try:
                texto_artigo = extrair_texto_blog(url)
                palavras = len(texto_artigo.split())
                st.write(f"✅ Scraping concluído — {palavras} palavras extraídas")
            except RuntimeError as e:
                status.update(label="Falha no scraping", state="error")
                st.error(f"**Erro no scraping:** {e}")
                return

            st.write("⏳ Enviando para o Gemini…")
            try:
                analise = analisar_congruencia(modelo, anuncio, texto_artigo)
            except RuntimeError as e:
                status.update(label="Falha na análise", state="error")
                st.error(f"**Erro na API:** {e}")
                return

            status.update(label="Análise concluída", state="complete")

        resultado = ResultadoAnalise(
            url_artigo=url,
            texto_anuncio=anuncio,
            score_congruencia=analise["score_congruencia"],
            promessa_entregue_no_inicio=analise["promessa_entregue_no_inicio"],
            diagnostico=analise["diagnostico"],
            sugestao_primeiro_paragrafo=analise["sugestao_primeiro_paragrafo"],
        )

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        render_resultado_completo(resultado)


# ══════════════════════════════════════════════
#  Aba 2 — Processamento em Lote
# ══════════════════════════════════════════════

def render_tab_lote(api_key: str):
    st.markdown("""
        Faça upload do arquivo **`anuncios_input.csv`** com as colunas
        `url_artigo` e `texto_anuncio`.
    """)

    arquivo = st.file_uploader(
        "Arraste o CSV ou clique para selecionar",
        type=["csv"],
        help="Colunas obrigatórias: url_artigo, texto_anuncio",
    )

    if arquivo is None:
        # Mostra template para download
        with st.expander("📋 Formato esperado do CSV"):
            st.code(
                'url_artigo,texto_anuncio\n'
                'https://blog.exemplo.com/artigo-1,"Texto do anúncio 1"\n'
                'https://blog.exemplo.com/artigo-2,"Texto do anúncio 2"',
                language="csv",
            )
        return

    # ── Leitura e validação ──
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
        st.error(f"Colunas ausentes: **{faltando}**. Encontradas: {list(df.columns)}")
        return

    df = df.dropna(subset=["url_artigo", "texto_anuncio"], how="all").reset_index(drop=True)
    total = len(df)

    st.markdown(f"""
        <div class="info-panel" style="text-align:center;">
            <div class="info-panel-header" style="justify-content:center;">
                📄 CSV carregado
            </div>
            <div class="info-panel-body">
                <strong>{total}</strong> pares (anúncio × artigo) prontos para análise
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button(f"🚀  Analisar {total} itens", use_container_width=True, type="primary"):
        modelo = criar_modelo(api_key)
        resultados: list[ResultadoAnalise] = []

        progress_bar = st.progress(0, text="Iniciando…")
        results_container = st.container()

        for idx, row in df.iterrows():
            numero = idx + 1
            url = str(row["url_artigo"]).strip()
            anuncio = str(row["texto_anuncio"]).strip()

            url_curta = url[:50] + "…" if len(url) > 50 else url
            progress_bar.progress(
                numero / total,
                text=f"Analisando {numero}/{total}: {url_curta}",
            )

            resultado = executar_analise(modelo, url, anuncio)
            resultados.append(resultado)

            with results_container:
                render_batch_row(resultado)

            if numero < total:
                time.sleep(1.5)

        progress_bar.progress(1.0, text="✅ Processamento concluído!")

        # ── Resumo ──
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        render_batch_summary(resultados)

        # ── Detalhes expansíveis ──
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        for r in resultados:
            if r.sucesso:
                with st.expander(f"🔍 {r.url_artigo[:60]} — Score: {r.score_congruencia}"):
                    render_resultado_completo(r)
            else:
                with st.expander(f"❌ {r.url_artigo[:60]} — ERRO"):
                    st.error(r.status_erro)

        # ── Download do relatório ──
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        registros = [r.to_dict() for r in resultados]
        df_saida = pd.DataFrame(registros, columns=[
            "url_artigo", "texto_anuncio", "score_congruencia",
            "promessa_entregue_no_inicio", "diagnostico",
            "sugestao_primeiro_paragrafo", "status_erro",
        ])

        csv_bytes = df_saida.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        st.download_button(
            label="📥  Baixar Relatório CSV",
            data=csv_bytes,
            file_name="relatorio_congruencia.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════
#  App Principal
# ══════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="O Retentor — Motor de Congruência",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    injetar_css()

    api_key = render_sidebar()

    render_hero()

    if not api_key:
        render_lock_screen()
        return

    tab_vivo, tab_lote = st.tabs(["⚡ Teste ao Vivo", "📦 Processamento em Lote"])

    with tab_vivo:
        render_tab_ao_vivo(api_key)

    with tab_lote:
        render_tab_lote(api_key)


if __name__ == "__main__":
    main()
