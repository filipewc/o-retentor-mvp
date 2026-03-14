# 🔍 O Retentor — Pilar 1: Motor de Congruência (v2 · Batch)

Ferramenta de processamento em lote que analisa se artigos de blog **cumprem a promessa** dos anúncios que levam leitores até eles, usando IA generativa (Gemini) com saída JSON nativa.

## O que mudou da v1 para a v2

| Aspecto | v1 (PoC) | v2 (Batch) |
|---------|----------|------------|
| Entrada | Variáveis hardcoded | CSV com N pares |
| Parse JSON | Limpeza manual de markdown | `response_mime_type` nativo + `response_schema` |
| Resiliência | Script quebrava no primeiro erro | Erros por linha registrados, loop nunca para |
| Saída | Print no terminal | CSV de relatório + resumo no terminal |
| Modelo | Criado a cada chamada | Reutilizado no lote inteiro |

## Arquitetura

```
┌─────────────────────┐
│  anuncios_input.csv  │   N linhas: (url_artigo, texto_anuncio)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│            LOOP RESILIENTE (linha a linha)       │
│                                                  │
│  ┌─────────────┐   erro?   ┌──────────────────┐ │
│  │  Scraping    │──────────▶│ Registra em      │ │
│  │  BS4+Requests│   não     │ status_erro      │ │
│  └──────┬──────┘   ┌───────│ e pula p/ próxima│ │
│         │          │       └──────────────────┘ │
│         ▼          │                             │
│  ┌──────────────┐  │                             │
│  │  Gemini API  │──┘ erro?  (mesmo tratamento)   │
│  │  JSON nativo │                                │
│  └──────────────┘                                │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌──────────────────────────┐
│ relatorio_congruencia.csv │   + resumo visual no terminal
└──────────────────────────┘
```

## Pré-requisitos

- Python 3.10+
- Chave de API do Google Gemini ([obter aqui](https://aistudio.google.com/apikey))

## Instalação

```bash
cd retentor_pilar1

python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt

cp .env.example .env
# Edite .env e cole sua GEMINI_API_KEY
```

## Uso

### 1. Prepare o CSV de entrada

Crie (ou edite) o arquivo `anuncios_input.csv`:

```csv
url_artigo,texto_anuncio
https://meu-blog.com/artigo-1,"Texto do anúncio que levou ao artigo 1"
https://meu-blog.com/artigo-2,"Texto do anúncio que levou ao artigo 2"
```

Um CSV de exemplo com 5 linhas (incluindo uma URL inválida para testar resiliência) já está incluso.

### 2. Execute

```bash
python motor_congruencia.py
```

### 3. Saída esperada no terminal

```
⏳  Carregando configurações...
⏳  Lendo arquivo de entrada: anuncios_input.csv
✅  5 pares (anúncio × artigo) carregados

════════════════════════════════════════════════════════════
  🚀  Iniciando processamento em lote: 5 itens
════════════════════════════════════════════════════════════

  [  1/5] https://blog.hubspot.com/marketing/how-to-start-a-bl...
          ✅ Scraping OK (3842 palavras)
          🟢 Score: 78/100
  [  2/5] https://neilpatel.com/br/blog/seo-o-guia-definitivo/
          ✅ Scraping OK (5000 palavras)
          🟡 Score: 62/100
  ...
  [  5/5] https://url-invalida-para-teste.xyz/artigo-que-nao-e...
          ❌ Scraping falhou → Falha de conexão: servidor inacessível

──────────────────────────────────────────────────────────
  📊  Resumo: 4 sucesso · 1 erros · 5 total
──────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════
  🔍  O RETENTOR — Resumo dos Scores
════════════════════════════════════════════════════════════

  🟢 [████████████████░░░░] 78  ✅  https://blog.hubspot.com/marketing/h...
  🟡 [████████████░░░░░░░░] 62  ❌  https://neilpatel.com/br/blog/seo-o-...
  ...

──────────────────────────────────────────────────────────
  Média: 71  |  Menor: 55  |  Maior: 82
──────────────────────────────────────────────────────────

  📁  Relatório exportado: relatorio_congruencia.csv
      → 5 linhas | 7 colunas
```

### 4. Relatório CSV de saída

O arquivo `relatorio_congruencia.csv` terá estas colunas:

| Coluna | Descrição |
|--------|-----------|
| `url_artigo` | URL original |
| `texto_anuncio` | Texto do anúncio original |
| `score_congruencia` | Score 0-100 |
| `promessa_entregue_no_inicio` | True/False |
| `diagnostico` | Análise da IA |
| `sugestao_primeiro_paragrafo` | Sugestão de reescrita |
| `status_erro` | Vazio se OK, ou descrição do erro |

## Estrutura de Arquivos

```
retentor_pilar1/
├── motor_congruencia.py        # Script principal (v2 batch)
├── anuncios_input.csv          # CSV de entrada (exemplo incluso)
├── relatorio_congruencia.csv   # CSV de saída (gerado após execução)
├── requirements.txt            # Dependências Python
├── .env.example                # Template de variáveis de ambiente
└── README.md                   # Este arquivo
```

## Próximos Passos

| Pilar | Nome | Status |
|-------|------|--------|
| 1 | Motor de Congruência | ✅ v2 Batch |
| 2 | Radar de Ponto de Fuga (GA4) | 🔜 |
| 3 | Motor de Recomendação Dinâmica | 🔜 |
| 4 | Auditoria de Links Internos (NetworkX) | 🔜 |
