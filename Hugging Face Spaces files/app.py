"""
app.py — Football RAG · Gradio UI for Hugging Face Spaces
Compatible with Gradio 6.x

Required Secrets (HF Space → Settings → Variables and secrets):
  PINECONE_API_KEY
  GROQ_API_KEY
"""

import gradio as gr
from retriever import query as rag_query

# ── Custom CSS — dark stadium aesthetic ───────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root {
    --green:   #00c853;
    --green-d: #009624;
    --pitch:   #0d1f12;
    --card:    #111d14;
    --border:  #1e3a22;
    --muted:   #4a7a52;
    --text:    #e8f5e9;
    --sub:     #8fbc8f;
    --warn:    #ffd740;
}

/* ── Global ── */
body, .gradio-container {
    background: var(--pitch) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }

/* ── Header ── */
#header {
    text-align: center;
    padding: 2.2rem 1rem 1.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
    background: linear-gradient(180deg, #0a1a0c 0%, var(--pitch) 100%);
}
#header h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 0.06em;
    color: var(--green);
    margin: 0;
    line-height: 1;
}
#header p { color: var(--sub); font-size: 0.92rem; margin: 0.4rem 0 0; }

/* ── Textboxes ── */
label > span {
    color: var(--sub) !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 600;
}
textarea, input[type="text"] {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    transition: border-color 0.2s;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--green) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(0,200,83,0.12) !important;
}

/* ── Answer box — green left border ── */
#answer-box textarea {
    border-left: 4px solid var(--green) !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
    min-height: 72px;
}

/* ── Stale warning ── */
#stale-box textarea {
    background: #1a1400 !important;
    border: 1px solid var(--warn) !important;
    color: var(--warn) !important;
    font-size: 0.85rem !important;
}
#stale-box label > span { color: var(--warn) !important; }

/* ── Latency ── */
#latency-box textarea {
    background: #0c1a0e !important;
    border-color: var(--border) !important;
    color: var(--sub) !important;
    font-size: 0.82rem !important;
}

/* ── Submit button ── */
button.primary {
    background: var(--green) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #0a1a0c !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.35rem !important;
    letter-spacing: 0.08em;
    height: 56px !important;
    transition: background 0.15s, transform 0.1s;
}
button.primary:hover {
    background: #00e676 !important;
    transform: translateY(-1px);
}
button.primary:active { transform: translateY(0); }

/* ── Accordion ── */
.accordion {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.accordion .label-wrap span { color: var(--sub) !important; font-size: 0.88rem !important; }

/* ── Markdown ── */
.prose, .prose * { color: var(--text) !important; }
.prose h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.05rem;
    letter-spacing: 0.06em;
    color: var(--green) !important;
    margin: 0 0 0.5rem;
}
.prose strong { color: var(--green) !important; }
.prose em { color: var(--sub) !important; font-style: italic; }
.prose code {
    background: #0a1a0c !important;
    color: var(--green) !important;
    padding: 0.1em 0.4em;
    border-radius: 4px;
    font-size: 0.87em;
}
.prose hr { border-color: var(--border) !important; margin: 1.2rem 0; }
.prose a { color: var(--green) !important; }

/* ── Examples table ── */
.examples table { background: var(--card) !important; border-radius: 8px; border: 1px solid var(--border) !important; }
.examples td { color: var(--sub) !important; font-size: 0.9rem !important; padding: 0.4rem 0.8rem !important; }
.examples tr:hover td { color: var(--text) !important; background: #162a19 !important; cursor: pointer; }
.examples th { display: none; }

/* ── Score panels ── */
.score-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
"""

# ── Formatting helpers ────────────────────────────────────────────────────────

def _bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_chunks(chunks: list) -> str:
    if not chunks:
        return "_No chunks retrieved._"
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "Unknown")
        url    = c.get("url", "")
        score  = c.get("rerank_score", c.get("score", 0.0))
        date   = c.get("fetched_date", "")
        text   = c.get("text", "").strip()

        header = f"**[{i}] {source}**"
        if score:
            header += f"  `score: {score:.4f}`"
        if date:
            header += f"  ·  fetched {date}"
        if url:
            header += f"\n🔗 {url}"
        parts.append(f"{header}\n\n{text}")
    return "\n\n---\n\n".join(parts)


def _fmt_faithfulness(details: list, score: float) -> str:
    pct = f"{score:.0%}"
    n   = len(details)
    sup = sum(1 for d in details if d["label"] == "supported")
    lines = [
        "### Faithfulness",
        f"**{pct}** `{_bar(score)}` · {sup}/{n} claims supported\n",
    ]
    for i, d in enumerate(details, 1):
        icon = "✅" if d["label"] == "supported" else "❌"
        lines.append(f"{icon} **Claim {i}:** {d['claim']}")
        if d.get("rationale"):
            lines.append(f"   ↳ _{d['rationale']}_")
        if d.get("evidence"):
            lines.append(f"   📌 `{d['evidence']}`")
    return "\n".join(lines)


def _fmt_relevancy(details: list, score: float) -> str:
    pct = f"{score:.0%}"
    lines = [
        "### Relevancy",
        f"**{pct}** `{_bar(score)}` · mean cosine similarity\n",
    ]
    for i, (q, sim) in enumerate(details, 1):
        lines.append(f"**Q{i}:** {q}")
        lines.append(f"   similarity: `{sim:.4f}`")
    return "\n".join(lines)


# ── Main handler ──────────────────────────────────────────────────────────────

def run_query(question: str):
    question = (question or "").strip()
    if not question:
        return (
            "Please enter a question.",
            "_No chunks retrieved._",
            "### Faithfulness\n—",
            "### Relevancy\n—",
            "",
            "",
        )

    try:
        result = rag_query(question)
    except EnvironmentError as e:
        return (f"⚠️ Configuration error: {e}", "", "—", "—", "", "")
    except Exception as e:
        return (f"⚠️ Error: {e}", "", "—", "—", "", "")

    answer    = result["answer"]
    chunks_md = _fmt_chunks(result["context_chunks"])
    faith_md  = _fmt_faithfulness(result["faithfulness_details"], result["faithfulness"])
    rel_md    = _fmt_relevancy(result["relevancy_details"],    result["relevancy"])
    latency   = (
        f"⏱  Retrieval {result['retrieval_time_sec']}s  ·  "
        f"Generation {result['generation_time_sec']}s  ·  "
        f"Total {result['total_time_sec']}s"
    )
    stale = (
        "⚠️  This query asks about recent events. The corpus is built from Wikipedia "
        "snapshots and may not reflect the very latest results."
        if result.get("stale_warning") else ""
    )

    return answer, chunks_md, faith_md, rel_md, latency, stale


# ── Examples ──────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["Who won the 2022 FIFA World Cup?"],
    ["How many Ballon d'Or awards has Lionel Messi won?"],
    ["What is the offside rule in football?"],
    ["Which club has won the most UEFA Champions League titles?"],
    ["Who are the top scorers in Premier League history?"],
    ["What is VAR and how does it work?"],
    ["How did Cristiano Ronaldo start his professional career?"],
    ["Which national team has won the most FIFA World Cups?"],
]

# ── Build UI ──────────────────────────────────────────────────────────────────
# NOTE: In Gradio 6, theme must NOT be passed to gr.Blocks().
#       It belongs in demo.launch() — but since we fully override
#       colors via CSS, no theme argument is needed at all.

with gr.Blocks(css=CUSTOM_CSS, title="Football RAG · Q&A") as demo:

    gr.HTML("""
    <div id="header">
      <h1>⚽ Football RAG</h1>
      <p>Retrieval-Augmented Generation &nbsp;·&nbsp; Wikipedia Football Corpus &nbsp;·&nbsp; Powered by Groq LLaMA-3.1</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            question_box = gr.Textbox(
                label="Ask anything about football",
                placeholder="e.g. Who scored the winning goal in the 2022 World Cup final?",
                lines=2,
            )
        with gr.Column(scale=1, min_width=130):
            submit_btn = gr.Button("SEARCH", variant="primary")

    stale_box = gr.Textbox(
        value="", label="Data freshness warning",
        interactive=False, lines=1, visible=True,
        elem_id="stale-box",
    )

    answer_box = gr.Textbox(
        label="Answer", interactive=False, lines=3,
        elem_id="answer-box",
    )

    with gr.Row():
        with gr.Column(elem_classes=["score-panel"]):
            faith_box = gr.Markdown("### Faithfulness\n—")
        with gr.Column(elem_classes=["score-panel"]):
            rel_box = gr.Markdown("### Relevancy\n—")

    latency_box = gr.Textbox(
        label="Latency", interactive=False, lines=1,
        elem_id="latency-box",
    )

    with gr.Accordion("📄  Retrieved context chunks", open=False):
        chunks_box = gr.Markdown("_Run a query to see retrieved passages._")

    gr.Examples(
        examples=EXAMPLES,
        inputs=question_box,
        label="Example questions — click to load",
    )

    gr.Markdown("""
---
**Pipeline:** Recursive chunking · Dense retrieval (Pinecone `sports-rag-v2`) · CrossEncoder rerank (`BAAI/bge-reranker-base`) · MMR deduplication (λ=0.6) · Groq LLaMA-3.1-8B

**Evaluation:** LLM-as-Judge — *Faithfulness* verifies each extracted claim against retrieved context · *Relevancy* measures cosine similarity between original query and 3 generated alternate questions
""")

    outputs = [answer_box, chunks_box, faith_box, rel_box, latency_box, stale_box]
    submit_btn.click(fn=run_query, inputs=question_box, outputs=outputs)
    question_box.submit(fn=run_query, inputs=question_box, outputs=outputs)


if __name__ == "__main__":
    demo.launch()
