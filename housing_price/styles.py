"""Streamlit visual system based on the project presentation palette."""

APP_CSS = """
<style>
:root {
    --cream: #F7F2EA;
    --paper: #FFFDF8;
    --charcoal: #263438;
    --slate: #425155;
    --muted: #697477;
    --coral: #D45B55;
    --coral-dark: #B74642;
    --gold: #D6A55C;
    --line: #DED7CC;
}

html, body, [class*="css"] {
    font-family: Inter, Aptos, "Segoe UI", sans-serif;
    color: var(--charcoal);
}

[data-testid="stAppViewContainer"] {
    background: var(--cream);
}

[data-testid="stHeader"] {
    background: rgba(247, 242, 234, 0.86);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: var(--charcoal);
    border-right: 0;
}

[data-testid="stSidebar"] * {
    color: #F8F3EB;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] [data-checked="true"] {
    background: rgba(212, 91, 85, 0.22);
}

.sidebar-brand {
    padding: 0.55rem 0.15rem 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.14);
    margin-bottom: 1rem;
}

.sidebar-kicker, .eyebrow {
    margin: 0 0 0.45rem;
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.sidebar-kicker { color: #F1A09A !important; }
.eyebrow { color: var(--coral); }

.sidebar-title {
    margin: 0;
    color: #FFFDF8 !important;
    font-size: 1.18rem;
    line-height: 1.25;
    font-weight: 730;
}

.sidebar-note {
    margin-top: 1.7rem;
    padding: 0.9rem;
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    font-size: 0.78rem;
    line-height: 1.5;
    color: #DDE1DF !important;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 250px;
    padding: 2.35rem 2.6rem 2.2rem;
    margin-bottom: 1.4rem;
    border-radius: 24px;
    background: var(--paper);
    border: 1px solid var(--line);
    box-shadow: 0 16px 40px rgba(38, 52, 56, 0.08);
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 710px;
}

.hero h1 {
    margin: 0 0 0.75rem;
    max-width: 670px;
    color: var(--charcoal);
    font-size: clamp(2rem, 4.6vw, 3.6rem);
    line-height: 0.99;
    letter-spacing: -0.045em;
    font-weight: 760;
}

.hero-copy {
    max-width: 650px;
    margin: 0;
    color: var(--slate);
    font-size: 1.04rem;
    line-height: 1.62;
}

.hero-wave {
    position: absolute;
    right: -80px;
    bottom: -34px;
    width: 520px;
    height: 190px;
    opacity: 0.52;
    z-index: 1;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 1rem 0 1.6rem;
}

.metric-card {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 1rem 1.05rem;
}

.metric-value {
    margin: 0;
    color: var(--charcoal);
    font-size: 1.55rem;
    line-height: 1;
    font-weight: 760;
    letter-spacing: -0.025em;
}

.metric-label {
    margin: 0.45rem 0 0;
    color: var(--muted);
    font-size: 0.77rem;
    line-height: 1.35;
}

.notice {
    margin: 0.8rem 0 1.25rem;
    padding: 0.9rem 1rem;
    background: #F1E5D9;
    border-left: 4px solid var(--gold);
    border-radius: 0 12px 12px 0;
    color: var(--slate);
    font-size: 0.88rem;
    line-height: 1.5;
}

.section-heading {
    margin: 1.7rem 0 0.3rem;
    color: var(--charcoal);
    font-size: 1.5rem;
    line-height: 1.2;
    letter-spacing: -0.02em;
}

.section-copy {
    margin: 0 0 1rem;
    color: var(--muted);
    line-height: 1.55;
}

div[data-testid="stForm"] {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.1rem 1.25rem 1.3rem;
    box-shadow: 0 10px 25px rgba(38, 52, 56, 0.05);
}

div[data-testid="stForm"] h3 {
    margin-top: 0.25rem;
    color: var(--charcoal);
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #FFFFFF;
    border-color: #CFC7BA;
}

.stButton > button, [data-testid="stFormSubmitButton"] > button {
    min-height: 2.9rem;
    border: 0;
    border-radius: 10px;
    background: var(--coral);
    color: white;
    font-weight: 720;
    box-shadow: none;
}

.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: var(--coral-dark);
    color: white;
    border: 0;
}

.prediction-card {
    position: relative;
    overflow: hidden;
    margin-top: 1.35rem;
    padding: 1.6rem 1.75rem;
    background: var(--charcoal);
    border-radius: 18px;
    color: white;
}

.prediction-label {
    margin: 0 0 0.45rem;
    color: #F1A09A;
    font-size: 0.74rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.prediction-value {
    margin: 0;
    color: #FFFDF8;
    font-size: clamp(2.35rem, 7vw, 4.25rem);
    line-height: 1;
    font-weight: 770;
    letter-spacing: -0.045em;
}

.prediction-caption {
    max-width: 680px;
    margin: 0.8rem 0 0;
    color: #DDE1DF;
    font-size: 0.86rem;
    line-height: 1.5;
}

.pipeline {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.65rem;
    margin: 1rem 0 1.5rem;
}

.pipeline-step {
    min-height: 104px;
    padding: 0.9rem;
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 13px;
}

.pipeline-number {
    color: var(--coral);
    font-size: 0.72rem;
    font-weight: 760;
    letter-spacing: 0.1em;
}

.pipeline-title {
    display: block;
    margin-top: 0.32rem;
    color: var(--charcoal);
    font-size: 0.9rem;
    line-height: 1.35;
    font-weight: 690;
}

.small-source {
    color: var(--muted);
    font-size: 0.76rem;
    line-height: 1.5;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

@media (max-width: 860px) {
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .pipeline { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .hero { padding: 1.7rem 1.35rem; min-height: 230px; }
    .hero-wave { opacity: 0.3; }
}

@media (max-width: 520px) {
    [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
    .metric-grid { grid-template-columns: 1fr 1fr; gap: 0.55rem; }
    .metric-card { padding: 0.8rem; }
    .metric-value { font-size: 1.28rem; }
    .pipeline { grid-template-columns: 1fr; }
    .hero { border-radius: 17px; }
    .prediction-card { padding: 1.3rem 1.2rem; }
}
</style>
"""


def hero(title: str, description: str, eyebrow: str) -> str:
    return f"""
    <section class="hero">
      <div class="hero-content">
        <p class="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p class="hero-copy">{description}</p>
      </div>
      <svg class="hero-wave" viewBox="0 0 520 190" aria-hidden="true">
        <path d="M-20 145 C 80 20, 155 215, 270 80 S 430 45, 560 125" fill="none" stroke="#D45B55" stroke-width="7"/>
        <path d="M-30 170 C 95 55, 170 228, 285 105 S 440 70, 570 150" fill="none" stroke="#D6A55C" stroke-width="4"/>
        <path d="M20 188 C 130 90, 210 235, 320 130 S 475 105, 590 175" fill="none" stroke="#263438" stroke-width="3"/>
      </svg>
    </section>
    """


def metric_grid(items: list[tuple[str, str]]) -> str:
    cards = "".join(
        f'<div class="metric-card"><p class="metric-value">{value}</p>'
        f'<p class="metric-label">{label}</p></div>'
        for value, label in items
    )
    return f'<div class="metric-grid">{cards}</div>'
