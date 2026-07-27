"""
TLDR BOT — Landing Page (Home.py)
Dark background matching the chat page.
Button sits directly under the tagline — no scroll needed.
Demo screenshot embedded below the hero as a preview card.
"""

import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="TLDR BOT",
    page_icon="⚡",
    layout="centered",
)

st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }

    .stApp { background: #0a0a0a; }

    .block-container {
        padding: 0 !important;
        max-width: 860px !important;
        margin: 0 auto !important;
    }

    /* Hero — just enough height to center the text block, no excess */
    .hero {
        min-height: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 20vh 2rem 0 2rem;
    }

    /* Big white headline */
    .title {
        font-size: clamp(3rem, 8vw, 5.5rem);
        font-weight: 900;
        letter-spacing: -0.035em;
        line-height: 1.05;
        color: #ffffff;
        margin: 0 0 1.25rem 0;
    }

    /* Readable grey tagline on dark bg */
    .tagline {
        font-size: clamp(1rem, 2vw, 1.1rem);
        color: #999999;
        line-height: 1.75;
        max-width: 420px;
        margin: 0 auto 1.75rem auto;
        font-weight: 400;
    }

    /* Format pills */
    .formats {
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-bottom: 1.25rem;
    }
    .fmt-pill {
        font-size: 0.72rem;
        font-weight: 700;
        color: #666;
        border: 1px solid #222;
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: #111;
    }

    /* Button — white pill on dark bg, sits right under the hero content */
    [data-testid="stButton"] {
        display: flex;
        justify-content: center;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    [data-testid="stButton"] > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 999px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 2.5rem !important;
        transition: opacity 0.15s ease;
    }
    [data-testid="stButton"] > button p {
        color: #000000 !important;
    }
    [data-testid="stButton"] > button:hover {
        opacity: 0.8 !important;
    }

    /* Demo preview card */
    .preview-wrap {
        padding: 0 1.5rem 4rem 1.5rem;
    }
    .preview-label {
        text-align: center;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #444;
        margin-bottom: 1rem;
    }
    .preview-card {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #1e1e1e;
        box-shadow: 0 0 60px rgba(0,0,0,0.6);
    }
    .preview-card img {
        width: 100%;
        display: block;
    }

    css
    /* Hide sidebar navigation on landing page before login */
    [data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Hero — headline, tagline, format pills
# --------------------------------------------------------------------------

st.markdown("""
<div class="hero">
    <h1 class="title">Your lectures,<br>actually understood.</h1>
    <p class="tagline">
        Drop a PDF or screenshot. Get each concept broken down
        with explanations and examples — then ask anything about it.
    </p>
    <div class="formats">
        <span class="fmt-pill">PDF</span>
        <span class="fmt-pill">PNG</span>
        <span class="fmt-pill">JPG</span>
        <span class="fmt-pill">JPEG</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# CTA button — real Streamlit widget, centered via CSS above
# Sits directly under the hero HTML block with no gap
# --------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Open Chat →", use_container_width=True):
        st.switch_page("pages/1_Chat.py")

# --------------------------------------------------------------------------
# Demo preview — your chat screenshot embedded as an image
# Replace the path below with your actual screenshot file once you have one.
# For now it shows a placeholder message if the file isn't found.
# --------------------------------------------------------------------------

# To add your screenshot:
# 1. Save your chat screenshot as demo.png in the same folder as Home.py
# 2. The card below will automatically display it

demo_path = Path("demo.png")

st.markdown('<div class="preview-wrap">', unsafe_allow_html=True)
st.markdown('<p class="preview-label">See it in action</p>', unsafe_allow_html=True)

if demo_path.exists():
    # Read and base64-encode the image so it embeds directly in HTML
    img_b64 = base64.b64encode(demo_path.read_bytes()).decode()
    st.markdown(f"""
    <div class="preview-card">
        <img src="data:image/png;base64,{img_b64}" alt="TLDR BOT in action" />
    </div>
    """, unsafe_allow_html=True)
else:
    # Placeholder shown until you drop in demo.png
    st.markdown("""
    <div class="preview-card" style="background:#111; padding: 4rem 2rem; text-align:center;">
        <p style="color:#444; font-size:0.9rem;">
            Drop a file called <code style="color:#666">demo.png</code> next to Home.py<br>
            to show your app preview here.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)