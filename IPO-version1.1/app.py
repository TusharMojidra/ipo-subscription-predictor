"""
IPO Subscription & Listing Gain Predictor - Streamlit dashboard.

Run it with:   streamlit run app.py
"""

import os

import altair as alt
import pandas as pd
import streamlit as st

import importlib.util
# shap is optional AND slow to import (~5s), so we only check that it is
# installed here and import it lazily when the explainer is actually opened.
SHAP_OK = importlib.util.find_spec("shap") is not None

try:
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:  # plotly is optional - falls back to static charts
    PLOTLY_OK = False

from predict_ipo import (CLASS_FEATURES, CLASS_ORDER, GAIN_FEATURES,
                         engineer_input, load_training_data, train_models)
from src.feature_engineering import add_log_features

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IPO Predictor | Subscription & Listing Gain",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme presets — each restyles the whole app via CSS variables (see below)
# ---------------------------------------------------------------------------
THEMES = {
    "Charcoal (dark)": {
        "bg": "#18181C", "bg2": "#17171B",
        "bg_rad1": "rgba(168,85,247,0.10)", "bg_rad2": "rgba(236,72,153,0.08)",
        "panel": "#1E1E24", "sidebar": "#151519",
        "border": "#2A2A32", "border_soft": "#232329",
        "text": "#FFFFFF", "text1": "#E4E4E7", "text2": "#D4D4D8",
        "dim": "#A1A1AA", "faint": "#71717A",
        "accent": "#A855F7", "accent2": "#EC4899", "lav": "#C084FC",
        "green": "#22C55E", "red": "#EF4444",
        "blue1": "#3B82F6", "blue2": "#2563EB", "blue3": "#4F8DFA",
        "pt": "plotly_dark", "cbg": "#1E1E24", "cf": "#A1A1AA", "cg": "#2A2A32", "ct": "#E4E4E7",
    },
    "Midnight (blue)": {
        "bg": "#0B1120", "bg2": "#0F172A",
        "bg_rad1": "rgba(56,189,248,0.10)", "bg_rad2": "rgba(129,140,248,0.08)",
        "panel": "#111C33", "sidebar": "#0A1020",
        "border": "#1E2A47", "border_soft": "#26345A",
        "text": "#F1F5F9", "text1": "#E2E8F0", "text2": "#CBD5E1",
        "dim": "#94A3B8", "faint": "#64748B",
        "accent": "#38BDF8", "accent2": "#818CF8", "lav": "#A5B4FC",
        "green": "#34D399", "red": "#F87171",
        "blue1": "#60A5FA", "blue2": "#3B82F6", "blue3": "#93C5FD",
        "pt": "plotly_dark", "cbg": "#111C33", "cf": "#94A3B8", "cg": "#1E2A47", "ct": "#E2E8F0",
    },
    "Emerald (dark)": {
        "bg": "#0E1A14", "bg2": "#0A1510",
        "bg_rad1": "rgba(52,211,153,0.10)", "bg_rad2": "rgba(45,212,191,0.08)",
        "panel": "#12211A", "sidebar": "#0B1511",
        "border": "#1F3A2E", "border_soft": "#183028",
        "text": "#F0FDF4", "text1": "#D1FAE5", "text2": "#A7F3D0",
        "dim": "#6EE7B7", "faint": "#5A8F77",
        "accent": "#34D399", "accent2": "#2DD4BF", "lav": "#5EEAD4",
        "green": "#34D399", "red": "#F87171",
        "blue1": "#34D399", "blue2": "#0D9488", "blue3": "#5EEAD4",
        "pt": "plotly_dark", "cbg": "#12211A", "cf": "#6EE7B7", "cg": "#1F3A2E", "ct": "#D1FAE5",
    },
}

# NOTE: `st.sidebar.radio` / `st.sidebar.selectbox` render NOTHING in this
# Streamlit version, so the pickers live inside the `with st.sidebar:` block
# below. Here we read their values from session state (default on first run)
# so the theme/font CSS can still be applied before the widgets render.
theme = st.session_state.get("theme_picker", "Charcoal (dark)")
CH = THEMES[theme]

# ---------------------------------------------------------------------------
# Font presets — swap the whole app's typeface (heading + body stacks)
# ---------------------------------------------------------------------------
FONTS = {
    "Modern Sans": {
        "head": "'Space Grotesk', sans-serif",
        "body": "'Inter', sans-serif",
    },
    "Elegant Serif": {
        "head": "'Playfair Display', serif",
        "body": "'Lora', serif",
    },
    "Tech Mono": {
        "head": "'Orbitron', sans-serif",
        "body": "'JetBrains Mono', monospace",
    },
    "Data Terminal": {  # fintech dashboard pairing (ui-ux-pro-max skill)
        "head": "'Fira Sans', sans-serif",
        "body": "'Fira Code', monospace",
    },
}
font_style = st.session_state.get("font_picker", "Modern Sans")
F = FONTS[font_style]

# ---------------------------------------------------------------------------
# Navbar behaviour: click a nav link -> click the matching native tab button;
# keep the active pill in sync; count-up the hero stat chips on first load.
# Runs once (guarded), survives Streamlit reruns via delegation + interval.
# ---------------------------------------------------------------------------
NAV_JS = """
<script>
(function () {
  if (window.__ipoNavInit) return;
  window.__ipoNavInit = true;
  // The native st.tabs widget (keyed "active_tab", on_change="rerun") is the
  // single source of truth. Nav links click ONLY the matching native tab — ONE
  // click = ONE rerun, so there is never a second racing rerun (the old radio+
  // tab double-click caused the tab bar/hero to double and snap back).
  document.addEventListener('click', function (e) {
    var link = e.target.closest('[data-nav-link]');
    if (!link) return;
    e.preventDefault();
    var name = link.getAttribute('data-nav-link');
    // Crossfade without double-content overlap: while the rerun streams in
    // the new section, Streamlit keeps the OLD panel mounted (it only removes
    // it once the new content has arrived) — so both panels are briefly
    // visible. A body-level class hides ALL panels during the switch (it
    // survives the rerun, unlike per-panel inline styles), then reveals the
    // single remaining panel with a smooth fade once the old tree is gone.
    document.body.classList.add('nav-switching');
    setTimeout(function () {
      var tabs = document.querySelectorAll('[data-testid="stTab"]');
      for (var i = 0; i < tabs.length; i++) {
        if (tabs[i].textContent.indexOf(name) !== -1) { tabs[i].click(); break; }
      }
      // Reveal once the old tree is FULLY gone. React's reconciliation can
      // flicker 1 -> 2 -> 1 panels mid-switch, so require a single visible
      // panel to persist across 3 consecutive polls (~300ms) — otherwise the
      // class drops too early and the old panel reappears on top of the new.
      var stable = 0;
      var tries = 0;
      var reveal = setInterval(function () {
        tries++;
        var panels = document.querySelectorAll('[role="tabpanel"]');
        var any = false;
        for (var i = 0; i < panels.length; i++) {
          var r = panels[i].getBoundingClientRect();
          if (r.width > 0 && r.height > 0) any = true;
        }
        if (panels.length === 1 && any) stable++;
        else stable = 0;
        if (stable >= 3 || tries > 40) {
          clearInterval(reveal);
          document.body.classList.remove('nav-switching');
        }
      }, 100);
    }, 80);
  });

  // Keep the navbar's active pill in sync with whichever native tab is open
  // (user may also click the native tabs directly).
  function syncNav() {
    var active = document.querySelector('[data-testid="stTab"][aria-selected="true"]');
    var txt = active ? active.textContent : '';
    var links = document.querySelectorAll('[data-nav-link]');
    for (var i = 0; i < links.length; i++) {
      var name = links[i].getAttribute('data-nav-link');
      links[i].classList.toggle('active', txt.indexOf(name) !== -1);
    }
  }
  setInterval(syncNav, 450);
  syncNav();

  function animateCount(el) {
    var target = parseFloat(el.getAttribute('data-count'));
    if (isNaN(target)) return;
    var dec = parseInt(el.getAttribute('data-decimals') || '0', 10);
    var suffix = el.getAttribute('data-suffix') || '';
    var dur = 1300, t0 = null;
    function step(t) {
      if (t0 === null) t0 = t;
      var p = Math.min((t - t0) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      var val = target * eased;
      el.textContent = (dec ? val.toFixed(dec) : Math.round(val).toLocaleString('en-IN')) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  var chips = document.querySelectorAll('[data-count]');
  for (var i = 0; i < chips.length; i++) animateCount(chips[i]);
})();
</script>
"""

# ---------------------------------------------------------------------------
# Custom CSS - premium dark fintech theme
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=Playfair+Display:wght@400;500;600;700;800&family=Lora:wght@400;500;600&family=Orbitron:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Fira+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500;600;700&display=swap');

/* Theme palette — defaults are Charcoal (dark). The selected theme overrides :root below. */
:root {
    --bg: #18181C; --bg2: #17171B;
    --bg-rad1: rgba(168,85,247,0.10); --bg-rad2: rgba(236,72,153,0.08);
    --panel: #1E1E24; --sidebar: #151519;
    --border: #2A2A32; --border-soft: #232329;
    --text: #FFFFFF; --text1: #E4E4E7; --text2: #D4D4D8;
    --dim: #A1A1AA; --faint: #71717A;
    --accent: #A855F7; --accent2: #EC4899; --lav: #C084FC;
    --green: #22C55E; --red: #EF4444;
    --blue1: #3B82F6; --blue2: #2563EB; --blue3: #4F8DFA;
    --font-head: 'Space Grotesk', sans-serif;
    --font-body: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(1000px 520px at 85% -10%, var(--bg-rad1), transparent 60%),
        radial-gradient(800px 460px at -10% 0%, var(--bg-rad2), transparent 55%),
        var(--bg);
}

h1, h2, h3 { font-family: var(--font-head); letter-spacing: -0.02em; color: var(--text1); }

/* --- Ticker tape --- */
.tape-wrap { background: var(--panel); border:1px solid var(--border); border-radius:14px; overflow:hidden; position:relative; margin-bottom:1.1rem; }
.tape-wrap::before, .tape-wrap::after { content:""; position:absolute; top:0; bottom:0; width:70px; z-index:2; pointer-events:none; }
.tape-wrap::before { left:0; background:linear-gradient(90deg, var(--bg), transparent); }
.tape-wrap::after { right:0; background:linear-gradient(-90deg, var(--bg), transparent); }
.tape-track { display:inline-flex; white-space:nowrap; animation:tape 45s linear infinite; will-change:transform; }
.tape-wrap:hover .tape-track { animation-play-state:paused; }
@keyframes tape { from{transform:translateX(0)} to{transform:translateX(-50%)} }
.tk { display:inline-flex; align-items:center; gap:8px; padding:9px 20px; font-size:12.5px; color:var(--text1); border-right:1px solid var(--border); font-family:var(--font-body); }
.tk b.up { color:var(--green); font-weight:600; }
.tk b.down { color:var(--red); font-weight:600; }

/* --- Sticky top navbar --- */
.topnav {
    position: sticky; top: 0; z-index: 900;
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    background: color-mix(in srgb, var(--bg) 84%, transparent);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 0.5rem 0.9rem; margin: 0.5rem 0 0.9rem 0;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}
.topnav .nav-brand { display:flex; align-items:center; gap:9px; font-family:var(--font-head); font-weight:700; color:var(--text1); font-size:0.95rem; }
.topnav .nb-ico {
    width:30px; height:30px; border-radius:9px; flex-shrink:0;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display:flex; align-items:center; justify-content:center; color:#fff;
    font-size:0.7rem; box-shadow: 0 4px 12px rgba(168,85,247,0.4);
}
.topnav .nav-links { display:flex; align-items:center; gap:4px; }
.topnav .nav-link {
    cursor:pointer; border:none; background:transparent;
    color:var(--dim); font-size:0.8rem; font-weight:600; font-family:var(--font-body);
    padding:0.44rem 0.85rem; border-radius:9px; white-space:nowrap;
    display:inline-flex; align-items:center; gap:6px;
    transition: all 0.2s ease;
}
.topnav .nav-link svg { flex-shrink:0; }
.topnav .nav-link:hover { color:var(--text1); background:rgba(168,85,247,0.12); }
.topnav .nav-link.active {
    color:#fff;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    box-shadow: 0 4px 14px rgba(168,85,247,0.35);
}

/* --- Hero banner --- */
.hero {
    position: relative; overflow: hidden;
    border-radius: 20px; border: 1px solid var(--border);
    padding: 2.1rem 2.3rem; margin: 0.2rem 0 0.5rem;
    background:
        radial-gradient(700px 300px at 85% -20%, rgba(168,85,247,0.30), transparent 60%),
        radial-gradient(600px 280px at 5% 115%, rgba(236,72,153,0.20), transparent 60%),
        linear-gradient(135deg, var(--bg2), var(--panel));
    display: flex; align-items: center; justify-content: space-between; gap: 1.6rem;
}
.hero .h-badge {
    display:inline-flex; align-items:center; gap:7px;
    font-size:0.66rem; font-weight:700; letter-spacing:0.14em;
    color:var(--lav); background:rgba(168,85,247,0.14);
    border:1px solid rgba(168,85,247,0.4); border-radius:999px; padding:0.32rem 0.85rem;
}
.hero h1 { margin:0.7rem 0 0.5rem 0; font-size:2.3rem; color:var(--text); line-height:1.12; }
.hero h1 .accent {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero .sub { color:var(--dim); font-size:0.92rem; max-width:560px; margin:0; }
.hero .h-ctas { display:flex; gap:10px; margin-top:1.2rem; flex-wrap:wrap; }
.hero .h-btn {
    cursor:pointer; border:none; border-radius:999px; padding:0.62rem 1.3rem;
    font-weight:600; font-size:0.85rem; font-family:var(--font-body);
    display:inline-flex; align-items:center; gap:8px; transition: all 0.22s ease;
}
.hero .h-btn.primary {
    color:#fff; background: linear-gradient(135deg, var(--accent), var(--accent2));
    box-shadow: 0 6px 18px rgba(168,85,247,0.4);
}
.hero .h-btn.primary:hover { transform:translateY(-2px); box-shadow: 0 10px 26px rgba(168,85,247,0.55); }
.hero .h-btn.ghost { color:var(--text1); background:rgba(255,255,255,0.05); border:1px solid var(--border); }
.hero .h-btn.ghost:hover { border-color:var(--accent); color:var(--lav); transform:translateY(-2px); }
.hero .h-art { position:relative; flex:0 0 310px; height:180px; }
.hero .h-art svg { width:100%; height:100%; display:block; }
/* floating stat chips over the hero art */
.h-chip {
    position:absolute; display:flex; flex-direction:column; align-items:center; gap:2px;
    background:var(--panel); border:1px solid var(--border); border-radius:14px;
    padding:0.5rem 0.9rem; box-shadow: 0 10px 28px rgba(0,0,0,0.45);
    animation: chipFloat 5s ease-in-out infinite;
}
.h-chip .hc-val { font-family:var(--font-head); font-weight:700; font-size:1.05rem; color:var(--text); line-height:1.1; }
.h-chip .hc-lbl { font-size:0.56rem; letter-spacing:0.14em; color:var(--faint); font-weight:700; }
.h-chip.c1 { top:6px; right:18px; animation-delay:0s; }
.h-chip.c2 { top:66px; right:140px; animation-delay:1.2s; }
.h-chip.c3 { bottom:8px; right:34px; animation-delay:2.1s; }
@keyframes chipFloat { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-9px); } }

/* --- KPI strip (metric columns) --- */
.kpis { display:flex; margin:1rem 0 1.2rem 0; background:var(--panel); border:1px solid var(--border); border-radius:16px; overflow:hidden; }
.kpi { flex:1; min-width:0; padding:0.9rem 1.1rem; }
.kpi + .kpi { border-left:1px solid var(--border); }
.kpi .kpi-label { font-size:0.62rem; font-weight:700; letter-spacing:0.16em; color:var(--faint); }
.kpi .kpi-val { font-size:1.35rem; font-weight:700; color:var(--text); font-family:var(--font-head); margin:0.25rem 0 0.1rem 0; }
.kpi .kpi-val .arr { font-size:0.95rem; margin-right:0.15rem; }
.kpi .kpi-val .up { color:var(--green); }
.kpi .kpi-val .down { color:var(--red); }
.kpi .kpi-sub { font-size:0.72rem; color:var(--faint); }

/* --- Cards --- */
.card {
    background: var(--panel);
    background-image: linear-gradient(105deg, transparent 42%, rgba(168,85,247,0.10) 50%, rgba(236,72,153,0.10) 56%, transparent 64%);
    background-size: 260% 100%;
    background-position: 130% 0;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.15rem 1.3rem;
    box-shadow: 0 8px 26px rgba(0,0,0,0.28);
}
.card .kicker { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.16em; color: var(--faint); margin-bottom: 0.55rem; }

/* --- Badges --- */
.badge { display:inline-block; padding: 0.4rem 1.2rem; border-radius: 999px; font-size: 1.25rem; font-weight: 700; letter-spacing: 0.04em; }
.badge-low    { background: rgba(239,68,68,0.14);   color: var(--red); border: 1px solid rgba(239,68,68,0.45); }
.badge-medium { background: rgba(168,85,247,0.16);  color: var(--lav); border: 1px solid rgba(168,85,247,0.5); }
.badge-high   { background: rgba(34,197,94,0.14);   color: var(--green); border: 1px solid rgba(34,197,94,0.45); }

.gain { font-size: 2.6rem; font-weight: 700; font-family: var(--font-head); line-height: 1.05; }
.gain-pos { color: var(--green); }
.gain-neg { color: var(--red); }
.gain .unit { font-size: 1rem; color: var(--faint); font-weight: 500; }

.insight {
    border-left: 3px solid var(--accent);
    background: rgba(168,85,247,0.08);
    padding: 0.65rem 0.95rem; border-radius: 0 10px 10px 0;
    color: var(--text2); margin: 0.6rem 0;
}

.section-title {
    font-size: 1.05rem; font-weight: 700; color: var(--text1);
    margin: 1.1rem 0 0.5rem 0; padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border);
    text-transform: uppercase; letter-spacing: 0.05em;
}

.stButton > button {
    border-radius: 999px; font-weight: 600; letter-spacing: 0.02em;
    border: none;
    background: linear-gradient(180deg, var(--blue1), var(--blue2));
    color: #FFFFFF;
    box-shadow: 0 6px 18px rgba(37,99,235,0.35);
}
.stButton > button:hover { background: linear-gradient(180deg, var(--blue3), var(--blue1)); color: #FFFFFF; }

div[data-testid="stMetricValue"] { color: var(--accent); font-family: var(--font-head); }

/* Tabs */
.stTabs [role="tablist"] { gap: 8px; border-bottom: 1px solid var(--border); justify-content: center; flex-wrap: wrap; padding: 0.4rem 0 0.9rem 0; }
.stTabs [data-testid="stTab"] {
    border-radius: 999px; padding: 0.55rem 1.35rem; font-weight: 600;
    color: var(--dim); border: 1px solid var(--border); background: var(--panel);
    font-family: var(--font-body); transition: all 0.22s ease; cursor: pointer;
}
.stTabs [data-testid="stTab"]:hover { color: var(--text1); border-color: var(--accent); }
.stTabs [data-testid="stTab"][aria-selected="true"] {
    color: #fff; border-color: transparent;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    box-shadow: 0 4px 16px rgba(168,85,247,0.35);
}
/* The native tab bar is hidden — the sticky top navbar drives navigation */
.stTabs [role="tablist"] { display: none; }
/* Smooth section switching WITHOUT double-content overlap. While a switch
   is in flight (body.nav-switching, added by NAV_JS), every panel is hidden
   so Streamlit's lingering old tree can never overlap the incoming one;
   when the old tree is removed the class drops and the panel fades in. */
[role="tabpanel"] { transition: opacity 0.22s ease; }
body.nav-switching [role="tabpanel"] { opacity: 0 !important; }

/* --- Sidebar --- */
[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--text1); }
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stMetric label { color: var(--text1); }
.side-logo { display:flex; align-items:center; gap:11px; margin:0.2rem 0 1.2rem 0; }
.side-logo .logo-ico {
    width:36px; height:36px; border-radius:11px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display:flex; align-items:center; justify-content:center;
    color:#fff; font-weight:700; font-family:var(--font-head); font-size:0.85rem;
    box-shadow: 0 6px 16px rgba(168,85,247,0.4);
}
.side-logo .logo-txt { font-weight:700; color:var(--text1); font-size:0.95rem; font-family:var(--font-head); line-height:1.15; }
.side-logo .logo-sub { color:var(--faint); font-size:0.66rem; letter-spacing:0.12em; }

/* --- Sidebar expand/collapse toggle buttons (>> / <<) ---
   Streamlit renders these with a near-invisible `fadedText60` color;
   give them a visible accent pill so they're obvious on dark themes. */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    background: rgba(168,85,247,0.14) !important;
    border: 1px solid var(--accent) !important;
    border: 1px solid color-mix(in srgb, var(--accent) 55%, transparent) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px rgba(168,85,247,0.25) !important;
    transition: all 0.25s ease !important;
}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
    background: rgba(168,85,247,0.30) !important;
    box-shadow: 0 0 16px rgba(168,85,247,0.5) !important;
    transform: scale(1.07);
}
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] * {
    color: var(--accent) !important;
}

/* --- Native widgets follow the theme (Streamlit 1.6x DOM) --- */
[data-testid="stWidgetLabel"] p, .stSelectbox label p, .stNumberInput label p, .stTextInput label p { color: var(--text1) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text1) !important; }
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4, [data-testid="stMarkdownContainer"] h5, [data-testid="stMarkdownContainer"] h6 { color: var(--text1) !important; }
[data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span { color: var(--faint) !important; }
[data-testid="stMetricLabel"] p { color: var(--text1) !important; }
.stRadio label p, .stCheckbox label p, .stToggle label p { color: var(--text1) !important; }
[data-testid="stExpander"] summary { color: var(--text1) !important; }

/* widget boxes: background + border + text */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stDateInput"] [data-baseweb="input"],
[data-testid="stTextArea"] [data-baseweb="textarea"] {
    background-color: var(--panel) !important;
    border-color: var(--border) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] [data-baseweb="input"] > div,
[data-testid="stTextInput"] [data-baseweb="input"] > div,
[data-testid="stDateInput"] [data-baseweb="input"] > div,
[data-testid="stTextArea"] [data-baseweb="textarea"] > div {
    background-color: var(--panel) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextArea"] textarea {
    color: var(--text1) !important;
}
[data-testid="stNumberInput"] [data-baseweb="input"] input::placeholder,
[data-testid="stTextInput"] [data-baseweb="input"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: var(--faint) !important;
}
[data-testid="stNumberInput"] button {
    background: var(--panel) !important;
    color: var(--text1) !important;
    border-color: var(--border) !important;
}

/* --- Pro-style promo card --- */
.promo { background:#FFFFFF; border-radius:16px; padding:1rem 1.1rem 1.1rem; margin-top:1.4rem; position:relative; box-shadow: 0 10px 30px rgba(0,0,0,0.22); }
.promo .pro-badge {
    position:absolute; top:-9px; left:14px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color:#fff; font-size:0.58rem; font-weight:700; letter-spacing:0.1em;
    padding:0.18rem 0.6rem; border-radius:999px;
}
.promo .p-title { color:#18181B; font-weight:700; font-size:1.2rem; margin:0.55rem 0 0.05rem; font-family:var(--font-head); }
.promo .p-sub { color:#71717A; font-size:0.72rem; margin:0 0 0.8rem; }
.promo .p-btn {
    display:inline-block; background:#18181B; color:#fff; border-radius:999px;
    padding:0.45rem 1.3rem; font-size:0.8rem; font-weight:600; text-decoration:none;
}

/* --- Recent IPOs table --- */
.ipo-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.ipo-table th {
    text-align:left; color:var(--faint); font-weight:600; font-size:0.68rem; letter-spacing:0.08em;
    padding:0.35rem 0.6rem; border-bottom:1px solid var(--border); text-transform:uppercase;
}
.ipo-table td { padding:0.55rem 0.6rem; border-bottom:1px solid var(--border-soft); color:var(--text1); }
.ipo-table tr:last-child td { border-bottom:none; }
.ipo-table .dim { color:var(--faint); }
.ipo-table .up { color:var(--green); font-weight:600; }
.ipo-table .down { color:var(--red); font-weight:600; }
.st-badge { border-radius:999px; padding:0.14rem 0.6rem; font-size:0.68rem; font-weight:600; }
.st-low    { background: rgba(239,68,68,0.14);   color:var(--red); }
.st-medium { background: rgba(168,85,247,0.16);  color:var(--lav); }
.st-high   { background: rgba(34,197,94,0.14);   color:var(--green); }

/* --- 3D hover tilt --- */
.card, .kpi { transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease, background-position 0.7s ease; will-change: transform; }
.card:hover {
    transform: perspective(900px) rotateX(2.5deg) rotateY(-2.5deg) translateY(-4px);
    box-shadow: 0 18px 44px rgba(0,0,0,0.45);
    border-color: rgba(168,85,247,0.45);
}
.kpi:hover {
    transform: perspective(900px) rotateX(2deg) rotateY(-2deg) translateY(-3px);
    box-shadow: 0 14px 34px rgba(0,0,0,0.4);
}

/* --- Tequila-style hover: rotating dashed ring + lift + glow --- */
@keyframes spin-dash { to { transform: rotate(360deg); } }

.card, .promo, .insight { position: relative; }
.card::after, .promo::after {
    content: ""; position: absolute; inset: -1px; border-radius: 17px;
    border: 1.5px dashed rgba(168,85,247,0);
    pointer-events: none; opacity: 0;
    transition: opacity 0.3s ease;
}
.card:hover::after, .promo:hover::after {
    opacity: 1; border-color: rgba(168,85,247,0.6);
    animation: spin-dash 9s linear infinite;
}
.card:hover {
    transform: perspective(900px) rotateX(1.5deg) rotateY(-1.5deg) translateY(-7px);
    box-shadow: 0 22px 52px rgba(0,0,0,0.5), 0 0 0 1px rgba(168,85,247,0.25), 0 0 42px rgba(168,85,247,0.18);
    border-color: rgba(168,85,247,0.5);
    background-position: -40% 0;
}
.promo:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 44px rgba(0,0,0,0.5), 0 0 34px rgba(168,85,247,0.22);
}
.insight:hover {
    border-left-color: var(--accent2);
    background: rgba(168,85,247,0.13);
    transform: translateX(4px);
    transition: transform 0.25s ease, background 0.25s ease;
}

/* badge + table-row + button pops */
.badge, .st-badge { transition: transform 0.2s ease, filter 0.2s ease, box-shadow 0.2s ease; }
.badge:hover, .st-badge:hover { transform: scale(1.08); filter: brightness(1.25); box-shadow: 0 4px 16px rgba(168,85,247,0.35); }
.ipo-table tr { transition: background 0.2s ease; }
.ipo-table tr:hover td { background: rgba(168,85,247,0.07); }
.stButton > button { transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease; }
.stButton > button:hover { transform: translateY(-2px) scale(1.02); filter: brightness(1.1); box-shadow: 0 10px 28px rgba(59,130,246,0.5); }
.section-title { transition: color 0.25s ease; }
.section-title:hover { color: var(--lav); }

/* --- Responsive: adapt to phones and tablets --- */
.table-scroll { overflow-x: auto; }
.ipo-table { min-width: 440px; }

@media (max-width: 900px) {
    .kpis { flex-wrap: wrap; }
    .kpi { flex: 1 1 50%; max-width: 50%; }
    .kpi:nth-child(odd) { border-left: none; }
    .kpi:nth-child(n+3) { border-top: 1px solid var(--border); }
    .kpi .kpi-val { font-size: 1.15rem; white-space: nowrap; }
}

@media (max-width: 720px) {
    /* stack all st.columns vertically on small screens */
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        flex: 1 1 100% !important; min-width: 100% !important;
    }
    .hero { flex-direction: column; align-items: flex-start; padding: 1.6rem 1.4rem; gap: 1.2rem; }
    .hero h1 { font-size: 1.75rem; }
    .hero .sub { font-size: 0.85rem; }
    .hero .h-art { flex: 0 0 auto; width: 100%; height: 170px; }
    .topnav { flex-wrap: wrap; }
    .topnav .nav-links { order: 3; width: 100%; overflow-x: auto; padding-bottom: 2px; }
    .topnav .nav-link { flex: 1 0 auto; text-align: center; min-height: 44px; justify-content: center; }
    .stepper { max-width: 100%; gap: 8px; }
    .step { font-size: 0.7rem; }
}

@media (max-width: 520px) {
    .kpi { flex: 1 1 100%; max-width: 100%; }
    .kpi:nth-child(n+2) { border-top: 1px solid var(--border); border-left: none; }
    .kpi .kpi-label { font-size: 0.58rem; }
    .card { padding: 0.9rem 1rem; }
    .tk { padding: 8px 12px; font-size: 11px; }
    .gain { font-size: 2rem; }
    .badge { font-size: 1.05rem; padding: 0.35rem 1rem; }
}

/* Base font: EVERYTHING follows the chosen font style.
   .stApp * is !important so it beats Streamlit's native fonts;
   the heading/display rules below it re-apply the head font. */
.stApp { font-family: var(--font-body) !important; }
.stApp * { font-family: var(--font-body) !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { font-family: var(--font-head) !important; }
.stApp .kpi .kpi-val, .stApp .gain, .stApp .side-logo .logo-txt,
.stApp .logo-ico, .stApp .sp-logo, .stApp .promo .p-title,
.stApp div[data-testid="stMetricValue"] { font-family: var(--font-head) !important; }
/* Material icons (hamburger, > >, X, dropdown arrows...) are ligature glyphs
   drawn with Streamlit's icon font — restore it so they never show raw text. */
.stApp [data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }

/* --- UI/UX Pro Max: interaction + accessibility polish --- */
.stApp button, .stApp a, .stApp [role="button"], .stApp [data-testid="stTab"],
.stApp .nav-link, .stApp .h-btn { cursor: pointer; }
.stApp *:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px; border-radius: 6px;
}
/* Prevent number inputs from capturing wheel/scroll events when focused */
.stApp input[type="number"]::-webkit-inner-spin-button,
.stApp input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none; margin: 0;
}
.stApp input[type="number"] { -moz-appearance: textfield; }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

/* --- Predictor 2-step progress indicator --- */
.stepper { display:flex; align-items:center; gap:10px; margin:0.9rem 0 0.2rem; max-width:560px; }
.step { display:inline-flex; align-items:center; gap:8px; font-size:0.78rem; font-weight:600; color:var(--faint); font-family:var(--font-body); white-space:nowrap; }
.step .s-dot {
    width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
    font-size:0.72rem; font-weight:700; background:var(--panel); border:1px solid var(--border); color:var(--dim);
}
.step.done { color:var(--text1); }
.step.done .s-dot { background:linear-gradient(135deg, var(--accent), var(--accent2)); color:#fff; border-color:transparent; box-shadow:0 3px 10px rgba(168,85,247,0.35); }
.step.active { color:var(--lav); }
.step.active .s-dot { border-color:var(--accent); color:var(--accent); box-shadow:0 0 0 3px rgba(168,85,247,0.18); }
.step-line { flex:1; height:2px; border-radius:999px; background:var(--border); max-width:80px; transition: background 0.3s ease; }
.step-line.fill { background: linear-gradient(90deg, var(--accent), var(--accent2)); }

/* --- 3D playground skeleton shimmer (shown only on the very first build) --- */
.skel3d { border:1px solid var(--border); border-radius:14px; padding:1.2rem 1.4rem 1.4rem; background:var(--panel); position:relative; overflow:hidden; margin:0.4rem 0; }
.skel3d .sk-line { height:14px; border-radius:999px; background:var(--border-soft); margin-bottom:12px; position:relative; overflow:hidden; }
.skel3d .sk-block { height:270px; border-radius:12px; background:var(--border-soft); position:relative; overflow:hidden; }
.skel3d .sk-line::after, .skel3d .sk-block::after {
    content:""; position:absolute; inset:0; transform:translateX(-100%);
    background:linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    animation:skShimmer 1.3s infinite;
}
@keyframes skShimmer { 100% { transform:translateX(100%); } }

/* --- Mobile-only floating zoom controls for 3D charts --- */
.m3z-wrap { position:absolute; top:50px; right:10px; z-index:10; display:none; flex-direction:column; gap:6px; }
.m3z-btn { width:36px; height:36px; border-radius:10px; border:1px solid rgba(255,255,255,0.15); background:rgba(15,15,30,0.85); backdrop-filter:blur(6px); color:rgba(255,255,255,0.85); font-size:18px; font-weight:600; cursor:pointer; display:flex; align-items:center; justify-content:center; -webkit-tap-highlight-color:transparent; transition:background 0.15s,transform 0.1s; }
.m3z-btn:active { transform:scale(0.92); background:rgba(168,85,247,0.35); }
@media (max-width:720px) {
  .m3z-wrap { display:flex; }
  [data-testid="stPlotlyChart"] { position:relative; }
}

#MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Apply the selected theme palette (overrides the :root defaults above)
T = THEMES[theme]
st.markdown(
    f"<style>:root{{"
    f"--bg:{T['bg']};--bg2:{T['bg2']};--bg-rad1:{T['bg_rad1']};--bg-rad2:{T['bg_rad2']};"
    f"--panel:{T['panel']};--sidebar:{T['sidebar']};--border:{T['border']};--border-soft:{T['border_soft']};"
    f"--text:{T['text']};--text1:{T['text1']};--text2:{T['text2']};--dim:{T['dim']};--faint:{T['faint']};"
    f"--accent:{T['accent']};--accent2:{T['accent2']};--lav:{T['lav']};"
    f"--green:{T['green']};--red:{T['red']};--blue1:{T['blue1']};--blue2:{T['blue2']};--blue3:{T['blue3']};"
    f"}}</style>",
    unsafe_allow_html=True,
)

# Apply the selected font preset (overrides the --font-* defaults above)
st.markdown(
    f"<style>:root{{"
    f"--font-head:{F['head']};--font-body:{F['body']};"
    f"}}</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Simple smooth cursor glow — a soft light that glides under the mouse
# ---------------------------------------------------------------------------
CURSOR_GLOW_HTML = """
<script>
(function(){
  if (document.getElementById('cursor-glow')) return;  // guard against reruns
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var el = document.createElement('div');
  el.id = 'cursor-glow';
  el.style.cssText =
    'position:fixed;left:0;top:0;width:320px;height:320px;margin:-160px 0 0 -160px;border-radius:50%;' +
    'pointer-events:none;z-index:99999;opacity:0;transition:opacity 0.35s ease;' +
    'background:radial-gradient(circle, rgba(168,85,247,0.09) 0%, rgba(236,72,153,0.05) 45%, transparent 70%);';
  document.body.appendChild(el);
  var tx = -999, ty = -999, x = -999, y = -999;
  document.addEventListener('pointermove', function(e){
    tx = e.clientX; ty = e.clientY;
    el.style.opacity = '1';
  });
  document.addEventListener('pointerleave', function(){ el.style.opacity = '0'; });
  document.addEventListener('pointercancel', function(){ el.style.opacity = '0'; });
  (function loop(){
    requestAnimationFrame(loop);
    x += (tx - x) * 0.14;
    y += (ty - y) * 0.14;
    el.style.transform = 'translate3d(' + x + 'px,' + y + 'px,0)';
  })();
})();
</script>
"""
st.html(CURSOR_GLOW_HTML, unsafe_allow_javascript=True)

# ---------------------------------------------------------------------------
# 3D widget at the top of the Predictor tab (pure CSS 3D — works in every browser)
THREE_JS_HTML = """
<style>
  /* Kill the browser scrollbar: the iframe body's default 8px margin and the
     taller mobile layout were overflowing the 175px iframe. */
  html, body { margin: 0; padding: 0; overflow: hidden; }
  #ipo3d { scrollbar-width: none; }
  #ipo3d::-webkit-scrollbar { display: none; }
  @keyframes ipoSpin   { to { transform: rotateX(16deg) rotateY(var(--ry, 0rad)); } }
  @keyframes ipoPulse  { 0%,100% { height: calc(var(--h) * 0.18); } 50% { height: var(--h); } }
  @keyframes ipoFloat  { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-16px); } }
  @keyframes ipoRingGlow { 0%,100% { opacity: 0.45; } 50% { opacity: 1; } }
  #scene3d { perspective: 460px; touch-action: none; cursor: grab; }
  #scene3d.dragging { cursor: grabbing; }
  #stage3d { transform-style: preserve-3d; animation: ipoSpin 22s linear infinite; will-change: transform; }
  /* Same 175px layout on every screen — the iframe height — no caption strip,
     nothing to overlap or overflow, so no scrollbar anywhere. */
</style>
<div id="ipo3d" style="width:100%;height:175px;border-radius:16px;overflow:hidden;position:relative;background:radial-gradient(600px 220px at 72% -10%, rgba(168,85,247,0.18), transparent 60%), linear-gradient(160deg,#1E1E24,#17171B);border:1px solid #2A2A32;">
  <div style="position:absolute;top:10px;left:16px;z-index:2;font:600 8px/1.4 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;letter-spacing:.14em;color:rgba(161,161,170,0.5);">IPO MARKET · 3D</div>
  <div id="scene3d" style="position:absolute;inset:0;display:flex;align-items:flex-end;justify-content:center;">
    <div id="stage3d" style="position:relative;width:0;height:0;transform:rotateX(16deg) rotateY(0rad);">
      <div style="position:absolute;left:-48px;top:-48px;width:96px;height:96px;border:1px solid rgba(168,85,247,0.5);border-radius:50%;transform:rotateX(90deg);animation:ipoRingGlow 3s ease-in-out infinite;"></div>
      <div style="position:absolute;left:-64px;top:-64px;width:128px;height:128px;border:1px dashed rgba(236,72,153,0.35);border-radius:50%;transform:rotateX(90deg);animation:ipoRingGlow 3s ease-in-out 1.5s infinite;"></div>
    </div>
  </div>
</div>
<script>
(function(){
  var el = document.getElementById('ipo3d');
  if(!el) return;
  var stage = document.getElementById('stage3d');
  var scene = document.getElementById('scene3d');
  if(!stage || !scene) return;
  // Heights scaled ~1.4x so the tallest bar reaches ~75% of the 175px widget
  var HEIGHTS = [48, 87, 67, 118, 56, 101, 78, 132, 64, 109, 84, 123];
  for(var i = 0; i < 12; i++){
    var b = document.createElement('div');
    var color = (i % 3 === 0) ? '#EC4899' : '#A855F7';
    b.style.cssText = 'position:absolute;bottom:0;left:-6px;width:12px;height:0;--h:' + HEIGHTS[i] + 'px;border-radius:6px 6px 2px 2px;' +
      'background:linear-gradient(180deg,' + color + ',rgba(24,24,28,0.05));box-shadow:0 0 14px rgba(168,85,247,0.45);' +
      'transform:rotateY(' + (i * 30) + 'deg) translateZ(46px);transform-origin:center bottom;';
    b.style.animation = 'ipoPulse ' + (2.6 + (i % 5) * 0.5) + 's ease-in-out ' + (i * 0.18) + 's infinite';
    stage.appendChild(b);
  }
  for(var p = 0; p < 26; p++){
    var d = document.createElement('div');
    var s = 1.5 + (p % 3);
    d.style.cssText = 'position:absolute;width:' + s + 'px;height:' + s + 'px;border-radius:50%;opacity:0.35;' +
      'background:' + (p % 2 ? '#C084FC' : '#EC4899') + ';left:' + (Math.random() * 100) + '%;top:' + (Math.random() * 100) + '%;';
    d.style.animation = 'ipoFloat ' + (3 + (p % 5)) + 's ease-in-out ' + (p * 0.4) + 's infinite';
    scene.appendChild(d);
  }
  var ry = 0, dragging = false, lastX = 0, startRy = 0;
  scene.addEventListener('pointerdown', function(e){
    dragging = true; lastX = e.clientX; startRy = ry;
    scene.classList.add('dragging');
    try { scene.setPointerCapture(e.pointerId); } catch(err) {}
  });
  scene.addEventListener('pointermove', function(e){
    if(!dragging) return;
    ry = startRy + (e.clientX - lastX) * 0.012;
    stage.style.transform = 'rotateX(16deg) rotateY(' + ry + 'rad)';
  });
  function endDrag(){ dragging = false; scene.classList.remove('dragging'); }
  scene.addEventListener('pointerup', endDrag);
  scene.addEventListener('pointercancel', endDrag);
  var last = performance.now();
  (function loop(now){
    requestAnimationFrame(loop);
    if(!dragging){
      ry += (now - last) * 0.00011;
      stage.style.transform = 'rotateX(16deg) rotateY(' + ry + 'rad)';
    }
    last = now;
  })(last);
})();
</script>
"""

# ---------------------------------------------------------------------------
# Data + model loading (cached). Models are saved to disk after the first
# training, so subsequent starts load instantly instead of retraining.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_data_and_models():
    df = load_training_data()
    clf, scaler_clf, reg, scaler_reg = train_models(df)
    return df, clf, scaler_clf, reg, scaler_reg


# Full-screen branded loader, shown only while the models are actually
# training on the very first start (after that it loads from disk instantly).
SPLASH_HTML = """
<style>
#splash {
    position: fixed; inset: 0; z-index: 99999;
    background: var(--bg, #18181C);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 20px; font-family: var(--font-body);
}
.sp-logo {
    width: 64px; height: 64px; border-radius: 18px;
    background: linear-gradient(135deg, var(--accent, #A855F7), var(--accent2, #EC4899));
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 700; font-family: var(--font-head); font-size: 1.1rem;
    animation: spPulse 1.6s ease-in-out infinite;
}
@keyframes spPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(168,85,247,0.45); transform: scale(1); }
    50%      { box-shadow: 0 0 0 24px rgba(168,85,247,0); transform: scale(1.06); }
}
.sp-title { color: var(--text1, #E4E4E7); font-size: 1.05rem; font-weight: 600; letter-spacing: 0.02em; }
.sp-bar { width: 220px; height: 5px; border-radius: 999px; background: var(--border, #2A2A32); overflow: hidden; position: relative; }
.sp-bar span { position: absolute; inset: 0; border-radius: 999px; background: linear-gradient(90deg, var(--accent, #A855F7), var(--accent2, #EC4899)); animation: spSlide 1.2s ease-in-out infinite; }
@keyframes spSlide { 0% { transform: translateX(-100%); } 100% { transform: translateX(240%); } }
.sp-status { position: relative; height: 18px; color: var(--faint, #71717A); font-size: 0.8rem; }
.sp-status span { position: absolute; left: 0; right: 0; text-align: center; opacity: 0; animation: spText 6s linear infinite; }
.sp-status .l1 { animation-delay: 0s; } .sp-status .l2 { animation-delay: 2s; } .sp-status .l3 { animation-delay: 4s; }
@keyframes spText { 0%, 4% { opacity: 0; } 8%, 30% { opacity: 1; } 36%, 100% { opacity: 0; } }
</style>
<div id="splash">
    <div class="sp-logo">IPO</div>
    <div class="sp-title">IPO Predictor</div>
    <div class="sp-bar"><span></span></div>
    <div class="sp-status">
        <span class="l1">Loading IPO data…</span>
        <span class="l2">Training subscription-tier model…</span>
        <span class="l3">Training listing-gain model…</span>
    </div>
</div>
"""

if not st.session_state.get("_models_ready"):
    splash = st.empty()
    splash.markdown(SPLASH_HTML, unsafe_allow_html=True)
    try:
        df, clf, scaler_clf, reg, scaler_reg = get_data_and_models()
    finally:
        splash.empty()
    st.session_state["_models_ready"] = True
else:
    df, clf, scaler_clf, reg, scaler_reg = get_data_and_models()

# Lazy SHAP explainer for the tier classifier (built once per session)
_tree_explainer = None


def get_tree_explainer():
    global _tree_explainer
    if _tree_explainer is None:
        import shap  # lazy: only pay the ~5s import when the user opens the explainer
        _tree_explainer = shap.TreeExplainer(clf)
    return _tree_explainer


def explain_gain(issue_size, offer_price, year, qib, hni, rii, total):
    """Exact additive explanation for the linear gain model:
    predicted = baseline + sum(feature contributions)."""
    eng = engineer_input(issue_size, offer_price, year, qib, hni, rii, total)
    row = scaler_reg.transform(pd.DataFrame(
        [{k: eng[k] for k in GAIN_FEATURES}], columns=GAIN_FEATURES))[0]
    contrib = {f: float(c * v) for f, c, v in zip(GAIN_FEATURES, reg.coef_, row)}
    return float(reg.intercept_), contrib


def impact_chart(values_dict, x_title):
    """Horizontal bar chart of feature impacts (green = positive, red = negative)."""
    d = pd.DataFrame({"Feature": list(values_dict.keys()), "Impact": list(values_dict.values())})
    d = d.reindex(d["Impact"].abs().sort_values().index)  # biggest impact on top
    chart = (alt.Chart(d)
             .mark_bar(cornerRadius=4)
             .encode(
                 x=alt.X("Impact:Q", title=x_title),
                 y=alt.Y("Feature:N", title=""),
                 color=alt.condition(alt.datum.Impact >= 0, alt.value("#22C55E"), alt.value("#EF4444")),
             )
             .properties(height=max(140, 42 * len(d))))
    return chart


def expected_mae():
    """Typical gain-model error (from the out-of-time evaluation), for the results card."""
    try:
        return float(pd.read_csv("reports/figures/regression_summary.csv", index_col=0)["MAE"].min())
    except Exception:
        return 14.0


# ---------------------------------------------------------------------------
# Cached heavy chart builders — the 3D scatter (animation frames) costs ~1.3s
# to build, so it is built ONCE per (data, color-mode, theme, font) combo and
# reused on every rerun. This is what makes navbar switching instant.
#
# NOTE: st.cache_data keys on the ARGUMENTS, not the function body — bump
# _FIG_VERSION whenever you change a builder's code so stale cached figures
# (e.g. an old dragmode) don't keep being served after a hot reload.
# ---------------------------------------------------------------------------
_FIG_VERSION = "v5"  # bump when figure code changes


@st.cache_data(show_spinner=False)
def build_3d_scatter_fig(df, color_mode, pt, cbg, cf, ct, cg, accent, head_font, body_font, _ver=_FIG_VERSION):
    """Animated 3D scatter (year-by-year play). Fully styled, returned read-only."""
    import plotly.express as px
    d3 = add_log_features(df.copy())
    color_map = {"Low": "#EF4444", "Medium": "#A855F7", "High": "#22C55E"}
    color_col = "Subscription_Class"
    if color_mode == "predicted":
        X = scaler_clf.transform(d3[CLASS_FEATURES])
        d3["Pred_Tier"] = clf.predict(X)
        color_col = "Pred_Tier"
    frames_list = []
    for y in sorted(d3["Year"].unique()):
        sub = d3[d3["Year"] <= y].copy()
        sub["frame"] = y
        frames_list.append(sub)
    df_anim = pd.concat(frames_list, ignore_index=True)
    fig = px.scatter_3d(
        df_anim, x="log_Total", y="log_Issue_Size", z="Listing Gain",
        color=color_col, color_discrete_map=color_map,
        hover_name="IPO_Name",
        hover_data=["Year", "Date", "Total", "Issue_Size(crores)", "Offer Price"],
        custom_data=["IPO_Name", "Year", "Subscription_Class"],
        animation_frame="frame", animation_group="IPO_Name",
        labels={"log_Total": "Total subscription (log)",
                "log_Issue_Size": "Issue size (log ₹ cr)",
                "Listing Gain": "Listing gain %"},
        opacity=0.85,
    )
    fig.update_layout(
        template=pt, height=480,
        margin=dict(l=0, r=0, t=45, b=0),
        paper_bgcolor=cbg,
        scene=dict(bgcolor=cbg),
        font=dict(family=body_font, color=cf),
        showlegend=False,  # custom responsive legend below (see HTML)
        dragmode="pan",   # allow users to drag-to-rotate / pan / zoom
    )
    if fig.layout.sliders:
        fig.layout.sliders[0].update(
            currentvalue={"prefix": "Year: ", "font": {"family": body_font, "color": cf}},
            font={"family": body_font, "color": cf},
            bgcolor=cbg, activebgcolor=accent)
    return fig


def hero_html(df):
    """Hero banner: gradient panel, headline, CTA buttons, SVG chart art and
    floating stat chips (numbers count up on first load via NAV_JS)."""
    try:
        acc = pd.read_csv("reports/figures/classification_summary.csv", index_col=0)["Accuracy"].max()
        acc_pct = f"{acc * 100:.0f}"
    except Exception:
        acc_pct = "62"
    avg = float(df["Listing Gain"].mean())
    n = len(df)
    return f"""
    <div class="hero">
      <div>
        <span class="h-badge">🤖 AI-POWERED &nbsp;·&nbsp; {n} IPOs &nbsp;·&nbsp; 2010–2026</span>
        <h1>IPO Subscription &amp; <span class="accent">Listing Gain</span> Predictor</h1>
        <p class="sub">Machine learning predicts an IPO's demand tier and day-1 listing gain before you invest — trained on {n} Indian IPOs (2010–2026).</p>
        <div class="h-ctas">
          <button class="h-btn primary" data-nav-link="Predictor"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M12 1.8v3.2M12 19v3.2M1.8 12H5M19 12h3.2"/></svg>Predict an IPO</button>
          <button class="h-btn ghost" data-nav-link="Data &amp; EDA"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M4.5 20.5v-9M10 20.5V3.5M15.5 20.5v-13M21 20.5H2.5"/></svg>Explore Data</button>
        </div>
      </div>
      <div class="h-art">
        <svg viewBox="0 0 300 165" preserveAspectRatio="none">
          <defs>
            <linearGradient id="hfill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.5"/>
              <stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>
            </linearGradient>
          </defs>
          <path d="M0,142 L40,122 L80,130 L120,98 L160,104 L200,74 L240,60 L300,22 L300,165 L0,165 Z" fill="url(#hfill)"/>
          <path d="M0,142 L40,122 L80,130 L120,98 L160,104 L200,74 L240,60 L300,22" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M300,22 L300,-6" stroke="var(--accent2)" stroke-width="2" stroke-dasharray="5 5" fill="none"/>
          <circle cx="120" cy="98" r="3.4" fill="var(--lav)"/>
          <circle cx="200" cy="74" r="3.4" fill="var(--lav)"/>
          <circle cx="300" cy="22" r="4.5" fill="var(--accent2)"/>
          <circle cx="300" cy="22" r="9" fill="none" stroke="var(--accent2)" stroke-opacity="0.5" stroke-width="1.5"/>
        </svg>
        <div class="h-chip c1"><span class="hc-val" data-count="{n}"></span><span class="hc-lbl">IPOs</span></div>
        <div class="h-chip c2"><span class="hc-val" data-count="{avg:.1f}" data-decimals="1" data-suffix="%"></span><span class="hc-lbl">AVG GAIN</span></div>
        <div class="h-chip c3"><span class="hc-val" data-count="{acc_pct}" data-suffix="%"></span><span class="hc-lbl">ACCURACY</span></div>
      </div>
    </div>
    """


def ticker_html(df, n=26):
    """Scrolling TradingView-style tape of recent IPOs and their day-1 gains."""
    items = []
    for _, r in df.tail(n).iterrows():
        g = float(r["Listing Gain"])
        cls = "up" if g >= 0 else "down"
        items.append(f'<span class="tk">{r["IPO_Name"]} <b class="{cls}">{g:+.1f}%</b></span>')
    track = '<div class="tape-track">' + "".join(items) + "".join(items) + '</div>'
    return f'<div class="tape-wrap">{track}</div>'


def kpi_cards(df):
    """Dense TradingView-style KPI strip with live numbers."""
    try:
        acc = pd.read_csv("reports/figures/classification_summary.csv", index_col=0)["Accuracy"].max()
        acc_txt = f"{acc * 100:.0f}%"
    except Exception:
        acc_txt = "—"
    try:
        r2 = pd.read_csv("reports/figures/regression_summary.csv", index_col=0)["R2"].max()
        r2_txt = f"{r2:.2f}"
    except Exception:
        r2_txt = "—"
    avg = float(df["Listing Gain"].mean())
    avg_arr = "<span class=\"arr up\">↗</span>" if avg >= 0 else "<span class=\"arr down\">↘</span>"
    return f"""
    <div class="kpis">
      <div class="kpi"><div class="kpi-label">IPO UNIVERSE</div><div class="kpi-val">{len(df)} <span class=\"arr up\">↗</span></div><div class="kpi-sub">2010 – 2026</div></div>
      <div class="kpi"><div class="kpi-label">AVG DAY-1 GAIN</div><div class="kpi-val">{avg:+.1f}% {avg_arr}</div><div class="kpi-sub">across all IPOs</div></div>
      <div class="kpi"><div class="kpi-label">TIER MODEL</div><div class="kpi-val">{acc_txt} <span class=\"arr up\">↗</span></div><div class="kpi-sub">best out-of-time accuracy</div></div>
      <div class="kpi"><div class="kpi-label">GAIN MODEL</div><div class="kpi-val">{r2_txt} <span class=\"arr up\">↗</span></div><div class="kpi-sub">best out-of-time R²</div></div>
    </div>
    """


def recent_ipo_table(df, n=6, highlight=None):
    """Status-badge table of the most recent IPOs (Recent-order style).

    ``highlight`` is an IPO name whose row gets a pink highlight (used by the
    click-to-highlight feature on the 3D scatter).
    """
    tier_cls = {"Low": "st-low", "Medium": "st-medium", "High": "st-high"}
    rows = []
    # Sort by Date descending so the truly recent IPOs appear first
    _df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False)
    for _, r in _df.head(n).iterrows():
        g = float(r["Listing Gain"])
        cls = "up" if g >= 0 else "down"
        hl = ' style="background:rgba(236,72,153,0.16);font-weight:600;"' \
            if r["IPO_Name"] == highlight else ""
        rows.append(
            f'<tr{hl}><td>{r["IPO_Name"]}</td>'
            f'<td><span class="st-badge {tier_cls[r["Subscription_Class"]]}">{r["Subscription_Class"]}</span></td>'
            f'<td class="{cls}">{g:+.1f}%</td>'
            f'<td class="dim">{r["Year"]}</td></tr>')
    head = "<thead><tr><th>Company</th><th>Status</th><th>Gain</th><th>Year</th></tr></thead>"
    return (f'<div class="table-scroll"><table class="ipo-table">'
            f'{head}<tbody>{"".join(rows)}</tbody></table></div>')


def subscription_donut(df):
    """Altair donut chart of the subscription class mix (purple/pink/green)."""
    mix = df["Subscription_Class"].value_counts().reindex(CLASS_ORDER).reset_index()
    mix.columns = ["Class", "count"]
    return (alt.Chart(mix)
            .mark_arc(innerRadius=52, outerRadius=78, stroke="#1E1E24", strokeWidth=3)
            .encode(
                theta=alt.Theta("count:Q", stack=True),
                color=alt.Color("Class:N", sort=CLASS_ORDER,
                                scale=alt.Scale(domain=CLASS_ORDER,
                                                range=["#EF4444", "#A855F7", "#22C55E"]),
                                legend=alt.Legend(orient="bottom", title=None,
                                                  labelColor="#A1A1AA")),
                tooltip=["Class", "count"],
            )
            .properties(height=240, padding={"top": 20, "bottom": 0, "left": 10, "right": 10}))


# Lookup for the "try a real IPO" feature (label -> row)
IPO_LABEL = (df["IPO_Name"] + "  (" + df["Date"].astype(str) + ")").tolist()
IPO_LOOKUP = {label: row for label, row in zip(IPO_LABEL, df.to_dict("records"))}


def predict_tier(issue_size, offer_price, year):
    """Tier prediction from pre-IPO facts only (works for upcoming IPOs)."""
    eng = engineer_input(issue_size, offer_price, year, 1, 1, 1, 1)
    Xc = scaler_clf.transform(pd.DataFrame(
        [{k: eng[k] for k in CLASS_FEATURES}], columns=CLASS_FEATURES))
    prob_dict = dict(zip(clf.classes_, clf.predict_proba(Xc)[0]))
    tier = max(prob_dict, key=prob_dict.get)
    return tier, prob_dict


def predict_gain(issue_size, offer_price, year, qib, hni, rii, total):
    """Gain prediction from subscription numbers (after bidding closes)."""
    eng = engineer_input(issue_size, offer_price, year, qib, hni, rii, total)
    Xr = scaler_reg.transform(pd.DataFrame(
        [{k: eng[k] for k in GAIN_FEATURES}], columns=GAIN_FEATURES))
    return float(reg.predict(Xr)[0])


def apply_real_ipo():
    label = st.session_state.real_ipo_select
    if not label:
        return
    row = IPO_LOOKUP[label]
    st.session_state.issue_size = float(row["Issue_Size(crores)"])
    st.session_state.offer_price = float(row["Offer Price"])
    st.session_state.year = float(row["Year"])
    st.session_state.qib = float(row["QIB"])
    st.session_state.hni = float(row["HNI"])
    st.session_state.rii = float(row["RII"])
    st.session_state.total = float(row["Total"])
    st.session_state.have_subs = True  # real IPOs always have subscription numbers


# ---------------------------------------------------------------------------
# Sidebar - quick facts
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="side-logo">'
        '<div class="logo-ico">IPO</div>'
        '<div class="logo-txt">IPO Predictor</div>'
        '</div>',
        unsafe_allow_html=True)
    st.radio("🎨 Theme", list(THEMES.keys()),
             index=list(THEMES.keys()).index(theme), key="theme_picker")
    st.selectbox("🔤 Font style", list(FONTS.keys()),
                 index=list(FONTS.keys()).index(font_style), key="font_picker")
    st.markdown("## 📊 Dataset at a glance")
    st.metric("IPOs trained on", f"{len(df)}")
    st.metric("Date range", f"{df['Year'].min()} – {df['Year'].max()}")
    st.metric("Avg. listing gain", f"{df['Listing Gain'].mean():+.1f}%")
    st.markdown("---")
    st.markdown("**Subscription mix**")
    for cls in CLASS_ORDER:
        n = int((df["Subscription_Class"] == cls).sum())
        st.markdown(f"`{cls:<6}` {n} IPOs  ({n / len(df) * 100:.0f}%)")
    st.markdown("---")
    st.markdown("**Best models (out-of-time test)**")
    try:
        cls_acc = pd.read_csv("reports/figures/classification_summary.csv", index_col=0)["Accuracy"].max()
        reg_r2 = pd.read_csv("reports/figures/regression_summary.csv", index_col=0)["R2"].max()
        st.markdown(f"- Tier: best accuracy **{cls_acc * 100:.0f}%**")
        st.markdown(f"- Gain: best R² **{reg_r2:.2f}**")
    except FileNotFoundError:
        st.markdown("- Tier: Random Forest · Gain: Linear Regression")
    st.markdown("---")
    st.caption("⚠️ Predictions are rough estimates — not financial advice.")
    st.markdown(
        '<div class="promo">'
        '<span class="pro-badge">PRO</span>'
        '<div class="p-title">IPO Predictor Pro</div>'
        '<div class="p-sub">Unlock every feature</div>'
        '<a class="p-btn">Upgrade now</a>'
        '</div>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Ticker tape + sticky navbar + hero banner + KPI strip
# ---------------------------------------------------------------------------
st.markdown(ticker_html(df), unsafe_allow_html=True)

# Sticky top navbar — links switch the native tabs below (see NAV_JS)
st.markdown(
    """
    <div class="topnav">
      <div class="nav-brand"><span class="nb-ico">IP</span>IPO Predictor</div>
      <div class="nav-links">
        <button class="nav-link" data-nav-link="Predictor"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M12 1.8v3.2M12 19v3.2M1.8 12H5M19 12h3.2"/></svg>Predictor</button>
        <button class="nav-link" data-nav-link="Data &amp; EDA"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M4.5 20.5v-9M10 20.5V3.5M15.5 20.5v-13M21 20.5H2.5"/></svg>Data &amp; EDA</button>
        <button class="nav-link" data-nav-link="Model Performance"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5.5" y="5.5" width="13" height="13" rx="2"/><rect x="9.5" y="9.5" width="5" height="5"/><path d="M9.5 1.8v3.7M14.5 1.8v3.7M9.5 18.5v3.7M14.5 18.5v3.7M1.8 9.5h3.7M1.8 14.5h3.7M18.5 9.5h3.7M18.5 14.5h3.7"/></svg>Model</button>
        <button class="nav-link" data-nav-link="About"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 10.8V17M12 7.2h.01"/></svg>About</button>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(hero_html(df), unsafe_allow_html=True)

# Navbar behaviour + hero count-up (runs once, survives reruns)
st.html(NAV_JS, unsafe_allow_javascript=True)

# Scroll preservation: save & restore scroll position across Streamlit reruns
# so number_input changes don't jump the user around the page.
SCROLL_FIX_HTML = """
<script>
(function(){
  if (window.__scrollFix) return;
  window.__scrollFix = true;
  var saved = null;
  var main = null;
  function getMain(){
    if (!main || !main.isConnected) main = document.querySelector('section[data-testid="stMain"]');
    return main;
  }
  /* --- Fix 1: Scroll preservation across Streamlit reruns --- */
  document.addEventListener('streamlit:rerun', function(){
    var m = getMain();
    if (m) saved = m.scrollTop;
  });
  document.addEventListener('focusout', function(e){
    if (e.target && e.target.type === 'number') {
      var m = getMain();
      if (m) saved = m.scrollTop;
    }
  });
  var observer = new MutationObserver(function(){
    if (saved === null) return;
    var m = getMain();
    if (m && m.scrollHeight > m.clientHeight) {
      var diff = Math.abs(m.scrollTop - saved);
      if (diff > 20) m.scrollTop = saved;
      saved = null;
    }
  });
  observer.observe(document.body, {childList: true, subtree: true});
  /* --- Fix 2: Let wheel scroll work even when cursor is inside a number input --- */
  document.addEventListener('wheel', function(e) {
    var t = e.target;
    if (t && t.type === 'number') {
      var m = getMain();
      if (m) {
        m.scrollTop += e.deltaY;
        e.preventDefault();
      }
    }
  }, {passive: false});
  /* --- Fix 3: Let touch-scroll work inside number inputs on mobile --- */
  document.addEventListener('touchmove', function(e) {
    var t = e.target;
    if (t && t.type === 'number') {
      t.blur();
    }
  }, {passive: true});
})();
</script>
"""
st.html(SCROLL_FIX_HTML, unsafe_allow_javascript=True)

st.markdown(kpi_cards(df), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs — keyed so Python can read the active section (and skip building the
# heavy 3D charts unless Data & EDA is actually on screen). on_change="rerun"
# makes the native tab the single source of truth: one tab click = one rerun,
# so navbar switching never races a second rerun.
# ---------------------------------------------------------------------------
tab_pred, tab_eda, tab_perf, tab_about = st.tabs(
    ["🎯 Predictor", "📊 Data & EDA", "🧠 Model Performance", "ℹ️ About"],
    key="active_tab", on_change="rerun")

# ===========================================================================
# TAB 1 - PREDICTOR
# ===========================================================================
with tab_pred:
    st.markdown('<div class="section-title">Predict an IPO</div>', unsafe_allow_html=True)
    st.caption("Enter an IPO's details below. Step 1 uses only pre-IPO facts; "
               "Step 2 uses the final subscription numbers (known after bidding closes).")

    # 2-step progress indicator (Step 2 fills once the toggle is switched on)
    _s2_done = bool(st.session_state.get("have_subs", False))
    st.markdown(
        '<div class="stepper">'
        '<span class="step done"><span class="s-dot">1</span>Pre-IPO facts</span>'
        f'<span class="step-line {"fill" if _s2_done else ""}"></span>'
        f'<span class="step {"done" if _s2_done else "active"}"><span class="s-dot">2</span>Subscription ratios</span>'
        '</div>',
        unsafe_allow_html=True)

    # Animated 3D intro widget
    st.iframe(THREE_JS_HTML, height=175)

    # Initial widget defaults - session_state is the single source of truth
    for _k, _v in {"issue_size": 500.0, "offer_price": 150.0, "year": 2026.0,
                   "qib": 45.0, "hni": 40.0, "rii": 12.0, "total": 25.0}.items():
        st.session_state.setdefault(_k, _v)

    col_inputs, col_results = st.columns([5, 6], gap="large")

    with col_inputs:
        with st.container(border=True):
            st.markdown("#### 🧾 Try a real IPO")
            st.selectbox(
                "Pick any IPO from the dataset to auto-fill the form:",
                options=[None] + IPO_LABEL,
                key="real_ipo_select",
                on_change=apply_real_ipo,
            )
            st.caption("Pick one → the inputs below fill themselves → press Predict.")

        st.markdown("### Step 1 · Pre-IPO details")
        issue_size = st.number_input(
            "Issue size (₹ crores)", min_value=0.1, step=50.0, key="issue_size")
        offer_price = st.number_input(
            "Offer price (₹)", min_value=1.0, step=10.0, key="offer_price")
        year = st.number_input(
            "Year of listing", min_value=2010, max_value=2030, step=1, key="year")

        have_subs = st.toggle(
            "Subscription closed — I have the final subscription numbers",
            value=False, key="have_subs",
            help="Turn this ON after bidding closes, then enter the QIB/HNI/RII/Total "
                 "numbers to also predict the listing gain.")

        if have_subs:
            st.markdown("### Step 2 · Subscription ratios (times)")
            qib = st.number_input("QIB", min_value=0.0, step=1.0, key="qib")
            hni = st.number_input("HNI", min_value=0.0, step=1.0, key="hni")
            rii = st.number_input("RII", min_value=0.0, step=1.0, key="rii")
            total = st.number_input("Total", min_value=0.0, step=1.0, key="total")
        else:
            st.caption("ℹ️ Pre-IPO mode: only the subscription tier is predicted. After "
                       "bidding closes, switch the toggle ON to also predict the listing gain.")

        if st.button("🚀 Predict this IPO", type="primary", width='stretch'):
            tier, prob_dict = predict_tier(issue_size, offer_price, float(year))
            results_payload = {
                "tier": tier, "probs": prob_dict,
                "real_label": st.session_state.get("real_ipo_select"),
            }
            if have_subs:
                gain = predict_gain(issue_size, offer_price, float(year),
                                    qib, hni, rii, total)
                results_payload["gain"] = gain
                # Explainability: additive explanation for the gain model
                baseline, contrib = explain_gain(issue_size, offer_price, float(year),
                                                 qib, hni, rii, total)
                results_payload["gain_expl"] = {"baseline": baseline, "contrib": contrib}
            # SHAP explanation for the tier classifier (predicted class)
            tier_expl = None
            if SHAP_OK:
                try:
                    eng = engineer_input(issue_size, offer_price, float(year), 1, 1, 1, 1)
                    row = scaler_clf.transform(pd.DataFrame(
                        [{k: eng[k] for k in CLASS_FEATURES}], columns=CLASS_FEATURES))
                    vals = get_tree_explainer()(row).values
                    if isinstance(vals, list):
                        vals = vals[list(clf.classes_).index(tier)][0]
                    elif vals.ndim == 3:
                        vals = vals[0][:, list(clf.classes_).index(tier)]
                    else:
                        vals = vals[0]
                    tier_expl = dict(zip(CLASS_FEATURES, vals))
                except Exception:
                    tier_expl = None
            results_payload["tier_expl"] = tier_expl
            st.session_state.results = results_payload

    with col_results:
        results = st.session_state.get("results")
        if results is None:
            st.markdown(
                '<div class="card"><div class="kicker">AWAITING INPUT</div>'
                'Fill the form on the left and press <b>Predict this IPO</b> — '
                'the model\'s verdict will appear here.</div>',
                unsafe_allow_html=True)
        else:
            tier, prob_dict = results["tier"], results["probs"]

            # --- Subscription tier card ---
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="kicker">PREDICTED SUBSCRIPTION TIER</div>', unsafe_allow_html=True)
            badge_cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}[tier]
            st.markdown(f'<span class="badge {badge_cls}">{tier.upper()}</span>', unsafe_allow_html=True)

            chart_data = pd.DataFrame({
                "Class": CLASS_ORDER,
                "Probability": [prob_dict[c] for c in CLASS_ORDER],
            })
            chart = (alt.Chart(chart_data)
                     .mark_bar(cornerRadius=6, size=26)
                     .encode(
                         x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                                 axis=alt.Axis(format="%", title="")),
                         y=alt.Y("Class:N", sort=CLASS_ORDER[::-1], title=""),
                         color=alt.Color("Class:N", legend=None,
                                         scale=alt.Scale(domain=CLASS_ORDER,
                                                         range=["#EF4444", "#A855F7", "#22C55E"])),
                     )
                     .properties(height=170))
            st.altair_chart(chart, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

            # --- Listing gain card ---
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="kicker">PREDICTED LISTING GAIN</div>', unsafe_allow_html=True)
            if "gain" in results:
                gain = results["gain"]
                gain_cls = "gain-pos" if gain >= 0 else "gain-neg"
                st.markdown(f'<div class="gain {gain_cls}">{gain:+.1f}<span class="unit">% on day 1</span></div>',
                            unsafe_allow_html=True)
                mae = expected_mae()
                st.caption(f"Typical model error is ±{mae:.0f} points → expect roughly "
                           f"{gain - mae:+.1f}% to {gain + mae:+.1f}%.")
            else:
                st.markdown("⏳ <b>Awaiting subscription numbers.</b> After bidding closes, "
                            "switch the toggle ON, enter QIB/HNI/RII/Total and predict again "
                            "to get the listing-gain forecast.", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # --- Real IPO comparison ---
            if results.get("real_label"):
                row = IPO_LOOKUP[results["real_label"]]
                actual_tier = row["Subscription_Class"]
                actual_gain = float(row["Listing Gain"])
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="kicker">HOW IT COMPARES · ACTUAL VS PREDICTED</div>',
                            unsafe_allow_html=True)
                st.markdown(f"**{row['IPO_Name']}** ({row['Date']})")
                c1, c2 = st.columns(2)
                c1.metric("Actual tier", actual_tier)
                c2.metric("Predicted tier", tier)
                c3, c4 = st.columns(2)
                c3.metric("Actual gain", f"{actual_gain:+.1f}%")
                pred_gain = results.get("gain")
                c4.metric("Predicted gain", "—" if pred_gain is None else f"{pred_gain:+.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)

            # --- Why this prediction? (explainability) ---
            with st.expander("🔍 Why this prediction? (SHAP explainability)"):
                if results.get("tier_expl") is not None:
                    st.markdown("**Subscription tier** — how each pre-IPO feature pushed the "
                                f"prediction toward **{tier.upper()}** (SHAP log-odds values):")
                    st.altair_chart(impact_chart(results["tier_expl"], "Impact on tier (log-odds)"),
                                    width='stretch')
                else:
                    st.info("Install SHAP (`pip install shap`) to see per-prediction "
                            "explanations for the tier.")
                g = results.get("gain_expl")
                if g:
                    st.markdown("**Listing gain** — the model starts from a baseline and adds "
                                "each feature's contribution:")
                    st.altair_chart(impact_chart(g["contrib"], "Contribution to gain (%)"),
                                    width='stretch')
                    st.caption(f"Baseline {g['baseline']:+.1f}% + contributions = "
                               f"predicted {results['gain']:+.1f}%")

# ===========================================================================
# TAB 2 - DATA & EDA
# ===========================================================================
with tab_eda:
    st.markdown('<div class="section-title">Market Overview</div>', unsafe_allow_html=True)

    # Row 1: subscription mix donut + recent IPOs table (reference grid style)
    c_left, c_right = st.columns(2, gap="large")
    with c_left:
        st.markdown('<div class="card"><div class="kicker">SUBSCRIPTION MIX</div>', unsafe_allow_html=True)
        st.altair_chart(subscription_donut(df), width='stretch')
        # Auto-dismiss the donut tooltip (class + count popout) after 5s so it
        # never sticks on the screen when the user scrolls away (mobile taps
        # and desktop clicks both leave it pinned). Works on both devices.
        st.html(
            '<script>\n'
            '(function () {\n'
            '  // vega-tooltip shows/hides via the CSS class "visible" on the\n'
            '  // .vg-tooltip element. Clicking/tapping a slice pins the tooltip,\n'
            '  // so poll: once it stays visible for 5 continuous seconds, hide it\n'
            '  // (works the same on mobile and desktop).\n'
            '  setInterval(function () {\n'
            '    var el = document.querySelector(".vg-tooltip");\n'
            '    if (!el) return;\n'
            '    if (el.classList.contains("visible")) {\n'
            '      if (!el.__shownAt) el.__shownAt = Date.now();\n'
            '      if (Date.now() - el.__shownAt > 5000) {\n'
            '        el.classList.remove("visible");\n'
            '        el.__shownAt = null;\n'
            '      }\n'
            '    } else {\n'
            '      el.__shownAt = null;\n'
            '    }\n'
            '  }, 500);\n'
            '})();\n'
            '</script>',
            unsafe_allow_javascript=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c_right:
        st.markdown('<div class="card"><div class="kicker">RECENT IPOS</div>', unsafe_allow_html=True)
        st.markdown(recent_ipo_table(df), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: average day-1 gain by year - purple gradient area chart
    if PLOTLY_OK:
        yearly = df.groupby("Year")["Listing Gain"].mean().reset_index()
        afig = px.area(yearly, x="Year", y="Listing Gain", markers=True)
        afig.update_traces(line=dict(color="#A855F7", width=2.5),
                           fillcolor="rgba(168,85,247,0.30)")
        afig.update_layout(
            template=CH["pt"], height=280, showlegend=False,
            margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor=CH["cbg"], plot_bgcolor=CH["cbg"],
            font=dict(family=F["body"], color=CH["cf"]),
            xaxis=dict(title="", gridcolor=CH["cg"], dtick=1),
            yaxis=dict(title="Avg day-1 gain %", gridcolor=CH["cg"]),
            # title anchored left + no modebar/drag-zoom = nothing can overlap
            # or break the chart on mobile
            title=dict(text="Average day-1 gain by year", x=0.01,
                       font=dict(family=F["head"], color=CH["ct"], size=15)),
            dragmode=False,
        )
        st.plotly_chart(afig, width='stretch',
                        config={"displayModeBar": False, "displaylogo": False})

    # Row 3: interactive 3D playground - timeline animation,
    #        predicted-vs-actual coloring.
    #        The figure is only built when the Data & EDA tab is actually
    #        open (tab_eda.open), so switching to other navbar sections
    #        never pays for it.
    if PLOTLY_OK and tab_eda.open:
        st.markdown('<div class="section-title">Interactive 3D Playground</div>',
                    unsafe_allow_html=True)
        pred_toggle = st.toggle("🎨 Color dots by **predicted** tier (model's view)",
                                key="eda_pred_toggle")
        st.caption("⏩ Press **Play** to watch IPOs list year by year — drag to rotate, scroll to zoom, hover any dot for details.")

        color_mode = "predicted" if pred_toggle else "actual"

        # Skeleton shimmer while the animated 3D figure builds the first time
        # (ui-ux-pro-max: loading feedback must match the expected wait)
        _play_built = st.session_state.get("_eda_3d_built", False)
        if not _play_built:
            _sk = st.empty()
            _sk.markdown(
                '<div class="skel3d"><div class="sk-line" style="width:38%"></div>'
                '<div class="sk-block"></div></div>',
                unsafe_allow_html=True)
        fig = build_3d_scatter_fig(df, color_mode, CH["pt"], CH["cbg"], CH["cf"],
                                   CH["ct"], CH["cg"], CH["accent"], F["head"], F["body"])
        if not _play_built:
            _sk.empty()
            st.session_state["_eda_3d_built"] = True

        st.plotly_chart(fig, width='stretch',
                        config={"displayModeBar": False, "displaylogo": False})

        # Mobile-only floating zoom buttons for the 3D scatter
        # Uses st.html (NOT iframed) so JS can reach the main page's
        # plotly charts and window.Plotly.
        _zoom_js = '<script>\n' + '\n'.join([
            '(function(){',
            'function addZoom(){',
            'var c=document.querySelectorAll("[data-testid=stPlotlyChart]");',
            'var last=c[c.length-1];',
            'if(!last||last.querySelector(".m3z-wrap"))return;',
            'var gd=last.querySelector(".js-plotly-plot");',
            'if(!gd||!gd.layout||!gd.layout.scene)return;',
            'var w=document.createElement("div");w.className="m3z-wrap";',
            'var bi=document.createElement("button");bi.className="m3z-btn";bi.textContent="+";bi.title="Zoom in";',
            'var bo=document.createElement("button");bo.className="m3z-btn";bo.textContent="\u2212";bo.title="Zoom out";',
            'w.appendChild(bi);w.appendChild(bo);last.appendChild(w);',
            'var sk=Object.keys(gd.layout).find(function(k){return k.indexOf("scene")===0;});',
            'function zoom(d){var cam=gd.layout[sk].camera||{};var eye=cam.eye||{x:1.25,y:1.25,z:1.25};',
            'var f=d>0?0.8:1.25;var ne={x:eye.x*f,y:eye.y*f,z:eye.z*f};',
            'window.Plotly&&window.Plotly.relayout(gd,sk+".camera.eye",ne);}',
            'bi.onclick=function(){zoom(1);};bo.onclick=function(){zoom(-1);};',
            '}',
            'addZoom();setTimeout(addZoom,2500);',
            '})();',
            '</script>'
        ])
        st.html(_zoom_js, unsafe_allow_javascript=True)

        # Inject the tier legend inside the chart element (bottom-left)
        _legend_color = CH["cf"]
        _legend_js = (
            '<style>.m3z-legend{position:absolute;top:55px;left:10px;z-index:10;display:flex;'
            'flex-wrap:wrap;gap:10px;align-items:center;font:11px sans-serif;'
            'color:' + _legend_color + ';opacity:.9;pointer-events:none;'
            'background:rgba(15,15,30,0.65);padding:4px 10px;border-radius:999px;'
            'border:1px solid rgba(255,255,255,0.08);backdrop-filter:blur(4px);}</style>\n'
            '<script>\n' + '\n'.join([
            '(function(){',
            'function addLegend(){',
            'var c=document.querySelectorAll("[data-testid=stPlotlyChart]");',
            'var last=c[c.length-1];',
            'if(!last||last.querySelector(".m3z-legend"))return;',
            'var gd=last.querySelector(".js-plotly-plot");',
            'if(!gd||!gd.layout||!gd.layout.scene)return;',
            'var leg=document.createElement("div");leg.className="m3z-legend";',
            'var t=document.createElement("span");t.textContent="TIER:";t.style.cssText="opacity:.6;letter-spacing:.05em";leg.appendChild(t);',
            'var tiers=[{c:"#EF4444",l:"Low"},{c:"#A855F7",l:"Med"},{c:"#22C55E",l:"High"}];',
            'tiers.forEach(function(t){',
            'var sp=document.createElement("span");sp.style.cssText="display:inline-flex;align-items:center;gap:5px";',
            'var dot=document.createElement("span");dot.style.cssText="width:8px;height:8px;border-radius:50%;background:"+t.c;',
            'var lb=document.createTextNode(t.l);',
            'sp.appendChild(dot);sp.appendChild(lb);leg.appendChild(sp);});',
            'last.appendChild(leg);',
            '}',
            'addLegend();setTimeout(addLegend,2500);',
            '})();',
            '</script>'
        ])
        )
        st.html(_legend_js, unsafe_allow_javascript=True)

    elif not PLOTLY_OK:
        st.info("Install plotly (`pip install plotly`) for the interactive 3D view.")

    # Ensure figures exist (generate on the fly if missing)
    fig_dir = "reports/figures"
    needed = ["correlation_heatmap.png", "distributions.png", "class_balance.png"]
    if not all(os.path.exists(os.path.join(fig_dir, f)) for f in needed):
        with st.spinner("Generating EDA figures..."):
            from src.eda import run_eda
            run_eda(df, outdir=fig_dir)

    # Layout matches each figure's shape: the wide two-panel distribution
    # pair gets the full width; the heatmap + class balance (similar aspect)
    # sit side by side — so nothing looks squashed or oversized on desktop.
    st.image(f"{fig_dir}/distributions.png",
             caption="Distribution of Total subscription (left) and Listing Gain (right)",
             width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        st.image(f"{fig_dir}/correlation_heatmap.png",
                 caption="How strongly features relate to each other (dark red = strong)",
                 width='stretch')
    with c2:
        st.image(f"{fig_dir}/class_balance.png",
                 caption="How many IPOs fall into each subscription tier",
                 width='stretch')

    st.markdown(
        '<div class="insight">💡 <b>Key insight:</b> subscription demand — especially '
        '<b>QIB</b> (institutional) and <b>Total</b> — correlates most strongly with listing gain '
        '(r ≈ 0.70). Issue size and offer price matter far less.</div>',
        unsafe_allow_html=True)

    with st.expander("👀 Peek at the cleaned data (first 10 rows)"):
        st.dataframe(df.head(10), width='stretch')

# ===========================================================================
# TAB 3 - MODEL PERFORMANCE
# ===========================================================================
with tab_perf:
    st.markdown('<div class="section-title">How well do the models predict?</div>',
                unsafe_allow_html=True)
    st.caption("All scores use an out-of-time split — models are trained on IPOs up to 2023 "
               "and tested on unseen 2024–2026 IPOs, exactly like predicting the future.")

    col_cls, col_reg = st.columns(2, gap="large")

    with col_cls:
        st.markdown("#### 🎯 Subscription tier (classification)")
        cls_path = "reports/figures/classification_summary.csv"
        if os.path.exists(cls_path):
            cls_sum = pd.read_csv(cls_path, index_col=0)
            st.dataframe(cls_sum[["Accuracy", "Precision", "Recall", "F1-Score"]].round(3),
                         width='stretch')
            best_acc = float(cls_sum["Accuracy"].max())
            st.markdown(
                f'<div class="insight"><b>Best accuracy ≈ {best_acc * 100:.0f}%</b> '
                '(vs ~33% random guessing) on never-seen 2024–2026 IPOs. '
                '<code>class_weight="balanced"</code> keeps the rarer classes from being ignored.</div>',
                unsafe_allow_html=True)
        else:
            st.info("Run `python main.py` to generate the summary tables.")
        with st.expander("Confusion matrices"):
            for name, img in [("Logistic Regression", "cm_logistic_regression.png"),
                              ("Random Forest", "cm_random_forest.png"),
                              ("SVM", "cm_svm.png")]:
                p = os.path.join(fig_dir, img)
                if os.path.exists(p):
                    st.image(p, caption=f"{name} — diagonal = correct predictions", width=420)

    with col_reg:
        st.markdown("#### 📈 Listing gain (regression)")
        reg_path = "reports/figures/regression_summary.csv"
        if os.path.exists(reg_path):
            reg_sum = pd.read_csv(reg_path, index_col=0)
            st.dataframe(reg_sum[["RMSE", "MAE", "R2"]].round(3), width='stretch')
            best_row = reg_sum["R2"].idxmax()
            best_r2 = float(reg_sum.loc[best_row, "R2"])
            best_mae = float(reg_sum.loc[best_row, "MAE"])
            if best_r2 >= 0:
                verdict = (f'<b>Best: {best_row} (R² ≈ {best_r2:.2f}, MAE ≈ {best_mae:.0f} pts).</b> '
                           'Predictions stay within a sensible range even on never-seen IPOs.')
            else:
                verdict = (f'<b>Best: {best_row} (R² = {best_r2:.2f}, MAE ≈ {best_mae:.0f} pts).</b> '
                           'A negative R² means the model cannot beat simply predicting the '
                           'average. Why? 2024–2026 subscription levels are far higher than '
                           'anything the model trained on before 2024 (median 26× vs 8×), so it '
                           'must extrapolate beyond its experience. This honest out-of-time test '
                           'exposes a limitation a random split would have hidden (it claimed R² ≈ 0.35).')
            st.markdown(f'<div class="insight">{verdict}</div>', unsafe_allow_html=True)
        else:
            st.info("Run `python main.py` to generate the summary tables.")
        imp_path = os.path.join(fig_dir, "feature_importance.png")
        if os.path.exists(imp_path):
            st.image(imp_path, caption="What drives listing gains — subscription demand dominates")

    st.markdown('<div class="section-title">Feature importance — the story the data tells</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="insight">💡 <b>QIB/HNI subscription is the strongest signal.</b> '
        'Institutional demand predicts first-day gains better than issue size or offer price. '
        'This replaces rumor-based (GMP) guessing with a data-driven signal.</div>',
        unsafe_allow_html=True)

# ===========================================================================
# TAB 4 - ABOUT
# ===========================================================================
with tab_about:
    st.markdown('<div class="section-title">About this project</div>', unsafe_allow_html=True)
    st.markdown(
        "**IPO Subscription & Listing Gain Prediction using Machine Learning**\n\n"
        "**The business problem:** retail investors have no reliable way to judge an IPO "
        "before applying — they depend on rumors and GMP talk. This project replaces that "
        "guesswork with data-driven predictions trained on 16 years of Indian IPO history.")

    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    steps = [
        ("1. Data collection", "Kaggle dataset of 652 Indian IPOs (2010–2026): subscription ratios, prices, listing gains."),
        ("2. Cleaning & EDA", "Parsed dates, imputed missing values, winsorized outliers, correlation & distribution analysis."),
        ("3. Task branching", "One target is split into two pipelines — classification (tier) and regression (gain %)."),
        ("4. Classification", "Logistic Regression, Random Forest & SVM predict Low / Medium / High demand from pre-IPO facts."),
        ("5. Regression", "Linear Regression, Decision Tree, Random Forest & Gradient Boosting predict day-1 gain %."),
        ("6. Deployment & insights", "This dashboard + feature-importance analysis rank what actually matters."),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")

    st.markdown('<div class="section-title">Dataset columns</div>', unsafe_allow_html=True)
    cols_info = pd.DataFrame({
        "Column": ["Date", "IPO_Name", "Issue_Size(crores)", "QIB", "HNI", "RII", "Total",
                   "Offer Price", "List Price", "Listing Gain"],
        "Meaning": [
            "Date the IPO opened", "Company name",
            "Total issue size in ₹ crores",
            "Qualified Institutional Buyers subscription (×)",
            "High Net-worth Individuals subscription (×)",
            "Retail Individual Investors subscription (×)",
            "Overall subscription ratio (×)",
            "Price band offered", "Price at listing",
            "Day-1 gain/loss vs offer price (%)",
        ],
    })
    st.dataframe(cols_info, width='stretch', hide_index=True)

    st.markdown('<div class="section-title">Team</div>', unsafe_allow_html=True)
    team = pd.DataFrame({
        "Name": ["Khandhar Piyush", "Khandla Ajay", "Khodada Ajay", "Meet", "Meghanathi Milan", "Mojidra Tushar"],
        "Enrollment No.": ["240130107057", "240130107058", "240130107059",
                           "240130107065", "240130107066", "240130107071"],
    })
    st.dataframe(team, width='stretch', hide_index=True)

    st.markdown(
        '<div class="insight">⚠️ <b>Disclaimer:</b> these predictions are for learning purposes. '
        'Stock markets are noisy — never base real investment decisions on this tool.</div>',
        unsafe_allow_html=True)
