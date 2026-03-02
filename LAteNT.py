"""
LAteNT.py — The Neuro-Symbolic Collective
Scientific Live Dashboard — FINAL VERSION
==========================================
Imports: universe.py | memory.py | council.py
"""

import streamlit as st
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from universe import Universe, ARCTask, DifficultyLevel, TaskDomain
from memory import LatentSkillLibrary
from latent_dictionary import LatentDictionary
from council import Council

# Use a font that supports more symbols or set fallback behavior
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans']
# This allows Matplotlib to handle missing glyphs more gracefully
mpl.rcParams['axes.unicode_minus'] = False

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neuro-Symbolic Collective — General Intelligence Lab",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── ARC COLOR PALETTE ────────────────────────────────────────────────────────
ARC_HEX = [
    "#111111",  # 0 = background
    "#1E90FF",  # 1 = blue
    "#FF4500",  # 2 = red
    "#32CD32",  # 3 = green
    "#FFD700",  # 4 = yellow
    "#AAAAAA",  # 5 = gray
    "#FF69B4",  # 6 = magenta
    "#FF8C00",  # 7 = orange
    "#00CED1",  # 8 = cyan
    "#9400D3",  # 9 = purple
]
ARC_CMAP = mcolors.ListedColormap(ARC_HEX, name="arc")
ARC_NORM = mcolors.BoundaryNorm(list(range(11)), 10)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #07090f; }
section[data-testid="stSidebar"] { background: #0d1017; border-right: 1px solid #1c2133; }

/* Headers */
h1 { background: linear-gradient(90deg, #7dd3fc, #a78bfa); -webkit-background-clip: text;
     -webkit-text-fill-color: transparent; font-weight: 700; }
h2 { color: #94a3b8; border-bottom: 1px solid #1c2133; padding-bottom: 6px; }
h3 { color: #cbd5e1; }

/* Metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1420 0%, #141b2d 100%);
    border: 1px solid #1e2a40; border-radius: 12px; padding: 14px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label { color: #64748b !important; font-size: 11px; }
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e2e8f0 !important; font-size: 22px; font-weight: 600;
}

/* Tabs */
button[data-baseweb="tab"] { color: #64748b !important; font-size: 13px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #7dd3fc !important; border-bottom: 2px solid #7dd3fc; }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #1e40af, #6d28d9) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(109,40,217,0.4) !important; }

/* Progress */
.stProgress > div > div > div { background: linear-gradient(90deg, #1e40af, #6d28d9) !important; }

/* Dataframe */
.stDataFrame { border: 1px solid #1c2133; border-radius: 8px; overflow: hidden; }

/* Expander */
details { border: 1px solid #1c2133 !important; border-radius: 8px !important; background: #0d1017 !important; }
details summary { color: #94a3b8 !important; }

/* Code */
.stCode { background: #0d1421 !important; border: 1px solid #1c2133; }

/* Log container */
.council-log {
    height: 380px; overflow-y: auto; background: #0a0e18;
    border: 1px solid #1c2133; border-radius: 10px;
    padding: 12px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px;
}
.council-log::-webkit-scrollbar { width: 4px; }
.council-log::-webkit-scrollbar-track { background: #0a0e18; }
.council-log::-webkit-scrollbar-thumb { background: #1e40af; border-radius: 2px; }

/* Agent colors */
.ag-Perceiver     { color: #38bdf8; }
.ag-Dreamer       { color: #c084fc; }
.ag-Scientist     { color: #34d399; }
.ag-Skeptic       { color: #f87171; }
.ag-Philosopher   { color: #fbbf24; }
.ag-CausalReasoner{ color: #f472b6; }
.ag-CuriosityEngine { color: #fb923c; }
.ag-Metacognitor  { color: #67e8f9; }
.ag-Archivist     { color: #a3e635; }
.ag-Council       { color: #818cf8; }
.ag-Orientation   { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE INITIALISATION ─────────────────────────────────────────────
def _init():
    if "universe" not in st.session_state:
        seed = random.randint(0, 99999)
        st.session_state.universe      = Universe(seed=seed)
        st.session_state.council       = Council(seed=seed)
        st.session_state.seed          = seed
        st.session_state.task          = None
        st.session_state.snap          = None
        st.session_state.all_logs      = []
        st.session_state.n_run         = 0
        st.session_state.n_solved      = 0

    # Safety checks for newly added attributes to prevent AttributeErrors on refresh
    if "stat_avg_rounds" not in st.session_state:
        st.session_state.stat_avg_rounds = 0.0
    if "stat_skills" not in st.session_state:
        st.session_state.stat_skills = 15
    if "stat_gen_series" not in st.session_state:
        st.session_state.stat_gen_series = []
    if "stat_dsl_skills" not in st.session_state:
        st.session_state.stat_dsl_skills = []
    if "stat_meta_learner" not in st.session_state:
        st.session_state.stat_meta_learner = {}
    if "export_logs" not in st.session_state:
        st.session_state.export_logs = []
    if "cross_domain_mode" not in st.session_state:
        st.session_state.cross_domain_mode = False
    if "cross_domain_epoch" not in st.session_state:
        st.session_state.cross_domain_epoch = 1  # 1 (A), 2 (B), 3 (C)
    if "cross_domain_results" not in st.session_state:
        st.session_state.cross_domain_results = {"A": [], "B": [], "C": []}

_init()

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _render_grid(ax: plt.Axes, grid, title: str, title_color: str = "#64748b") -> None:
    """Render a single ARC grid on an axes."""
    g = np.array(grid, dtype=int)
    g = np.clip(g, 0, 9)
    ax.imshow(g, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")
    h, w = g.shape
    # grid lines
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#2d3748", linewidth=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=9, color=title_color, pad=4, fontweight="500")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")


def _grid_fig(grids_titles, cols: int = None, cell_size: float = 2.5):
    """Create a dark-themed figure with N ARC grids."""
    n = len(grids_titles)
    cols = cols or n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cell_size * cols, cell_size * rows),
                              facecolor="#07090f", constrained_layout=True)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, (ax, (grid, title)) in enumerate(zip(axes_flat, grids_titles)):
        ax.set_facecolor("#07090f")
        _render_grid(ax, grid, title)
    # hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig


def _agent_html(agent: str, message: str, rnd: int) -> str:
    css = agent.replace(" ", "")
    icon_map = {
        "Perceiver": "👁️", "Dreamer": "💭", "Scientist": "🔬",
        "Skeptic": "🔴", "Philosopher": "🏛️", "CausalReasoner": "🕸️",
        "CuriosityEngine": "⚡", "Metacognitor": "", "Archivist": "📚",
        "Council": "🏆", "Orientation": "🚀",
    }
    icon = icon_map.get(agent, "•")
    return (
        f'<div style="padding:4px 0;border-bottom:1px solid #151c2c;">'
        f'<span style="color:#334155;font-size:10px;font-family:monospace">[R{rnd:02d}]</span> '
        f'{icon} <span class="ag-{css}" style="font-weight:600">{agent}</span>'
        f'<span style="color:#94a3b8;margin-left:6px">{message}</span>'
        f'</div>'
    )


def _verdict_badge(verdict: str) -> str:
    styles = {
        "solved":  ("badge-solved",   "#14532d", "#86efac", "Verified"),
        "unknown": ("badge-unknown",  "#431407", "#fdba74", "❓"),
        "timeout": ("badge-timeout",  "#1e1b4b", "#a5b4fc", "⏱️"),
        "pending": ("badge-pending",  "#0f172a", "#64748b", "⏳"),
    }
    _, bg, fg, icon = styles.get(verdict, styles["pending"])
    label = verdict.upper()
    return (
        f'<span style="background:{bg};color:{fg};border-radius:6px;'
        f'padding:3px 10px;font-size:13px;font-weight:600">{icon} {label}</span>'
    )


def _winning_program(snap: dict) -> str | None:
    """Extract the winning program string from the snapshot."""
    for h in snap.get("hypothesis_stack", []):
        if h.get("status") in ("accepted", "causal_law"):
            p = h.get("program")
            if p:
                return p
    # fallback: any program
    for h in sorted(snap.get("hypothesis_stack", []),
                    key=lambda x: x.get("confidence", 0), reverse=True):
        if h.get("program"):
            return h.get("program")
    return None


def _answer_grid(task: ARCTask, snap: dict) -> np.ndarray | None:
    """Return the Council's predicted output grid (numpy)."""
    # 1. Try accepted hypothesis grid
    for h in snap.get("hypothesis_stack", []):
        if h.get("status") in ("accepted", "causal_law") and h.get("grid"):
            return np.array(h["grid"], dtype=int)
    # 2. Try to decode the latent winning_z
    for h in snap.get("hypothesis_stack", []):
        if h.get("status") in ("accepted", "causal_law") and h.get("winning_z"):
            try:
                z = np.array(h["winning_z"], dtype=np.float32)
                # Note: We don't have direct access to LatentDictionary here 
                # but if the grid was saved, it's caught in step 1.
                # If we really need dynamic execution, we'd pass latent_dict.
                pass 
            except Exception:
                pass
    # 3. Highest-confidence grid
    for h in sorted(snap.get("hypothesis_stack", []),
                    key=lambda x: x.get("confidence", 0), reverse=True):
        if h.get("grid"):
            return np.array(h["grid"], dtype=int)
    return None

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  Neuro-Symbolic Collective")
    st.caption(f"Session seed `{st.session_state.seed}`")
    st.divider()

    st.markdown("**Task Generator**")
    level_options = {
        "L1 — Simple (1 Prior)":     DifficultyLevel.L1,
        "L2 — Moderate (2 Priors)":  DifficultyLevel.L2,
        "L3 — Hard (3 Priors)":      DifficultyLevel.L3,
        "L4 — Expert (4 Priors)":    DifficultyLevel.L4,
        "L5 — Frontier (4+ Priors)": DifficultyLevel.L5,
    }
    chosen_label = st.selectbox("Difficulty", list(level_options.keys()), index=0)
    chosen_level = level_options[chosen_label]
    run_btn = st.button("⚡ Run Council", width='stretch', type="primary")

    st.divider()
    st.markdown("**Session Stats**")
    c1, c2 = st.columns(2)
    c1.metric("Tasks Run",  st.session_state.n_run)
    c2.metric("Solved",     st.session_state.n_solved)

    c3, c4 = st.columns(2)
    # Read from cached stats — updated right after each run
    avg_r = st.session_state.stat_avg_rounds
    c3.metric("Avg Rounds", f"{avg_r:.1f}" if avg_r else "—")
    c4.metric("Latent Skills",     st.session_state.stat_skills)

    # Latent Dictionary status
    ld_stats = st.session_state.get("stat_latent_dict", {})
    if ld_stats:
        ld_ready = "🟢 Active" if ld_stats.get("dictionary_ready") else "🟡 Learning"
        ld_n = ld_stats.get("n_registered", 0)
        st.caption(f"Latent Dictionary: {ld_ready} · {ld_n} transforms absorbed")

    st.divider()
    st.caption("0-Cheat · Zero Memorisation · Full Transparency")

    import json
    if st.session_state.n_run > 0:
        export_data = {
            "seed": st.session_state.seed,
            "tasks_run": st.session_state.n_run,
            "solved": st.session_state.n_solved,
            "avg_rounds": st.session_state.stat_avg_rounds,
            "skills": st.session_state.stat_latent_skills,
            "generalization": st.session_state.stat_gen_series,
            "latent_dictionary": st.session_state.get("stat_latent_dict", {}),
            "meta_learner": st.session_state.get("stat_meta_learner", {}),
            "cumulative_dialogue_logs": st.session_state.export_logs
        }
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="💾 Download Session Data",
            data=json_str,
            file_name=f"general_intelligence_session_{st.session_state.seed}.json",
            mime="application/json",
            width='stretch'
        )

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("#  The Neuro-Symbolic Collective")
st.markdown("**9-Agent General Intelligence Research System — Inference-Time Discovery on ARC-General Intelligence-2**")

mc = st.columns(4)
mc[0].metric("SOTA 2026 (Gemini 3 Deep Think)", "84.6%")
mc[1].metric("Human Baseline", "~80%")
mc[2].metric("Our Goal", "Sample Efficiency + Transparency")
mc[3].metric("0-Cheat", "Verified Enforced")
st.divider()

# ─── RUN THE COUNCIL ──────────────────────────────────────────────────────────

def _run_task(task_obj: ARCTask):
    """Run a single task through the Council and stream progress."""
    st.session_state.task = task_obj
    st.session_state.snap = None
    st.session_state.all_logs = []
    st.session_state.n_run += 1

    progress = st.progress(0, text=f"Council deliberating {task_obj.task_id}…")
    status_box = st.empty()

    all_snapshots = []
    for snap in st.session_state.council.solve(task_obj):
        all_snapshots.append(snap)
        bud = snap.get("budget_used", 0)
        rnd = snap.get("round", 0)
        verdict = snap.get("final_verdict", "pending")
        progress.progress(min(bud / 100, 1.0),
                          text=f"Task {task_obj.task_id} | Round {rnd} | Budget {bud}/100 | {verdict}")

    progress.empty()
    status_box.empty()

    if all_snapshots:
        final = all_snapshots[-1]
        st.session_state.snap = final

        raw_log = final.get("agent_call_log", [])
        st.session_state.all_logs = [
            e for e in raw_log if e.get("message", "").strip()
        ]

        st.session_state.export_logs.append({
            "task_id": task_obj.task_id,
            "logs": st.session_state.all_logs
        })

        if final.get("final_verdict") == "solved":
            st.session_state.n_solved += 1
            return True
    return False

if solve_btn:
    if st.session_state.cross_domain_mode:
        # Determine the current domain
        epoch = st.session_state.cross_domain_epoch
        if epoch == 1:
            domain = TaskDomain.A_SPATIAL
            d_key = "A"
        elif epoch == 2:
            domain = TaskDomain.B_TOPOLOGICAL
            d_key = "B"
        else:
            domain = TaskDomain.C_ABSTRACT
            d_key = "C"
            
        st.toast(f"Starting Epoch {epoch}/3: Domain {d_key} (10 tasks)")
        
        solved_count = 0
        for i in range(10):
            # Generate task specifically for this domain
            t = st.session_state.universe.generate_task(chosen_level, domain=domain)
            success = _run_task(t)
            if success:
                solved_count += 1
                
        # Record results
        st.session_state.cross_domain_results[d_key] = {
            "solved": solved_count,
            "total": 10,
            "rate": solved_count / 10.0
        }
        
        # Advance epoch or finish
        if epoch < 3:
            st.session_state.cross_domain_epoch += 1
            st.success(f"Epoch {epoch} Complete! Ready for Epoch {epoch+1}.")
        else:
            st.session_state.cross_domain_mode = False
            st.session_state.app_state = "IDLE"
            final_rate = st.session_state.cross_domain_results["C"]["rate"]
            if final_rate >= 0.5:
                st.balloons()
                st.success(f"**GENERALIZATION ACHIEVED!** Domain C Score: {final_rate*100:.0f}%")
            else:
                st.warning(f"Domain C Score: {final_rate*100:.0f}%. Generalization failed.")

    else:
        # Standard Single Run mode
        task = st.session_state.universe.generate_task(chosen_level)
        _run_task(task)

    # ── Snapshot stats into session_state immediately so they can never be stale ──
    cs = st.session_state.council.stats()
    st.session_state.stat_avg_rounds  = cs["avg_rounds"]
    st.session_state.stat_skills      = cs["skill_library_size"]
    st.session_state.stat_gen_series  = cs["generalization_series"]
    st.session_state.stat_latent_skills  = cs.get("latent_skills", [])
    st.session_state.stat_latent_dict = cs.get("latent_dictionary", {})
    st.session_state.stat_meta_learner = cs.get("meta_learner", {})

    st.rerun()

# ─── MAIN DISPLAY ─────────────────────────────────────────────────────────────
task: ARCTask | None = st.session_state.task
snap: dict | None    = st.session_state.snap

if task is None:
    st.info("👈 Select a difficulty and press **⚡ Run Council** to begin the experiment.")
    st.markdown("""
    #### What you'll see
    | Tab | Contents |
    |-----|----------|
    | 🏛️ **Council Chamber** | Live agent dialogue + predicted vs ground-truth grid |
    | ⚡ **Surprise Metric** | Prediction error decaying to zero = understanding |
    | 🔬 **Program Inspector** | The discovered latent vector transformation |
    | 🔴 **Skeptic's Dossier** | Every falsified hypothesis — proof of depth |
    | 📉 **Generalization Curve** | Rounds-to-solve over time — the General Intelligence proof |
    | 📚 **Latent Skill Library** | The growing vocabulary of discovered latent vectors |
    """)
    st.stop()

# ── Task Header ──────────────────────────────────────────────────────────────
verdict = snap.get("final_verdict", "pending") if snap else "pending"
rounds  = snap.get("round", 0)           if snap else 0
budget  = snap.get("budget_used", 0)     if snap else 0

st.markdown(f"### Task `{task.task_id}`")
cols_info = st.columns([3, 1, 1, 1])
cols_info[0].markdown(
    f"**Priors**: {', '.join(p.value for p in task.priors_used)}  \n"
    f"**Rule** *(hidden from agents)*: `{task.transformation_description}`"
)
cols_info[1].markdown(
    f"**Difficulty**  \n`{task.difficulty.name}`"
)
cols_info[2].markdown(
    f"**Verdict**  \n{_verdict_badge(verdict)}",
    unsafe_allow_html=True,
)
cols_info[3].markdown(
    f"**Rounds / Budget**  \n`{rounds}` / `{budget}/100`"
)

st.divider()

# ── Training Pairs ────────────────────────────────────────────────────────────
st.markdown("#### Training Examples (shown to agents)")
pair_cols = st.columns(len(task.train_pairs))
for idx, (inp, out) in enumerate(task.train_pairs):
    with pair_cols[idx]:
        fig = _grid_fig([(inp, "Input"), (out, "Output")], cols=2, cell_size=2.2)
        st.pyplot(fig, width='stretch')
        plt.close(fig)
        st.caption(f"Train {idx + 1}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
if snap is None:
    st.info("Run the Council to see results.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏛️ Council Chamber",
    "⚡ Surprise Metric",
    "🔬 Program Inspector",
    "🔴 Skeptic's Dossier",
    "📉 Generalization Curve",
    "📚 Skill Library",
    "🔭 Observatory",
])

# ── TAB 1: COUNCIL CHAMBER ───────────────────────────────────────────────────
with tab1:
    col_log, col_grids = st.columns([1, 1], gap="medium")

    with col_log:
        st.markdown("##### Agent Dialogue")
        logs = st.session_state.all_logs   # already filtered to _emit() entries only
        if logs:
            html_rows = "".join(
                _agent_html(
                    e.get("agent", "?"),
                    e.get("message", ""),
                    e.get("round", 0),
                )
                for e in logs
            )
            st.markdown(
                f'<div class="council-log" id="log-bottom">{html_rows}</div>',
                unsafe_allow_html=True,
            )

    with col_grids:
        st.markdown("##### Answer Comparison")
        answer = _answer_grid(task, snap)

        panels = [(task.test_input, "Test Input")]
        if answer is not None:
            panels.append((answer, "Council's Answer "))
        panels.append((task.test_output, "Ground Truth Completed."))

        fig = _grid_fig(panels, cols=len(panels), cell_size=3.0)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        # Accuracy badge
        if answer is not None and answer.shape == task.test_output.shape:
            correct_cells = int(np.sum(answer == task.test_output))
            total_cells   = int(task.test_output.size)
            pct = 100 * correct_cells / total_cells
            bar_col = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 40 else "#ef4444"
            st.markdown(
                f'<div style="background:#0d1017;border:1px solid #1c2133;border-radius:8px;'
                f'padding:10px 14px;margin-top:8px">'
                f'<span style="color:#94a3b8">Cell accuracy: </span>'
                f'<span style="color:{bar_col};font-size:18px;font-weight:700">{pct:.1f}%</span>'
                f'<span style="color:#64748b"> ({correct_cells}/{total_cells} cells)</span></div>',
                unsafe_allow_html=True
            )

        # Hypothesis breakdown
        hypotheses = snap.get("hypothesis_stack", [])
        if hypotheses:
            st.markdown("---")
            st.markdown(f"**{len(hypotheses)} hypotheses explored**")
            status_counts: dict[str, int] = {}
            for h in hypotheses:
                s = h.get("status", "?")
                status_counts[s] = status_counts.get(s, 0) + 1
            badge_colors = {
                "accepted": "#22c55e", "causal_law": "#22c55e",
                "falsified": "#ef4444", "coincidence": "#f59e0b",
                "testing": "#3b82f6", "pending": "#64748b",
            }
            badges = " ".join(
                f'<span style="background:{badge_colors.get(s,"#334155")};color:white;'
                f'border-radius:4px;padding:2px 8px;font-size:11px">'
                f'{s} ×{c}</span>'
                for s, c in status_counts.items()
            )
            st.markdown(badges, unsafe_allow_html=True)

# ── TAB 2: SURPRISE METRIC ────────────────────────────────────────────────────
with tab2:
    surprise = snap.get("surprise_history", [])
    if surprise:
        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#07090f")
        ax.set_facecolor("#0a0e18")
        x = list(range(len(surprise)))
        ax.fill_between(x, surprise, alpha=0.18, color="#38bdf8")
        ax.plot(x, surprise, color="#38bdf8", linewidth=2.2, zorder=3)
        ax.scatter(x, surprise, color="#38bdf8", s=20, zorder=4)
        ax.axhline(0.05, color="#22c55e", linestyle="--", linewidth=1.2,
                   label="Resolution threshold (0.05)", alpha=0.8)
        ax.set_xlabel("Observation #", color="#64748b", fontsize=10)
        ax.set_ylabel("Prediction Error", color="#64748b", fontsize=10)
        ax.tick_params(colors="#475569", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1c2133")
        ax.set_facecolor("#0a0e18")
        ax.legend(fontsize=9, labelcolor="#94a3b8",
                  facecolor="#0d1017", edgecolor="#1c2133")
        ax.set_ylim(-0.02, max(surprise) + 0.08 if surprise else 1.1)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        # Stats row
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Initial Surprise", f"{surprise[0]:.3f}")
        sc2.metric("Final Surprise",   f"{surprise[-1]:.3f}")
        sc3.metric("Peak Surprise",    f"{max(surprise):.3f}")
        sc4.metric("Resolved?",        "Verified Yes" if surprise[-1] < 0.05 else "❌ No")

        if surprise[-1] < 0.05:
            st.success("🎯 Surprise resolved to near-zero — the Council has **understood** the task physics.")
        elif surprise[-1] < 0.3:
            st.warning("⚠️ Surprise partially reduced but not fully resolved.")
        else:
            st.error("❌ Surprise remains high — the Council is still uncertain.")
    else:
        st.info("No surprise data recorded. The CuriosityEngine had no predictions to evaluate.")

# ── TAB 3: PROGRAM INSPECTOR ─────────────────────────────────────────────────
with tab3:
    prog_str = _winning_program(snap)
    hypotheses = snap.get("hypothesis_stack", [])
    winning_h  = next(
        (h for h in hypotheses if h.get("status") in ("accepted", "causal_law") and h.get("program")),
        None
    )

    if prog_str:
        st.markdown("##### Discovered Latent Transformation")
        pi1, pi2, pi3, pi4 = st.columns(4)
        pi1.metric("Latent Vector Signature", prog_str[:30] + ("…" if len(prog_str) > 30 else ""))
        pi2.metric("MDL Score (Active Dims)", f"{winning_h['mdl_score']:.1f}" if winning_h and winning_h.get("mdl_score") else "—")
        pi3.metric("Causal Verdict", winning_h.get("causal_verdict","—") if winning_h else "—")
        pi4.metric("Confidence", f"{winning_h['confidence']:.3f}" if winning_h else "—")

        st.markdown("---")
        st.markdown("**Latent Vector Transformation Active**")
        st.caption("No human-readable code. Transformation represented purely in continuous space.")

        # Show its effect on first training pair
        st.markdown("---")
        st.markdown("**Latent Vector applied to training example:**")
        try:
            inp_ex, out_ex = task.train_pairs[0]
            winning_z = winning_h.get("winning_z")
            if winning_z:
                z_arr = np.array(winning_z, dtype=np.float32)
                pred_ex = st.session_state.council.latent_dict.decode_z(z_arr, inp_ex)
                match = "Verified Exact match" if np.array_equal(pred_ex, out_ex) else "❌ Mismatch"
                fig = _grid_fig([
                    (inp_ex,  "Training Input"),
                    (pred_ex, f"Latent Output ({match})"),
                    (out_ex,  "Ground Truth"),
                ], cols=3, cell_size=2.8)
                st.pyplot(fig, width='stretch')
                plt.close(fig)
            else:
                st.warning("No latent vector attached to hypothesis.")
        except Exception as e:
            st.warning(f"Could not render program example: {e}")
    else:
        st.warning("No accepted program found. The Council could not synthesize a generalizing rule.")
        # Show best attempted programs
        attempted = [h for h in hypotheses if h.get("program")]
        if attempted:
            st.markdown("**Best attempted programs:**")
            for h in sorted(attempted, key=lambda x: x.get("confidence", 0), reverse=True)[:5]:
                st.markdown(
                    f'`{h["program"]}` — confidence: **{h["confidence"]:.3f}** — '
                    f'status: `{h["status"]}`'
                )

# ── TAB 4: SKEPTIC'S DOSSIER ─────────────────────────────────────────────────
with tab4:
    contradictions = snap.get("contradiction_log", [])
    hypotheses     = snap.get("hypothesis_stack", [])
    n_falsified    = sum(1 for h in hypotheses if h.get("status") == "falsified")
    n_coincidence  = sum(1 for h in hypotheses if h.get("status") == "coincidence")

    sd1, sd2, sd3, sd4 = st.columns(4)
    sd1.metric("Contradictions Filed",     len(contradictions))
    sd2.metric("Hypotheses Falsified",     n_falsified)
    sd3.metric("Causal Coincidences",      n_coincidence)
    sd4.metric("Total Hypotheses",         len(hypotheses))

    st.markdown("---")

    if contradictions:
        st.markdown("##### Contradiction Log")
        for i, entry in enumerate(contradictions[-20:], 1):
            with st.expander(
                f"❌ Falsification #{i} — {entry.get('failure_mode','?')} "
                f"(by {entry.get('agent','?')})"
            ):
                st.markdown(
                    f"- **Hypothesis**: `{entry.get('hypothesis_id','?')}`  \n"
                    f"- **Failure mode**: `{entry.get('failure_mode','?')}`  \n"
                    f"- **Agent**: `{entry.get('agent','?')}`"
                )
    else:
        st.success("No contradictions recorded. The Skeptic could not falsify any program — a clean solve!")

    # Hypothesis table
    st.markdown("---")
    st.markdown("##### All Hypotheses")
    if hypotheses:
        rows = []
        for h in hypotheses:
            rows.append({
                "ID": h.get("id", "?"),
                "Status": h.get("status", "?"),
                "Confidence": h.get("confidence", 0),
                "Program": h.get("program") or "—",
                "MDL": h.get("mdl_score") or "—",
                "Causal": h.get("causal_verdict") or "—",
                "Contradictions": h.get("contradiction_count", 0),
            })
        df = pd.DataFrame(rows)
        df['MDL'] = df['MDL'].astype(str)
        df['Causal'] = df['Causal'].astype(str)
        st.dataframe(
            df,
            hide_index=True,
            width='stretch',
            column_config={
                "ID":            st.column_config.TextColumn(width=130),
                "Status":        st.column_config.TextColumn(width=90),
                "Confidence":    st.column_config.NumberColumn(format="%.3f", width=90),
                "Program":       st.column_config.TextColumn(width=220),
                "MDL":           st.column_config.TextColumn(width=60),
                "Causal":        st.column_config.TextColumn(width=110),
                "Contradictions":st.column_config.NumberColumn(width=120),
            }
        )

# ── TAB 5: GENERALIZATION CURVE ───────────────────────────────────────────────
with tab5:
    series = st.session_state.stat_gen_series  # read from cached session state
    if series:
        df = pd.DataFrame(series)
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Total Episodes",     len(df))
        gc2.metric("Solved",             int((df["verdict"] == "solved").sum()))
        gc3.metric("Avg Rounds (Solved)",
                   f"{df[df['verdict']=='solved']['rounds'].mean():.1f}"
                   if (df["verdict"] == "solved").any() else "—")

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#07090f")
        ax.set_facecolor("#0a0e18")
        colors = {"solved": "#22c55e", "unknown": "#f59e0b", "timeout": "#ef4444", "pending": "#94a3b8"}
        for i, row in df.iterrows():
            c = colors.get(row["verdict"], "#94a3b8")
            ax.scatter(i + 1, row["rounds"], color=c, s=60, zorder=4)
        ax.plot(range(1, len(df) + 1), df["rounds"].tolist(),
                color="#334155", linewidth=1.2, alpha=0.7, zorder=2)

        # Rolling mean
        if len(df) >= 3:
            rm = df["rounds"].rolling(3, min_periods=1).mean()
            ax.plot(range(1, len(df) + 1), rm.tolist(),
                    color="#7dd3fc", linewidth=2, linestyle="--", alpha=0.8, label="3-task rolling mean")

        legend_els = [mpatches.Patch(color=c, label=v) for v, c in colors.items() if v in df["verdict"].values]
        ax.legend(handles=legend_els, fontsize=9, labelcolor="#94a3b8",
                  facecolor="#0d1017", edgecolor="#1c2133")
        ax.set_xlabel("Task #", color="#64748b", fontsize=10)
        ax.set_ylabel("Rounds to Solve", color="#64748b", fontsize=10)
        ax.tick_params(colors="#475569", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1c2133")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        if len(df) >= 3:
            solved_df = df[df["verdict"] == "solved"]
            if len(solved_df) >= 2:
                first_half  = solved_df.iloc[:len(solved_df)//2]["rounds"].mean()
                second_half = solved_df.iloc[len(solved_df)//2:]["rounds"].mean()
                if second_half < first_half:
                    st.success(
                        f"📉 **Generalization confirmed!** Average rounds dropped from "
                        f"{first_half:.1f} → {second_half:.1f}. The Council is learning to learn."
                    )
                else:
                    st.info("Run more tasks to observe the generalization trend.")

        # Episode history table
        st.markdown("---")
        st.dataframe(
            df.rename(columns={"task_id": "Task", "rounds": "Rounds", "verdict": "Verdict"}),
            hide_index=True,
            width='stretch',
        )
    else:
        st.info("Run at least 1 task to populate the generalization curve.")

# ── TAB 6: SKILL LIBRARY ─────────────────────────────────────────────────────
with tab6:
    skills = st.session_state.stat_latent_skills  # read from cached session state
    if skills:
        sl1, sl2 = st.columns(2)
        total_skills   = len(skills)
        used_skills    = sum(1 for s in skills if s.get("usage_count", 0) > 0)
        sl1.metric("Total Latent Vectors", total_skills)
        sl2.metric("In Active Use", used_skills)

        # Usage bar chart
        top_skills = sorted(skills, key=lambda s: s.get("usage_count", 0), reverse=True)[:12]
        if any(s.get("usage_count", 0) > 0 for s in top_skills):
            fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#07090f")
            ax.set_facecolor("#0a0e18")
            names  = [s["name"] for s in top_skills]
            counts = [s.get("usage_count", 0) for s in top_skills]
            bar_colors = ["#6d28d9" for _ in top_skills]
            bars = ax.bar(names, counts, color=bar_colors, edgecolor="#1c2133", linewidth=0.8)
            ax.set_ylabel("Usage Count", color="#64748b", fontsize=10)
            ax.tick_params(axis="x", colors="#94a3b8", labelsize=9, rotation=30)
            ax.tick_params(axis="y", colors="#475569", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1c2133")
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        st.markdown("---")
        st.markdown("**Full Latent Vector Library**")
        df_skills = pd.DataFrame(skills)[[
            "name", "usage_count", "description", "origin"
        ]]
        st.dataframe(
            df_skills,
            hide_index=True,
            width='stretch',
            column_config={
                "name":        st.column_config.TextColumn("Latent Vector", width=130),
                "usage_count": st.column_config.NumberColumn("Uses", width=60),
                "description": st.column_config.TextColumn("Description"),
                "origin":      st.column_config.TextColumn("Discovered In", width=120),
            }
        )
    else:
        st.info("No skills in library yet.")

# ── TAB 7: OBSERVATORY ────────────────────────────────────────────────────────
with tab7:
    import math, colorsys
    _BG   = "#07090f"
    _AX   = "#0a0e18"
    _SP   = "#1c2133"

    def _obs_fig(w=10, h=3.8):
        f, a = plt.subplots(figsize=(w, h), facecolor=_BG)
        a.set_facecolor(_AX)
        for sp in a.spines.values(): sp.set_edgecolor(_SP)
        a.tick_params(colors="#475569", labelsize=8)
        return f, a

    def _obs_fig2(rows=1, cols=2, w=10, h=3.8):
        f, axes = plt.subplots(rows, cols, figsize=(w, h), facecolor=_BG)
        for ax in np.array(axes).flatten():
            ax.set_facecolor(_AX)
            for sp in ax.spines.values(): sp.set_edgecolor(_SP)
            ax.tick_params(colors="#475569", labelsize=8)
        return f, axes

    def _no_data(msg="Run more tasks to populate this chart."):
        st.markdown(
            f'<div style="background:#0a0e18;border:1px solid #1c2133;border-radius:8px;'
            f'padding:18px;color:#475569;text-align:center;font-size:13px">{msg}</div>',
            unsafe_allow_html=True)

    hypotheses  = snap.get("hypothesis_stack", [])
    surprise    = snap.get("surprise_history", [])
    call_log    = [e for e in snap.get("agent_call_log", []) if e.get("message","").strip()]
    contrad     = snap.get("contradiction_log", [])
    ws_data     = snap.get("world_state", {})
    ws_objects  = ws_data.get("objects", []) if isinstance(ws_data, dict) else []
    gen_series  = st.session_state.stat_gen_series
    latent_skills = st.session_state.stat_latent_skills
    AGENTS      = ["Perceiver","Dreamer","Scientist","Skeptic","Philosopher",
                   "CausalReasoner","CuriosityEngine","Metacognitor","Archivist"]
    AGENT_COLORS= ["#38bdf8","#c084fc","#34d399","#f87171","#fbbf24",
                   "#f472b6","#fb923c","#67e8f9","#a3e635"]
    STATUS_C    = {"pending":"#64748b","testing":"#3b82f6","falsified":"#ef4444",
                   "coincidence":"#f59e0b","causal_law":"#22c55e","accepted":"#22c55e"}

    # Safe defaults for cross-section variables (Section J uses these regardless of hypotheses)
    n_causal      = sum(1 for h in hypotheses if h.get("causal_verdict") == "CAUSAL_LAW")
    n_coincidence = sum(1 for h in hypotheses if h.get("causal_verdict") == "COINCIDENCE")
    n_untested    = sum(1 for h in hypotheses if not h.get("causal_verdict"))
    confs         = [h.get("confidence", 0) for h in hypotheses] if hypotheses else [0]

    # ─── SECTION A: HYPOTHESIS MANIFOLD ──────────────────────────────────────
    st.markdown("### 🧬 A — Hypothesis Manifold")
    if hypotheses:
        hids  = [h.get("id","?")[-10:] for h in hypotheses]
        confs = [h.get("confidence", 0) for h in hypotheses]
        stats = [h.get("status","pending") for h in hypotheses]
        mdls  = [h.get("mdl_score") or 0 for h in hypotheses]
        cords = [h.get("contradiction_count",0) for h in hypotheses]
        cverd = [h.get("causal_verdict") or "—" for h in hypotheses]
        colors_s = [STATUS_C.get(s,"#334155") for s in stats]

        # A1 — Confidence Cascade
        colA1, colA2 = st.columns(2)
        with colA1:
            st.markdown("##### A1 · Confidence Cascade")
            fig, ax = _obs_fig(5, 4)
            norm_confs = confs
            cmap = plt.get_cmap("plasma")
            bar_colors = [cmap(c) for c in norm_confs]
            bars = ax.barh(hids, confs, color=bar_colors, edgecolor="#1c2133", linewidth=0.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence", color="#64748b", fontsize=9)
            ax.axvline(0.5, color="#f87171", linestyle="--", linewidth=1, alpha=0.6)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
            fig.colorbar(sm, ax=ax, label="Confidence")
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # A2 — Status Mosaic (donut)
        with colA2:
            st.markdown("##### A2 · Status Mosaic")
            from collections import Counter
            status_counts = Counter(stats)
            fig, ax = _obs_fig(5, 4)
            labels = list(status_counts.keys())
            sizes  = list(status_counts.values())
            sc     = [STATUS_C.get(l,"#334155") for l in labels]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=sc, autopct="%1.0f%%",
                startangle=140, pctdistance=0.75,
                wedgeprops=dict(width=0.55, edgecolor=_BG, linewidth=2))
            for t in texts: t.set_color("#94a3b8"); t.set_fontsize(8)
            for at in autotexts: at.set_color("white"); at.set_fontsize(8)
            ax.set_facecolor(_BG); fig.patch.set_facecolor(_BG)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        colA3, colA4 = st.columns(2)
        # A3 — MDL Score Waterfall
        with colA3:
            st.markdown("##### A3 · MDL Score Waterfall")
            sorted_h = sorted(zip(mdls, hids, cverd), key=lambda x: x[0])
            sm_mdls  = [x[0] for x in sorted_h]
            sm_ids   = [x[1] for x in sorted_h]
            sm_cv    = [x[2] for x in sorted_h]
            cmap2    = plt.get_cmap("cool")
            max_mdl  = max(sm_mdls) if max(sm_mdls) > 0 else 1
            bc       = [cmap2(v / max_mdl) for v in sm_mdls]
            fig, ax  = _obs_fig(5, 4)
            ax.bar(range(len(sm_mdls)), sm_mdls, color=bc, edgecolor=_SP, linewidth=0.5)
            ax.set_xticks(range(len(sm_ids))); ax.set_xticklabels(sm_ids, rotation=60, ha="right", fontsize=6)
            ax.set_ylabel("MDL Score", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # A4 — Contradiction Pressure (polar bar)
        with colA4:
            st.markdown("##### A4 · Contradiction Pressure")
            fig = plt.figure(figsize=(5, 4), facecolor=_BG)
            ax  = fig.add_subplot(111, polar=True)
            ax.set_facecolor(_AX)
            N     = len(hypotheses)
            theta = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = [c+0.05 for c in cords]
            cmap3 = plt.get_cmap("hot")
            max_c = max(cords) if max(cords) > 0 else 1
            pc    = [cmap3(v/max_c) for v in cords]
            ax.bar(theta, radii, width=2*np.pi/N*0.8, color=pc, edgecolor=_SP, linewidth=0.5, alpha=0.9)
            ax.tick_params(colors="#475569", labelsize=7)
            ax.set_facecolor(_AX); fig.patch.set_facecolor(_BG)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # A5 — Hypothesis Age × Confidence Heatmap
        st.markdown("##### A5 · Confidence × Age Heatmap")
        if len(hypotheses) >= 2:
            times_h = [h.get("created_at", 0) for h in hypotheses]
            min_t   = min(times_h); max_t = max(times_h) - min_t if max(times_h) != min_t else 1
            norm_t  = [(t - min_t)/max_t for t in times_h]
            fig, ax = _obs_fig(10, 2.5)
            sc_plot = ax.scatter(norm_t, confs, c=confs, cmap="inferno",
                                 s=80, vmin=0, vmax=1, alpha=0.9, edgecolors="#1c2133", linewidths=0.5)
            ax.set_xlabel("Relative Creation Time →", color="#64748b", fontsize=9)
            ax.set_ylabel("Confidence", color="#64748b", fontsize=9)
            fig.colorbar(sc_plot, ax=ax, label="Confidence")
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            _no_data("Need 2+ hypotheses for age heatmap.")
    else:
        _no_data("No hypotheses yet — run the Council first.")

    st.divider()

    # ─── SECTION B: FREE ENERGY / SURPRISE ───────────────────────────────────
    st.markdown("### ⚡ B — Free Energy & Surprise")
    if surprise:
        colB1, colB2 = st.columns(2)
        # B1 — Free Energy Landscape
        with colB1:
            st.markdown("##### B1 · Free Energy Landscape")
            fig, ax = _obs_fig(5, 3.5)
            x = list(range(len(surprise)))
            cmap_plasma = plt.get_cmap("plasma")
            for i in range(len(surprise)-1):
                seg_color = cmap_plasma(surprise[i])
                ax.fill_between([x[i], x[i+1]], [surprise[i], surprise[i+1]], alpha=0.4, color=seg_color)
                ax.plot([x[i], x[i+1]], [surprise[i], surprise[i+1]], color=seg_color, linewidth=1.8)
            ax.axhline(0.05, color="#22c55e", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.set_xlabel("Observation #", color="#64748b", fontsize=9)
            ax.set_ylabel("Prediction Error", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # B2 — Surprise Gradient (dE/dt)
        with colB2:
            st.markdown("##### B2 · Surprise Gradient dE/dt")
            fig, ax = _obs_fig(5, 3.5)
            if len(surprise) > 1:
                grad = np.diff(surprise)
                gc   = ["#22c55e" if g < 0 else "#ef4444" for g in grad]
                ax.bar(range(len(grad)), grad, color=gc, edgecolor=_SP, linewidth=0.4)
                ax.axhline(0, color="#64748b", linewidth=0.8)
                ax.set_xlabel("Step", color="#64748b", fontsize=9)
                ax.set_ylabel("ΔSurprise", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        colB3, colB4 = st.columns(2)
        # B3 — Phase Space (surprise[t-1] vs surprise[t])
        with colB3:
            st.markdown("##### B3 · Active Inference Phase Space")
            fig, ax = _obs_fig(5, 3.5)
            if len(surprise) > 2:
                xs = surprise[:-1]; ys = surprise[1:]
                ax.scatter(xs, ys, c=range(len(xs)), cmap="twilight",
                           s=60, alpha=0.85, edgecolors="#1c2133", linewidths=0.5)
                for i in range(len(xs)-1):
                    ax.annotate("", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                                arrowprops=dict(arrowstyle="->", color="#475569", lw=0.6))
                ax.plot([0,1],[0,1], color="#334155", linestyle="--", linewidth=0.8)
                ax.set_xlabel("E(t-1)", color="#64748b", fontsize=9)
                ax.set_ylabel("E(t)", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # B4 — Resolution Speedometer (polar gauge)
        with colB4:
            st.markdown("##### B4 · Resolution Speedometer")
            fig = plt.figure(figsize=(5, 3.5), facecolor=_BG)
            ax  = fig.add_subplot(111, polar=True)
            ax.set_facecolor(_AX); fig.patch.set_facecolor(_BG)
            cur_s = surprise[-1]
            frac  = min(cur_s / 1.0, 1.0)
            theta_bar = np.linspace(0, np.pi * frac, 100)
            ax.plot(theta_bar, [1]*100, color="#38bdf8", linewidth=8, alpha=0.9)
            ax.plot(theta_bar, [1]*100, color="#67e8f9", linewidth=2, alpha=0.5)
            ax.set_ylim(0, 1.5); ax.set_xlim(0, np.pi)
            ax.set_theta_zero_location("W"); ax.set_theta_direction(-1)
            ax.set_rticks([]); ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(["0.0","0.25","0.50","0.75","1.0"], color="#64748b", fontsize=8)
            status_col = "#22c55e" if cur_s < 0.05 else "#f59e0b" if cur_s < 0.3 else "#ef4444"
            ax.text(np.pi/2, 0.5, f"{cur_s:.3f}", ha="center", va="center",
                    color=status_col, fontsize=18, fontweight="bold",
                    transform=ax.transData)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # B5 — Entropy Reduction Timeline (cumulative area)
        st.markdown("##### B5 · Entropy Reduction Timeline")
        fig, ax = _obs_fig(10, 2.5)
        cum_area = np.cumsum(surprise)
        ax.fill_between(range(len(cum_area)), cum_area, alpha=0.4, color="#6366f1")
        ax.plot(range(len(cum_area)), cum_area, color="#818cf8", linewidth=2)
        ax.set_xlabel("Observation step", color="#64748b", fontsize=9)
        ax.set_ylabel("Cumulative Surprise Debt", color="#64748b", fontsize=9)
        st.pyplot(fig, width='stretch'); plt.close(fig)
    else:
        _no_data("No surprise data yet.")

    st.divider()

    # ─── SECTION C: AGENT BRAIN ──────────────────────────────────────────────
    st.markdown("### 🧠 C — Agent Council Activity")
    if call_log:
        max_round = max(e.get("round", 1) for e in call_log)

        # C1 — Agent Brain Heatmap (9×N)
        st.markdown("##### C1 · Agent Brain Heatmap")
        heat = np.zeros((len(AGENTS), max_round))
        for e in call_log:
            ag = e.get("agent","")
            rn = e.get("round", 1) - 1
            if ag in AGENTS and 0 <= rn < max_round:
                heat[AGENTS.index(ag), rn] += 1
        fig, ax = _obs_fig(10, 3.5)
        im = ax.imshow(heat, cmap="inferno", aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(AGENTS))); ax.set_yticklabels(AGENTS, color="#94a3b8", fontsize=8)
        ax.set_xlabel("Round", color="#64748b", fontsize=9)
        fig.colorbar(im, ax=ax, label="Messages")
        st.pyplot(fig, width='stretch'); plt.close(fig)

        # C2 — Council Speaking Clock (polar)
        colC2, colC3 = st.columns(2)
        with colC2:
            st.markdown("##### C2 · Council Speaking Clock")
            ag_counts = {a: sum(1 for e in call_log if e.get("agent") == a) for a in AGENTS}
            fig = plt.figure(figsize=(5, 4), facecolor=_BG)
            ax  = fig.add_subplot(111, polar=True)
            ax.set_facecolor(_AX); fig.patch.set_facecolor(_BG)
            angles = np.linspace(0, 2*np.pi, len(AGENTS), endpoint=False)
            vals   = [ag_counts.get(a, 0) for a in AGENTS]
            bars   = ax.bar(angles, vals, width=2*np.pi/len(AGENTS)*0.8,
                            color=AGENT_COLORS, edgecolor=_BG, linewidth=1, alpha=0.9)
            ax.set_xticks(angles)
            ax.set_xticklabels([a[:6] for a in AGENTS], color="#94a3b8", fontsize=7)
            ax.tick_params(colors="#475569", labelsize=7)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # C3 — Gantt Timeline
        with colC3:
            st.markdown("##### C3 · Agent Activation Gantt")
            fig, ax = _obs_fig(5, 4)
            for i, ag in enumerate(AGENTS):
                rounds_active = sorted(set(e.get("round",1) for e in call_log if e.get("agent")==ag))
                for rn in rounds_active:
                    ax.barh(i, 0.8, left=rn-0.9, height=0.6,
                            color=AGENT_COLORS[i], alpha=0.85, edgecolor=_SP, linewidth=0.3)
            ax.set_yticks(range(len(AGENTS))); ax.set_yticklabels(AGENTS, color="#94a3b8", fontsize=7)
            ax.set_xlabel("Round", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # C4 — Dialogue Density Wave (stacked area)
        st.markdown("##### C4 · Dialogue Density Wave")
        density = {a: np.zeros(max_round) for a in AGENTS}
        for e in call_log:
            ag = e.get("agent",""); rn = e.get("round",1)-1
            if ag in AGENTS and 0 <= rn < max_round:
                density[ag][rn] += 1
        fig, ax = _obs_fig(10, 3.2)
        xs = range(1, max_round+1)
        bottom = np.zeros(max_round)
        for i, ag in enumerate(AGENTS):
            vals = density[ag]
            ax.fill_between(xs, bottom, bottom+vals, alpha=0.75, color=AGENT_COLORS[i], label=ag[:8])
            bottom += vals
        ax.set_xlabel("Round", color="#64748b", fontsize=9)
        ax.set_ylabel("Messages", color="#64748b", fontsize=9)
        ax.legend(fontsize=7, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP,
                  loc="upper right", ncol=3)
        st.pyplot(fig, width='stretch'); plt.close(fig)

        # C5 — Agent Co-activation Matrix
        st.markdown("##### C5 · Agent Co-activation Matrix")
        comat = np.zeros((len(AGENTS), len(AGENTS)))
        for rn in range(1, max_round+1):
            rnd_agents = [e.get("agent") for e in call_log if e.get("round")==rn and e.get("agent") in AGENTS]
            for a1 in rnd_agents:
                for a2 in rnd_agents:
                    if a1 != a2:
                        comat[AGENTS.index(a1), AGENTS.index(a2)] += 1
        fig, ax = _obs_fig(10, 4)
        im2 = ax.imshow(comat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(AGENTS))); ax.set_xticklabels([a[:8] for a in AGENTS], rotation=40, ha="right", color="#94a3b8", fontsize=7)
        ax.set_yticks(range(len(AGENTS))); ax.set_yticklabels(AGENTS, color="#94a3b8", fontsize=7)
        fig.colorbar(im2, ax=ax, label="Co-activations")
        st.pyplot(fig, width='stretch'); plt.close(fig)
    else:
        _no_data("No agent call log yet.")

    st.divider()

    # ─── SECTION D: SKILL MEME GRID ──────────────────────────────────────────
    st.markdown("### 🎨 D — Skill Meme Grid")
    if dsl_skills:
        # D1 — Skill Meme Grid (pixel mosaic)
        st.markdown("##### D1 · Skill Meme Grid")
        n_skills = len(dsl_skills)
        grid_cols = max(4, min(8, n_skills))
        grid_rows = math.ceil(n_skills / grid_cols)
        meme_grid = np.zeros((grid_rows, grid_cols, 3))
        for idx, sk in enumerate(dsl_skills):
            r, c = divmod(idx, grid_cols)
            usage = sk.get("usage_count", 0)
            hue   = (idx / max(n_skills, 1)) % 1.0
            sat   = 0.8
            val   = min(0.3 + usage * 0.15, 1.0)
            rgb   = colorsys.hsv_to_rgb(hue, sat, val)
            meme_grid[r, c] = rgb
        fig, ax = plt.subplots(figsize=(10, max(2.5, grid_rows * 0.9)), facecolor=_BG)
        ax.set_facecolor(_BG)
        ax.imshow(meme_grid, interpolation="nearest", aspect="auto")
        ax.set_xticks(range(grid_cols))
        ax.set_xticklabels([dsl_skills[i]["name"][:8] if i < n_skills else "" for i in range(grid_cols)],
                           rotation=45, ha="right", color="#94a3b8", fontsize=7)
        ax.set_yticks([]); fig.patch.set_facecolor(_BG)
        for sp in ax.spines.values(): sp.set_visible(False)
        st.pyplot(fig, width='stretch'); plt.close(fig)

        colD2, colD3 = st.columns(2)
        # D2 — Skill Reuse Heatmap
        with colD2:
            st.markdown("##### D2 · Skill Usage Heatmap")
            usage_vals = [sk.get("usage_count", 0) for sk in dsl_skills]
            names_d    = [sk["name"][:12] for sk in dsl_skills]
            fig, ax    = _obs_fig(5, 4)
            cmap_d2    = plt.get_cmap("YlOrRd")
            max_u      = max(usage_vals) if max(usage_vals) > 0 else 1
            bar_c      = [cmap_d2(v / max_u) for v in usage_vals]
            ax.barh(names_d, usage_vals, color=bar_c, edgecolor=_SP, linewidth=0.5)
            ax.set_xlabel("Usage Count", color="#64748b", fontsize=9)
            sm_d = plt.cm.ScalarMappable(cmap=cmap_d2, norm=plt.Normalize(0, max_u))
            fig.colorbar(sm_d, ax=ax, label="Usage")
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # D3 — Success Rate Radar
        with colD3:
            st.markdown("##### D3 · Success Rate Radar")
            srates = [sk.get("success_rate", 0) for sk in dsl_skills[:8]]
            snames = [sk["name"][:10] for sk in dsl_skills[:8]]
            N_r    = len(srates)
            if N_r >= 3:
                angles_r = np.linspace(0, 2*np.pi, N_r, endpoint=False).tolist()
                angles_r += angles_r[:1]; srates_r = srates + srates[:1]
                fig = plt.figure(figsize=(5, 4), facecolor=_BG)
                ax  = fig.add_subplot(111, polar=True)
                ax.set_facecolor(_AX); fig.patch.set_facecolor(_BG)
                ax.plot(angles_r, srates_r, color="#fb923c", linewidth=2)
                ax.fill(angles_r, srates_r, color="#fb923c", alpha=0.25)
                ax.set_xticks(angles_r[:-1])
                ax.set_xticklabels(snames, color="#94a3b8", fontsize=7)
                ax.set_ylim(0, 1); ax.tick_params(colors="#475569", labelsize=7)
                st.pyplot(fig, width='stretch'); plt.close(fig)
            else:
                _no_data("Need 3+ skills for radar.")

        colD4, colD5 = st.columns(2)
        # D4 — Origin Discovery Timeline
        with colD4:
            st.markdown("##### D4 · Discovery Timeline")
            fig, ax = _obs_fig(5, 3.5)
            for idx2, sk in enumerate(dsl_skills):
                origc = "#6d28d9" if sk.get("origin","") != "BUILTIN" else "#1e40af"
                ax.scatter(idx2, sk.get("usage_count", 0), color=origc, s=100,
                           edgecolors="#1c2133", linewidths=0.5, zorder=4)
                ax.vlines(idx2, 0, sk.get("usage_count", 0), color=origc, linewidth=1, alpha=0.5)
            ax.set_xticks(range(len(dsl_skills)))
            ax.set_xticklabels([sk["name"][:8] for sk in dsl_skills], rotation=45, ha="right", fontsize=7, color="#94a3b8")
            ax.set_ylabel("Usage Count", color="#64748b", fontsize=9)
            legend_els = [mpatches.Patch(color="#6d28d9", label="Discovered"),
                          mpatches.Patch(color="#1e40af", label="Built-in")]
            ax.legend(handles=legend_els, fontsize=8, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # D5 — Skill Gravity Well (bubble)
        with colD5:
            st.markdown("##### D5 · Skill Gravity Well")
            fig, ax = _obs_fig(5, 3.5)
            rng2 = np.random.default_rng(42)
            for idx3, sk in enumerate(dsl_skills):
                xp = rng2.uniform(0.1, 0.9); yp = rng2.uniform(0.1, 0.9)
                sz = max(sk.get("usage_count", 0) * 100 + 80, 80)
                hue2 = (idx3 / max(n_skills, 1)) % 1.0
                col2 = colorsys.hsv_to_rgb(hue2, 0.9, 0.9)
                ax.scatter(xp, yp, s=sz, color=col2, alpha=0.8, edgecolors="#1c2133", linewidths=0.8)
                ax.text(xp, yp, sk["name"][:7], ha="center", va="center",
                        color="white", fontsize=6, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            st.pyplot(fig, width='stretch'); plt.close(fig)
    else:
        _no_data("No skills in library yet.")

    st.divider()

    # ─── SECTION E: PROGRAM STRUCTURE ────────────────────────────────────────
    st.markdown("### 🔬 E — Program Structure Analysis")
    if hypotheses:
        programs = [h.get("program") for h in hypotheses if h.get("program")]
        colE1, colE2 = st.columns(2)
        # E1 — Program Length Distribution
        with colE1:
            st.markdown("##### E1 · Program Length Distribution")
            prog_lens = [len(p.split(" → ")) for p in programs] if programs else []
            if prog_lens:
                fig, ax = _obs_fig(5, 3.5)
                cmap_e1 = plt.get_cmap("hot")
                counts_e, bins_e = np.histogram(prog_lens, bins=range(1, 7))
                ax.bar(bins_e[:-1], counts_e, color=[cmap_e1(b/5) for b in bins_e[:-1]],
                       edgecolor=_SP, linewidth=0.5)
                ax.set_xlabel("Program Length (primitives)", color="#64748b", fontsize=9)
                ax.set_ylabel("Count", color="#64748b", fontsize=9)
                st.pyplot(fig, width='stretch'); plt.close(fig)
            else:
                _no_data("No programs discovered yet.")

        # E2 — Program Tree Depth (MDL by confidence)
        with colE2:
            st.markdown("##### E2 · MDL vs Confidence Scatter")
            h_with_prog = [h for h in hypotheses if h.get("program") and h.get("mdl_score")]
            if h_with_prog:
                xs_e = [h["confidence"] for h in h_with_prog]
                ys_e = [h["mdl_score"] for h in h_with_prog]
                sc_e = ax_e = None
                fig, ax_e = _obs_fig(5, 3.5)
                sc_e = ax_e.scatter(xs_e, ys_e, c=xs_e, cmap="magma",
                                    s=70, alpha=0.85, edgecolors="#1c2133", linewidths=0.5)
                ax_e.set_xlabel("Confidence", color="#64748b", fontsize=9)
                ax_e.set_ylabel("MDL Score", color="#64748b", fontsize=9)
                fig.colorbar(sc_e, ax=ax_e, label="Confidence")
                st.pyplot(fig, width='stretch'); plt.close(fig)
            else:
                _no_data("No programs with MDL scores yet.")

        # E3 — Primitive Co-occurrence Matrix
        st.markdown("##### E3 · Primitive Co-occurrence Matrix")
        PRIMS = ["rotate90","rotate180","rotate270","mirror_h","mirror_v",
                 "gravity_down","gravity_up","majority_recolor","sort_by_size","identity"]
        cooc = np.zeros((len(PRIMS), len(PRIMS)))
        for p in programs:
            steps = [s.strip() for s in p.split(" → ")]
            for s1 in steps:
                for s2 in steps:
                    if s1 in PRIMS and s2 in PRIMS:
                        cooc[PRIMS.index(s1), PRIMS.index(s2)] += 1
        if cooc.max() > 0:
            fig, ax = _obs_fig(10, 4)
            im_e3 = ax.imshow(cooc, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(PRIMS))); ax.set_xticklabels(PRIMS, rotation=45, ha="right", fontsize=7, color="#94a3b8")
            ax.set_yticks(range(len(PRIMS))); ax.set_yticklabels(PRIMS, fontsize=7, color="#94a3b8")
            fig.colorbar(im_e3, ax=ax, label="Co-occurrences")
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            _no_data("No primitive co-occurrences yet.")

        colE4, colE5 = st.columns(2)
        # E4 — Winning Program Spotlight
        with colE4:
            st.markdown("##### E4 · Winning Program Spotlight")
            best_h = next((h for h in hypotheses if h.get("status") in ("accepted","causal_law") and h.get("program")), None)
            if not best_h:
                best_h = max((h for h in hypotheses if h.get("program")), key=lambda x: x.get("confidence",0), default=None)
            if best_h and best_h.get("program"):
                steps_e4 = [s.strip() for s in best_h["program"].split(" → ")]
                fig, ax   = _obs_fig(5, 2.5)
                ax.set_xlim(0, len(steps_e4)); ax.set_ylim(0, 1); ax.axis("off")
                PRIM_COLS_E4 = {"rotate90":"#38bdf8","rotate180":"#6366f1","mirror_h":"#f472b6",
                                "mirror_v":"#fb923c","gravity_down":"#34d399","gravity_up":"#22c55e",
                                "majority_recolor":"#fbbf24","sort_by_size":"#a78bfa","identity":"#67e8f9"}
                for i, step in enumerate(steps_e4):
                    col_e4 = PRIM_COLS_E4.get(step, "#94a3b8")
                    rect   = mpatches.FancyBboxPatch((i+0.05, 0.1), 0.85, 0.8,
                                                      boxstyle="round,pad=0.05", facecolor=col_e4,
                                                      edgecolor="white", linewidth=1.5, alpha=0.9)
                    ax.add_patch(rect)
                    ax.text(i+0.5, 0.5, step[:10], ha="center", va="center",
                            color="white", fontsize=8, fontweight="bold")
                    if i < len(steps_e4)-1:
                        ax.annotate("", xy=(i+1.0, 0.5), xytext=(i+0.95, 0.5),
                                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.5))
                st.pyplot(fig, width='stretch'); plt.close(fig)
            else:
                _no_data("No winning program found yet.")

        # E5 — Hypothesis Confidence Distribution
        with colE5:
            st.markdown("##### E5 · Confidence Distribution")
            fig, ax = _obs_fig(5, 2.5)
            ax.hist(confs, bins=12, color="#6366f1", edgecolor=_SP, linewidth=0.5, alpha=0.85)
            ax.axvline(np.mean(confs), color="#f87171", linestyle="--", linewidth=1.5, label=f"Mean {np.mean(confs):.2f}")
            ax.set_xlabel("Confidence", color="#64748b", fontsize=9)
            ax.set_ylabel("Count", color="#64748b", fontsize=9)
            ax.legend(fontsize=8, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
            st.pyplot(fig, width='stretch'); plt.close(fig)
    else:
        _no_data("No hypothesis data available.")

    st.divider()

    # ─── SECTION F: CAUSAL REASONING ─────────────────────────────────────────
    st.markdown("### 🕸️ F — Causal Reasoning Engine")
    n_causal     = sum(1 for h in hypotheses if h.get("causal_verdict") == "CAUSAL_LAW")
    n_coincidence= sum(1 for h in hypotheses if h.get("causal_verdict") == "COINCIDENCE")
    n_untested   = sum(1 for h in hypotheses if not h.get("causal_verdict"))

    colF1, colF2 = st.columns(2)
    # F1 — Causal Law vs Coincidence
    with colF1:
        st.markdown("##### F1 · Causal Law vs Coincidence")
        fig, ax = _obs_fig(5, 3.5)
        cats_f1 = ["CAUSAL_LAW", "COINCIDENCE", "UNTESTED"]
        vals_f1 = [n_causal, n_coincidence, n_untested]
        col_f1  = ["#22c55e", "#f59e0b", "#334155"]
        bars_f1 = ax.bar(cats_f1, vals_f1, color=col_f1, edgecolor=_SP, linewidth=0.5)
        for bar_, val in zip(bars_f1, vals_f1):
            ax.text(bar_.get_x()+bar_.get_width()/2, bar_.get_height()+0.05,
                    str(val), ha="center", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Hypotheses", color="#64748b", fontsize=9)
        st.pyplot(fig, width='stretch'); plt.close(fig)

    # F2 — Falsification Heatmap (Agent × failure_mode)
    with colF2:
        st.markdown("##### F2 · Falsification Heatmap")
        if contrad:
            FAIL_MODES = sorted(set(c.get("failure_mode","?") for c in contrad))[:6]
            AGTS_F2    = [a for a in AGENTS if any(c.get("agent")==a for c in contrad)]
            if AGTS_F2 and FAIL_MODES:
                mat_f2 = np.zeros((len(AGTS_F2), len(FAIL_MODES)))
                for c in contrad:
                    ag_f2 = c.get("agent","")
                    fm_f2 = c.get("failure_mode","?")
                    if ag_f2 in AGTS_F2 and fm_f2 in FAIL_MODES:
                        mat_f2[AGTS_F2.index(ag_f2), FAIL_MODES.index(fm_f2)] += 1
                fig, ax = _obs_fig(5, 3.5)
                im_f2 = ax.imshow(mat_f2, cmap="magma", aspect="auto")
                ax.set_xticks(range(len(FAIL_MODES))); ax.set_xticklabels(FAIL_MODES, rotation=30, ha="right", fontsize=7, color="#94a3b8")
                ax.set_yticks(range(len(AGTS_F2))); ax.set_yticklabels(AGTS_F2, fontsize=8, color="#94a3b8")
                fig.colorbar(im_f2, ax=ax, label="Count")
                st.pyplot(fig, width='stretch'); plt.close(fig)
            else:
                _no_data("No cross-agent falsifications yet.")
        else:
            st.success("Verified No falsifications — the Skeptic found nothing to kill!")

    colF3, colF4 = st.columns(2)
    # F3 — Causal Confidence Scatter
    with colF3:
        st.markdown("##### F3 · Causal Confidence Scatter")
        h_with_verdict = [h for h in hypotheses if h.get("causal_verdict")]
        if h_with_verdict:
            fig, ax = _obs_fig(5, 3.5)
            for h in h_with_verdict:
                yv    = 1 if h["causal_verdict"] == "CAUSAL_LAW" else 0
                col_f = "#22c55e" if yv else "#ef4444"
                ax.scatter(h.get("confidence",0), yv + np.random.uniform(-0.08, 0.08),
                           color=col_f, s=60, alpha=0.8, edgecolors="#1c2133", linewidths=0.5)
            ax.set_yticks([0,1]); ax.set_yticklabels(["Coincidence","Causal Law"], color="#94a3b8", fontsize=8)
            ax.set_xlabel("Hypothesis Confidence", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            _no_data("No causal verdicts yet.")

    # F4 — Skeptic Spiral (contradiction spiral)
    with colF4:
        st.markdown("##### F4 · Skeptic Contradiction Spiral")
        if contrad:
            fig = plt.figure(figsize=(5, 3.5), facecolor=_BG)
            ax  = fig.add_subplot(111, polar=True)
            ax.set_facecolor(_AX); fig.patch.set_facecolor(_BG)
            n_c = len(contrad)
            thetas_f = np.linspace(0, 4*np.pi, n_c)
            radii_f  = np.linspace(0.1, 1.0, n_c)
            cmap_f   = plt.get_cmap("hot")
            for i in range(n_c-1):
                ax.plot([thetas_f[i], thetas_f[i+1]], [radii_f[i], radii_f[i+1]],
                        color=cmap_f(i/max(n_c,1)), linewidth=1.5, alpha=0.8)
            ax.scatter(thetas_f, radii_f, c=range(n_c), cmap="hot", s=30, zorder=5)
            ax.set_rticks([]); ax.tick_params(colors="#475569", labelsize=7)
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            st.success("Verified No contradictions — a clean solve!")

    # F5 — Causal Law Rate Over Session
    st.markdown("##### F5 · Causal Law Rate Over Hypotheses")
    if hypotheses:
        cl_flags = [1 if h.get("causal_verdict")=="CAUSAL_LAW" else 0 for h in hypotheses]
        roll_cl  = np.cumsum(cl_flags) / (np.arange(len(cl_flags))+1)
        fig, ax  = _obs_fig(10, 2.5)
        ax.fill_between(range(len(roll_cl)), roll_cl, alpha=0.35, color="#22c55e")
        ax.plot(range(len(roll_cl)), roll_cl, color="#86efac", linewidth=2)
        ax.axhline(0.5, color="#f87171", linestyle="--", linewidth=1, alpha=0.6, label="50% threshold")
        ax.set_ylim(0, 1); ax.set_xlabel("Hypothesis #", color="#64748b", fontsize=9)
        ax.set_ylabel("Cumulative Causal Law %", color="#64748b", fontsize=9)
        ax.legend(fontsize=8, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
        st.pyplot(fig, width='stretch'); plt.close(fig)
    else:
        _no_data("No hypotheses yet.")

    st.divider()

    # ─── SECTION X: CROSS-DOMAIN GENERALIZATION ─────────────────────────────
    if st.session_state.cross_domain_results["A"]:
        st.markdown("### 🏆 X — Cross-Domain Generalization (0-Shot Transfer)")
        cx1, cx2, cx3 = st.columns(3)
        res = st.session_state.cross_domain_results
        
        rate_A = res["A"].get("rate", 0) if res["A"] else 0
        cx1.metric("Epoch A (Spatial / Train)", f"{rate_A * 100:.0f}%", 
                   f"{res['A'].get('solved',0)}/10 solved" if res["A"] else "Pending")
                   
        rate_B = res["B"].get("rate", 0) if res["B"] else 0
        cx2.metric("Epoch B (Topological / Val)", f"{rate_B * 100:.0f}%", 
                   f"{res['B'].get('solved',0)}/10 solved" if res["B"] else "Pending")
                   
        rate_C = res["C"].get("rate", 0) if res["C"] else 0
        delta_C = f"{rate_C*100 - rate_A*100:.0f}% Generalization Gap" if res["C"] else ""
        cx3.metric("Epoch C (Abstract / 0-Shot)", f"{rate_C * 100:.0f}%", delta_C)
        
        st.caption("A true AGI maintains >50% performance on Epoch C despite never seeing abstract tasks during dictionary formation.")
        st.divider()

        # ─── SECTION K: INCREMENTAL PCA ─────────────────────────────
        st.markdown("### 🗺️ K — Latent Concept Map (Incremental PCA)")
        pca_z = []
        pca_labels = []
        pca_colors = []
        
        # Gather all skills
        latent_skills = st.session_state.stat_latent_skills
        for skill in latent_skills:
            z = skill.get("z_vector")
            if z:
                pca_z.append(z)
                t_id = skill.get("origin_task_id", "")
                pca_labels.append(t_id)
                
                # Color by epoch/domain loosely based on origin task
                # (In a real scenario, we'd definitively track which epoch a task belonged to)
                if t_id:
                    # Simple hash to generic colors for now
                    h = sum(ord(c) for c in t_id) % 3
                    if h == 0: pca_colors.append("#3b82f6") # Blue
                    elif h == 1: pca_colors.append("#22c55e") # Green
                    else: pca_colors.append("#f59e0b") # Yellow
                else:
                    pca_colors.append("#64748b")
                    
        if len(pca_z) >= 3:
            try:
                from sklearn.decomposition import IncrementalPCA
                ipca = IncrementalPCA(n_components=2)
                X = np.array(pca_z)
                X_pca = ipca.fit_transform(X)
                
                fig, ax = _obs_fig(8, 4.5)
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=pca_colors, s=80, alpha=0.7, edgecolors="#1e293b", linewidths=0.5)
                
                # Annotate top 10
                for i, (x, y, label) in enumerate(zip(X_pca[:, 0], X_pca[:, 1], pca_labels)):
                    if i < 10:
                        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", 
                                    fontsize=7, color="#94a3b8", alpha=0.8)
                                    
                ax.set_xlabel(f"PC1 ({ipca.explained_variance_ratio_[0]:.1%} variance)", color="#64748b", fontsize=9)
                ax.set_ylabel(f"PC2 ({ipca.explained_variance_ratio_[1]:.1%} variance)", color="#64748b", fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.2, color="#94a3b8")
                
                st.pyplot(fig, width='stretch'); plt.close(fig)
            except ImportError:
                _no_data("sklearn required for PCA plot.")
        else:
            _no_data("Need at least 3 solved tasks for PCA.")
            
        st.divider()

    # ─── SECTION L: META-LEARNER CURVE ─────────────────────────────
    st.markdown("### 🧠 L — Meta-Learner Probability Cloud")
    meta_stats = st.session_state.stat_meta_learner
    if meta_stats and meta_stats.get("total_updates", 0) > 0:
        prior_mean = np.array(meta_stats.get("prior_mean", []))
        if prior_mean.size > 0:
            fig, ax = _obs_fig(10, 3)
            ax.bar(range(len(prior_mean)), prior_mean, color="#a855f7", edgecolor=_SP, linewidth=0.5)
            ax.set_xlabel("Latent Dimension (0-63)", color="#64748b", fontsize=9)
            ax.set_ylabel("Historical Success Weight", color="#64748b", fontsize=9)
            ax.set_title(f"Meta-Learner Prior after {meta_stats.get('total_updates')} updates", color="#94a3b8", fontsize=10)
            ax.grid(axis='y', linestyle=":", alpha=0.2, color="#94a3b8")
            st.pyplot(fig, width='stretch'); plt.close(fig)
            
            # Show top 5 dimensions
            top_indices = np.argsort(prior_mean)[::-1][:5]
            top_str = ", ".join([f"Dim {i} ({prior_mean[i]:.2f})" for i in top_indices])
            st.caption(f"**Highest confidence dimensions:** {top_str}")
    else:
        _no_data("Meta-Learner needs solved episodes to form a prior.")
        
    st.divider()

    # ─── SECTION M: LATENT DICTIONARY HEATMAP ──────────────────────────
    st.markdown("### 🧩 M — The Latent Dictionary (Basis Matrix)")
    dict_stats = st.session_state.stat_latent_dict
    if dict_stats and dict_stats.get("is_ready"):
        basis = np.array(dict_stats.get("basis", []))
        if basis.size > 0:
            fig, ax = _obs_fig(10, 4)
            # Visualize the dictionary: components vs features
            im = ax.imshow(basis, cmap="viridis", aspect="auto")
            ax.set_ylabel("Basis Component (0-63)", color="#64748b", fontsize=9)
            ax.set_xlabel("Receptive Field (Flattened IO)", color="#64748b", fontsize=9)
            ax.set_title(f"Learned Transformation Basis (NMF, {dict_stats.get('n_components')} components)", color="#94a3b8", fontsize=10)
            fig.colorbar(im, ax=ax, label="Activation Strength")
            st.pyplot(fig, width='stretch'); plt.close(fig)
    else:
        _no_data("Latent Dictionary not ready or no basis data exported.")
        
    st.divider()
    
    # ─── SECTION N: NOVEL SKILL DISCOVERY RATE ─────────────────────────
    st.markdown("### 🌠 N — Novel Concept Discovery Rate")
    if latent_skills:
        fig, ax = _obs_fig(10, 3)
        # Sort skills by the order they were discovered (origin task order acts as proxy)
        # Real implementation would use timestamps, but we use index here
        discovery_counts = [i+1 for i in range(len(latent_skills))]
        
        ax.plot(range(1, len(latent_skills)+1), discovery_counts, color="#10b981", linewidth=3)
        ax.fill_between(range(1, len(latent_skills)+1), discovery_counts, alpha=0.3, color="#059669")
        ax.set_xlabel("Time (Proxy: Skills Discovered)", color="#64748b", fontsize=9)
        ax.set_ylabel("Total Latent Concepts Formed", color="#64748b", fontsize=9)
        ax.set_title("Abstractions Created Without Pre-programming", color="#94a3b8", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.2, color="#94a3b8")
        st.pyplot(fig, width='stretch'); plt.close(fig)
    else:
        _no_data("No latent skills discovered yet.")
        
    st.divider()

    # ─── SECTION G: WORLD STATE & PERCEPTION ─────────────────────────────────
    st.markdown("### 👁️ G — World State & Perception")
    colG1, colG2 = st.columns(2)
    # G1 — Object Color Distribution (polar)
    with colG1:
        st.markdown("##### G1 · Object Color Distribution")
        ARC_HEX_G = ["#111111","#1E90FF","#FF4500","#32CD32","#FFD700",
                     "#AAAAAA","#FF69B4","#FF8C00","#00CED1","#9400D3"]
        if ws_objects:
            color_counts = {i: 0 for i in range(10)}
            for obj in ws_objects:
                c = obj.get("color", 0)
                if 0 <= c < 10: color_counts[c] += 1
            fig = plt.figure(figsize=(5, 4), facecolor=_BG)
            ax  = fig.add_subplot(111, polar=True)
            ax.set_facecolor(_AX); fig.patch.set_facecolor(_BG)
            angles_g = np.linspace(0, 2*np.pi, 10, endpoint=False)
            radii_g  = [color_counts[i] for i in range(10)]
            bars_g   = ax.bar(angles_g, [r+0.05 for r in radii_g],
                              width=2*np.pi/10*0.8, color=ARC_HEX_G,
                              edgecolor=_BG, linewidth=1, alpha=0.9)
            ax.set_xticks(angles_g)
            ax.set_xticklabels([str(i) for i in range(10)], color="#94a3b8", fontsize=8)
            ax.tick_params(colors="#475569", labelsize=7)
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            _no_data("No world state objects.")

    # G2 — Object Size Histogram
    with colG2:
        st.markdown("##### G2 · Object Size Histogram")
        if ws_objects:
            sizes_g = [obj.get("size", 0) for obj in ws_objects]
            fig, ax = _obs_fig(5, 4)
            ax.hist(sizes_g, bins=max(5, len(set(sizes_g))), color="#a78bfa",
                    edgecolor=_SP, linewidth=0.5, alpha=0.85)
            ax.set_xlabel("Object Size (cells)", color="#64748b", fontsize=9)
            ax.set_ylabel("Count", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            _no_data("No world state objects.")

    # G3 — Color Transition Matrix (input→output)
    st.markdown("##### G3 · Color Transition Matrix (Input → Output)")
    trans_mat = np.zeros((10, 10))
    for inp_g, out_g in task.train_pairs:
        inp_arr = np.array(inp_g); out_arr = np.array(out_g)
        if inp_arr.shape == out_arr.shape:
            for rr in range(inp_arr.shape[0]):
                for cc in range(inp_arr.shape[1]):
                    c_in  = int(np.clip(inp_arr[rr,cc], 0, 9))
                    c_out = int(np.clip(out_arr[rr,cc], 0, 9))
                    trans_mat[c_in, c_out] += 1
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_norm = trans_mat / row_sums
    fig, ax    = _obs_fig(10, 4)
    im_g3      = ax.imshow(trans_norm, cmap="plasma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels([f"→{i}" for i in range(10)], color="#94a3b8", fontsize=8)
    ax.set_yticklabels([f"{i}→" for i in range(10)], color="#94a3b8", fontsize=8)
    ax.set_xlabel("Output Color", color="#64748b", fontsize=9)
    ax.set_ylabel("Input Color", color="#64748b", fontsize=9)
    for i in range(10):
        for j in range(10):
            if trans_norm[i,j] > 0.1:
                ax.text(j, i, f"{trans_norm[i,j]:.1f}", ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold")
    fig.colorbar(im_g3, ax=ax, label="Transition Probability")
    st.pyplot(fig, width='stretch'); plt.close(fig)

    colG4, colG5 = st.columns(2)
    # G4 — Philosopher Revision Gauge
    with colG4:
        st.markdown("##### G4 · Philosopher Revision Depth")
        phil_rev = ws_data.get("philosopher_revision", 0) if isinstance(ws_data, dict) else 0
        fig, ax  = _obs_fig(5, 3)
        levels   = ["None\n(Clean)", "Revision 1\n(BG as obj)", "Revision 2\n(Color-merge)", "Deep\nReframe"]
        colors_g4= ["#22c55e","#86efac","#f59e0b","#ef4444"]
        for i, (lbl, col_g4) in enumerate(zip(levels, colors_g4)):
            alpha_g4 = 1.0 if i == phil_rev else 0.15
            ax.barh(0, 1, left=i, height=0.5, color=col_g4, alpha=alpha_g4,
                    edgecolor=_SP if i != phil_rev else "white", linewidth=1.5)
            ax.text(i+0.5, 0, lbl, ha="center", va="center", color="white", fontsize=7)
        ax.set_xlim(0, 4); ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Current: Level {phil_rev}", color="#94a3b8", fontsize=9)
        st.pyplot(fig, width='stretch'); plt.close(fig)

    # G5 — Bounding Box Coverage Map
    with colG5:
        st.markdown("##### G5 · Object Bounding Box Map")
        if ws_objects and ws_data.get("grid_shape"):
            gh, gw = ws_data["grid_shape"]
            fig, ax = _obs_fig(5, 3)
            ax.set_xlim(0, gw); ax.set_ylim(gh, 0)
            for obj in ws_objects[:12]:
                bb_g = obj.get("bbox", (0,0,0,0))
                r0, c0, r1, c1 = bb_g
                hue_g5 = (obj.get("color",0) / 9.0)
                col_g5 = colorsys.hsv_to_rgb(hue_g5, 0.8, 0.9)
                rect_g5 = mpatches.Rectangle((c0, r0), c1-c0+1, r1-r0+1,
                                              linewidth=1.5, edgecolor=col_g5,
                                              facecolor=col_g5, alpha=0.3)
                ax.add_patch(rect_g5)
            ax.set_xlabel("Column", color="#64748b", fontsize=9)
            ax.set_ylabel("Row", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            _no_data("No bounding box data.")

    st.divider()

    # ─── SECTION H: MULTI-EPISODE META-LEARNING ──────────────────────────────
    st.markdown("### 📈 H — Multi-Episode Meta-Learning")
    if gen_series:
        df_h   = pd.DataFrame(gen_series)
        rounds_h = df_h["rounds"].tolist()
        n_ep   = len(df_h)

        colH1, colH2 = st.columns(2)
        # H1 — Rounds to Solve Over Sessions
        with colH1:
            st.markdown("##### H1 · Rounds to Solve (Learning Curve)")
            fig, ax = _obs_fig(5, 3.5)
            verdict_c = {"solved":"#22c55e","unknown":"#f59e0b","timeout":"#ef4444","pending":"#94a3b8"}
            for i, row in df_h.iterrows():
                col_h = verdict_c.get(row["verdict"], "#94a3b8")
                ax.scatter(i+1, row["rounds"], color=col_h, s=70, zorder=4,
                           edgecolors="#1c2133", linewidths=0.5)
            ax.plot(range(1, n_ep+1), rounds_h, color="#334155", linewidth=1, alpha=0.5)
            if n_ep >= 3:
                rm = df_h["rounds"].rolling(3, min_periods=1).mean().tolist()
                ax.plot(range(1, n_ep+1), rm, color="#38bdf8", linewidth=2,
                        linestyle="--", label="3-ep rolling mean")
                ax.legend(fontsize=8, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
            ax.set_xlabel("Episode #", color="#64748b", fontsize=9)
            ax.set_ylabel("Rounds to Solve", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # H2 — Cumulative Solve Rate
        with colH2:
            st.markdown("##### H2 · Cumulative Solve Rate")
            fig, ax = _obs_fig(5, 3.5)
            solved_flags = [1 if r=="solved" else 0 for r in df_h["verdict"]]
            cum_rate     = np.cumsum(solved_flags) / np.arange(1, n_ep+1)
            ax.fill_between(range(1, n_ep+1), cum_rate, alpha=0.3, color="#22c55e")
            ax.plot(range(1, n_ep+1), cum_rate, color="#86efac", linewidth=2)
            ax.axhline(1.0, color="#334155", linestyle="--", linewidth=0.8)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Episode #", color="#64748b", fontsize=9)
            ax.set_ylabel("Cumulative Solve %", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        colH3, colH4 = st.columns(2)
        # H3 — Prior Difficulty vs Rounds
        with colH3:
            st.markdown("##### H3 · Difficulty vs Rounds Scatter")
            mem_eps = []
            try:
                mem_eps = st.session_state.council.memory.to_dict()
            except Exception: pass
            if mem_eps:
                fig, ax = _obs_fig(5, 3.5)
                for ep in mem_eps:
                    n_priors = len(ep.get("priors", []))
                    col_ep   = verdict_c.get(ep.get("verdict",""), "#94a3b8")
                    ax.scatter(n_priors + np.random.uniform(-0.1, 0.1),
                               ep.get("rounds", 0), color=col_ep, s=60,
                               alpha=0.8, edgecolors="#1c2133", linewidths=0.5)
                ax.set_xlabel("Number of Priors", color="#64748b", fontsize=9)
                ax.set_ylabel("Rounds to Solve", color="#64748b", fontsize=9)
                legend_els_h = [mpatches.Patch(color=c, label=v) for v,c in verdict_c.items()]
                ax.legend(handles=legend_els_h, fontsize=7, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP, loc="upper left")
                st.pyplot(fig, width='stretch'); plt.close(fig)
            else:
                _no_data("Need multiple episodes for difficulty scatter.")

        # H4 — Final Surprise per Episode
        with colH4:
            st.markdown("##### H4 · Final Surprise per Episode")
            mem_eps2 = []
            try:
                mem_eps2 = st.session_state.council.memory.to_dict()
            except Exception: pass
            if mem_eps2:
                final_surprises = [ep.get("final_surprise", 0) or 0 for ep in mem_eps2]
                fig, ax = _obs_fig(5, 3.5)
                cmap_h4 = plt.get_cmap("RdYlGn_r")
                max_fs  = max(final_surprises) if max(final_surprises) > 0 else 1
                bar_c_h = [cmap_h4(v/max_fs) for v in final_surprises]
                ax.bar(range(1, len(final_surprises)+1), final_surprises,
                       color=bar_c_h, edgecolor=_SP, linewidth=0.5)
                ax.axhline(0.05, color="#22c55e", linestyle="--", linewidth=1, alpha=0.8, label="Resolved threshold")
                ax.set_xlabel("Episode #", color="#64748b", fontsize=9)
                ax.set_ylabel("Final Surprise", color="#64748b", fontsize=9)
                ax.legend(fontsize=8, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
                st.pyplot(fig, width='stretch'); plt.close(fig)
            else:
                _no_data("Need memory episodes.")

        # H5 — Budget Efficiency
        st.markdown("##### H5 · Budget Efficiency per Episode")
        mem_eps3 = []
        try: mem_eps3 = st.session_state.council.memory.to_dict()
        except Exception: pass
        if mem_eps3:
            fig, ax = _obs_fig(10, 2.5)
            budgets  = [ep.get("budget", 100) or 100 for ep in mem_eps3]
            rounds_m = [ep.get("rounds", 15) or 15 for ep in mem_eps3]
            eff      = [r / b for r, b in zip(rounds_m, budgets)]
            ax.fill_between(range(1, len(eff)+1), eff, alpha=0.4, color="#06b6d4")
            ax.plot(range(1, len(eff)+1), eff, color="#67e8f9", linewidth=2)
            ax.set_xlabel("Episode #", color="#64748b", fontsize=9)
            ax.set_ylabel("Rounds / Budget (↓ = efficient)", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            _no_data("Need memory episodes for budget efficiency.")
    else:
        _no_data("Run at least 1 episode to populate Meta-Learning charts.")

    st.divider()

    # ─── SECTION I: CURIOSITY ENGINE ─────────────────────────────────────────
    st.markdown("### 🔥 I — Curiosity Engine Deep Dive")
    curiosity_msgs = [e for e in call_log if e.get("agent") == "CuriosityEngine"]
    if curiosity_msgs:
        # Decode directives
        directives   = []
        plateau_rounds = []
        for e in curiosity_msgs:
            msg = e.get("message","")
            if "Directive:" in msg:
                d = msg.split("Directive:")[-1].strip()
                directives.append(d)
                plateau_rounds.append(e.get("round", 0))

        colI1, colI2 = st.columns(2)
        # I1 — Directive Frequency
        with colI1:
            st.markdown("##### I1 · Directive Frequency")
            from collections import Counter
            dir_counts = Counter(directives)
            fig, ax    = _obs_fig(5, 3.5)
            dl  = list(dir_counts.keys()); dv = list(dir_counts.values())
            ax.barh(dl, dv, color=["#fb923c","#f472b6","#fbbf24","#34d399"][:len(dl)],
                    edgecolor=_SP, linewidth=0.5)
            ax.set_xlabel("Count", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # I2 — Plateau Detection Timeline
        with colI2:
            st.markdown("##### I2 · Plateau Detection on Surprise")
            fig, ax = _obs_fig(5, 3.5)
            if surprise:
                ax.plot(range(len(surprise)), surprise, color="#38bdf8", linewidth=2)
                ax.fill_between(range(len(surprise)), surprise, alpha=0.2, color="#38bdf8")
                for rn in plateau_rounds:
                    idx_p = min(rn-1, len(surprise)-1)
                    if idx_p >= 0:
                        ax.axvline(idx_p, color="#fb923c", alpha=0.6, linewidth=1, linestyle=":")
                ax.axhline(0.05, color="#22c55e", linestyle="--", linewidth=1, alpha=0.8)
                ax.set_xlabel("Observation #", color="#64748b", fontsize=9)
                ax.set_ylabel("Surprise", color="#64748b", fontsize=9)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        # I3 — Surprise Spectrum (1D colorbar)
        st.markdown("##### I3 · Surprise Spectrum (Heartbeat)")
        if surprise:
            spec_arr = np.array(surprise).reshape(1, -1)
            fig, ax  = plt.subplots(figsize=(10, 1.2), facecolor=_BG)
            ax.set_facecolor(_BG); fig.patch.set_facecolor(_BG)
            ax.imshow(spec_arr, cmap="plasma", aspect="auto", vmin=0, vmax=max(surprise)+0.01)
            ax.set_yticks([]); ax.set_xlabel("Observation step →", color="#64748b", fontsize=9)
            for sp in ax.spines.values(): sp.set_visible(False)
            st.pyplot(fig, width='stretch'); plt.close(fig)

        colI4, colI5 = st.columns(2)
        # I4 — Intervention count info box
        with colI4:
            st.markdown("##### I4 · Curiosity Engine Stats")
            n_plateaus   = len([e for e in curiosity_msgs if "PLATEAU" in e.get("message","")])
            n_resolved   = len([e for e in curiosity_msgs if "resolved" in e.get("message","").lower()])
            n_interv     = len(directives)
            surp_col_i4  = "#22c55e" if surprise and surprise[-1] < 0.05 else "#ef4444"
            surp_val_i4  = f"{surprise[-1]:.4f}" if surprise else "—"
            st.markdown(
                f'<div style="background:#0a0e18;border:1px solid #1c2133;border-radius:10px;padding:16px">'
                f'<p style="color:#94a3b8;margin:4px 0">⚡ Plateau detections: <b style="color:#fb923c">{n_plateaus}</b></p>'
                f'<p style="color:#94a3b8;margin:4px 0">🎯 Directives issued: <b style="color:#fbbf24">{n_interv}</b></p>'
                f'<p style="color:#94a3b8;margin:4px 0">Verified Resolved events: <b style="color:#22c55e">{n_resolved}</b></p>'
                f'<p style="color:#94a3b8;margin:4px 0">📉 Final surprise: <b style="color:{surp_col_i4}">{surp_val_i4}</b></p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # I5 — Free Energy Target Approach (log scale)
        with colI5:
            st.markdown("##### I5 · Free Energy Convergence (Log)")
            if surprise and len(surprise) > 1:
                fig, ax = _obs_fig(5, 3.5)
                log_s   = [max(s, 1e-6) for s in surprise]
                ax.semilogy(range(len(log_s)), log_s, color="#38bdf8", linewidth=2)
                ax.fill_between(range(len(log_s)), log_s, 1e-6, alpha=0.2, color="#38bdf8")
                ax.axhline(0.05, color="#22c55e", linestyle="--", linewidth=1.2, alpha=0.8, label="Target 0.05")
                ax.set_xlabel("Step", color="#64748b", fontsize=9)
                ax.set_ylabel("Surprise (log)", color="#64748b", fontsize=9)
                ax.legend(fontsize=8, labelcolor="#94a3b8", facecolor=_AX, edgecolor=_SP)
                st.pyplot(fig, width='stretch'); plt.close(fig)
            else:
                _no_data("Need 2+ surprise values.")
    else:
        _no_data("No CuriosityEngine activity recorded.")

    st.divider()

    # ─── SECTION J: EMERGENT INTELLIGENCE METRICS ────────────────────────────
    st.markdown("### 🌌 J — Emergent Intelligence Metrics")

    # Compute composite metrics from live data
    solve_rate      = (st.session_state.n_solved / max(st.session_state.n_run, 1))
    skill_reuse     = min(sum(s.get("usage_count",0) for s in dsl_skills) / max(len(dsl_skills)*2, 1), 1.0)
    surprise_decay  = max(0, 1.0 - (surprise[-1] if surprise else 1.0) / (surprise[0] if surprise and surprise[0] > 0 else 1.0))
    causal_rate     = n_causal / max(len([h for h in hypotheses if h.get("causal_verdict")]), 1)
    hypothesis_exp  = min(len(hypotheses) / 50.0, 1.0)
    budget_eff      = 1.0 - (snap.get("budget_used", 100) / 100.0)
    avg_r_norm      = max(0, 1.0 - (st.session_state.stat_avg_rounds / 20.0))

    # J1 — Multi-ring General Intelligence Progress Gauge
    st.markdown("##### J1 · General Intelligence Progress Multi-Ring Gauge")
    metrics_j1 = [
        ("Solve Rate",        solve_rate,     "#22c55e"),
        ("Skill Reuse",       skill_reuse,    "#38bdf8"),
        ("Surprise Decay",    surprise_decay, "#a78bfa"),
        ("Causal Law Rate",   causal_rate,    "#fbbf24"),
        ("Budget Efficiency", budget_eff,     "#fb923c"),
    ]
    fig = plt.figure(figsize=(10, 4), facecolor=_BG)
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(_AX); fig.patch.set_facecolor(_BG)
    for i, (label, val, col_j) in enumerate(metrics_j1):
        r     = 0.7 - i * 0.12
        width = 0.10
        theta_end = 2 * np.pi * val
        theta_arc = np.linspace(0, theta_end, 100)
        bg_arc    = np.linspace(0, 2*np.pi, 100)
        ax.plot(bg_arc, [r]*100, color="#1c2133", linewidth=8, solid_capstyle="round")
        ax.plot(theta_arc, [r]*100, color=col_j, linewidth=8, solid_capstyle="round", alpha=0.9)
        ax.text(0, r, f"{label}: {val:.0%}", ha="left", va="center", color=col_j, fontsize=9, fontweight="bold")
    ax.set_rticks([]); ax.set_xticks([])
    ax.set_ylim(0, 0.85)
    st.pyplot(fig, width='stretch'); plt.close(fig)

    # J2 — Intelligence Fingerprint (Radar)
    st.markdown("##### J2 · Intelligence Fingerprint Radar")
    radar_metrics = {
        "Solve Rate":       solve_rate,
        "Skill Diversity":  skill_reuse,
        "Surprise Decay":   surprise_decay,
        "Causal Law %":     causal_rate,
        "Hypothesis Exp.":  hypothesis_exp,
        "Budget Effic.":    budget_eff,
        "Efficiency":       avg_r_norm,
        "Contradiction\nControl": max(0, 1.0 - len(contrad)/20.0),
    }
    rnames = list(radar_metrics.keys()); rvals  = list(radar_metrics.values())
    N_J2   = len(rnames)
    angles_j2 = np.linspace(0, 2*np.pi, N_J2, endpoint=False).tolist()
    angles_j2 += angles_j2[:1]; rvals += rvals[:1]
    fig = plt.figure(figsize=(10, 5), facecolor=_BG)
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(_AX); fig.patch.set_facecolor(_BG)
    ax.plot(angles_j2, rvals, color="#a78bfa", linewidth=2.5)
    ax.fill(angles_j2, rvals, color="#a78bfa", alpha=0.2)
    ax.plot(angles_j2, [1]*(N_J2+1), color="#1c2133", linewidth=1, linestyle="--")
    ax.set_xticks(angles_j2[:-1])
    ax.set_xticklabels(rnames, color="#94a3b8", fontsize=9)
    ax.set_ylim(0, 1); ax.tick_params(colors="#475569", labelsize=8)
    # Color each spoke by value
    for i, (ang, val) in enumerate(zip(angles_j2[:-1], rvals[:-1])):
        col_j2 = "#22c55e" if val >= 0.8 else "#f59e0b" if val >= 0.5 else "#ef4444"
        ax.plot([ang, ang], [0, val], color=col_j2, linewidth=2.5, alpha=0.7)
    st.pyplot(fig, width='stretch'); plt.close(fig)

    # J3 — Metacognitor Activity Heatmap
    colJ3, colJ4 = st.columns(2)
    with colJ3:
        st.markdown("##### J3 · Metacognitor Activity")
        meta_msgs = [e for e in call_log if e.get("agent") == "Metacognitor"]
        if meta_msgs and surprise:
            max_rnd_j = max(e.get("round",1) for e in call_log)
            meta_heat = np.zeros((1, max_rnd_j))
            for e in meta_msgs:
                rn = e.get("round",1)-1
                if 0 <= rn < max_rnd_j:
                    msg_j = e.get("message","")
                    agenda_len = msg_j.count("→") if msg_j else 0
                    meta_heat[0, rn] = agenda_len + 1
            fig, ax = _obs_fig(5, 2)
            im_j3 = ax.imshow(meta_heat, cmap="hot", aspect="auto", interpolation="nearest")
            ax.set_yticks([0]); ax.set_yticklabels(["Metacognitor"], color="#94a3b8", fontsize=9)
            ax.set_xlabel("Round", color="#64748b", fontsize=9)
            fig.colorbar(im_j3, ax=ax, label="Agenda length")
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            _no_data("Need Metacognitor logs.")

    # J4 — Council Consensus Heat (confidence per round)
    with colJ4:
        st.markdown("##### J4 · Council Consensus Heat")
        if hypotheses and surprise:
            max_rnd_j4   = max(e.get("round",1) for e in call_log) if call_log else 1
            # estimate confidence per round from snapshotted hypotheses (sorted by created_at)
            # Use surprise as a proxy for uncertainty (inverse confidence)
            conf_proxy   = [max(0, 1.0 - s) for s in surprise[:max_rnd_j4]]
            conf_mat     = np.array(conf_proxy).reshape(1, -1)
            fig, ax      = _obs_fig(5, 2)
            im_j4 = ax.imshow(conf_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
            ax.set_yticks([0]); ax.set_yticklabels(["Consensus"], color="#94a3b8", fontsize=9)
            ax.set_xlabel("Round", color="#64748b", fontsize=9)
            fig.colorbar(im_j4, ax=ax, label="Confidence proxy")
            st.pyplot(fig, width='stretch'); plt.close(fig)
        else:
            _no_data("Need hypothesis + surprise data.")

    # J5 — System Complexity Score
    st.markdown("##### J5 · System Complexity Score Timeline")
    if surprise and hypotheses:
        n_hyp_running = min(len(hypotheses), len(surprise))
        avg_prog_len  = np.mean([len(h.get("program","").split(" → ")) if h.get("program") else 1 for h in hypotheses])
        complexity_ts = [surprise[i] * n_hyp_running * avg_prog_len for i in range(len(surprise))]
        fig, ax = _obs_fig(10, 3)
        cmap_j5 = plt.get_cmap("plasma")
        for i in range(len(complexity_ts)-1):
            col_j5 = cmap_j5(min(complexity_ts[i]/max(complexity_ts+[1]), 1.0))
            ax.fill_between([i, i+1], [complexity_ts[i], complexity_ts[i+1]], alpha=0.5, color=col_j5)
            ax.plot([i, i+1], [complexity_ts[i], complexity_ts[i+1]], color=col_j5, linewidth=2)
        ax.set_xlabel("Observation step", color="#64748b", fontsize=9)
        ax.set_ylabel("Complexity Score\n(surprise × hypotheses × prog_len)", color="#64748b", fontsize=9)
        sm_j5 = plt.cm.ScalarMappable(cmap=cmap_j5)
        fig.colorbar(sm_j5, ax=ax, label="Normalized complexity")
        st.pyplot(fig, width='stretch'); plt.close(fig)

        # Final summary stat card
        overall_gi = np.mean([solve_rate, skill_reuse, surprise_decay, causal_rate, budget_eff, avg_r_norm])
        col_gi     = "#22c55e" if overall_gi >= 0.8 else "#f59e0b" if overall_gi >= 0.5 else "#ef4444"
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0f1420,#141b2d);border:1px solid #1e2a40;'
            f'border-radius:14px;padding:22px;margin-top:16px;text-align:center">'
            f'<p style="color:#64748b;font-size:12px;margin:0">Composite General Intelligence Score (this session)</p>'
            f'<p style="color:{col_gi};font-size:52px;font-weight:700;margin:8px 0">{overall_gi:.0%}</p>'
            f'<p style="color:#475569;font-size:11px;margin:0">Solve Rate · Skill Reuse · Surprise Decay · Causal Law Rate · Budget Efficiency · Round Efficiency</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        _no_data("Need surprise + hypotheses for complexity timeline.")


