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

from universe import Universe, ARCTask, DifficultyLevel
from council import Council, DSL

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neuro-Symbolic Collective — AGI Lab",
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
        st.session_state.task          = None   # current ARCTask object
        st.session_state.snap          = None   # final blackboard snapshot dict
        st.session_state.all_logs      = []     # _emit() entries for display
        st.session_state.n_run         = 0
        st.session_state.n_solved      = 0
        # Cached stats — updated right after each run so they're never stale
        st.session_state.stat_avg_rounds     = 0.0
        st.session_state.stat_skills         = 15
        st.session_state.stat_gen_series     = []
        st.session_state.stat_dsl_skills     = []

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
        "solved":  ("badge-solved",   "#14532d", "#86efac", "✅"),
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
    # 2. Re-execute winning program on test input
    prog_str = _winning_program(snap)
    if prog_str:
        try:
            prims = [p.strip() for p in prog_str.split("→")]
            return DSL.execute(task.test_input, prims)
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
    run_btn = st.button("⚡ Run Council", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Session Stats**")
    c1, c2 = st.columns(2)
    c1.metric("Tasks Run",  st.session_state.n_run)
    c2.metric("Solved",     st.session_state.n_solved)

    c3, c4 = st.columns(2)
    # Read from cached stats — updated right after each run
    avg_r = st.session_state.stat_avg_rounds
    c3.metric("Avg Rounds", f"{avg_r:.1f}" if avg_r else "—")
    c4.metric("Skills",     st.session_state.stat_skills)

    st.divider()
    st.caption("0-Cheat · Zero Memorisation · Full Transparency")

    import json
    if st.session_state.n_run > 0:
        export_data = {
            "seed": st.session_state.seed,
            "tasks_run": st.session_state.n_run,
            "solved": st.session_state.n_solved,
            "avg_rounds": st.session_state.stat_avg_rounds,
            "skills": st.session_state.stat_dsl_skills,
            "generalization": st.session_state.stat_gen_series,
            "final_logs": st.session_state.all_logs
        }
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="💾 Download Session Data",
            data=json_str,
            file_name=f"agi_session_{st.session_state.seed}.json",
            mime="application/json",
            use_container_width=True
        )

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("#  The Neuro-Symbolic Collective")
st.markdown("**9-Agent AGI Research System — Inference-Time Discovery on ARC-AGI-2**")

mc = st.columns(4)
mc[0].metric("SOTA 2026 (Gemini 3 Deep Think)", "84.6%")
mc[1].metric("Human Baseline", "~80%")
mc[2].metric("Our Goal", "Sample Efficiency + Transparency")
mc[3].metric("0-Cheat", "✅ Enforced")
st.divider()

# ─── RUN THE COUNCIL ──────────────────────────────────────────────────────────
if run_btn:
    task = st.session_state.universe.generate_task(chosen_level)
    st.session_state.task = task
    st.session_state.snap = None
    st.session_state.all_logs = []
    st.session_state.n_run += 1

    progress = st.progress(0, text="Council deliberating…")
    status_box = st.empty()

    all_snapshots = []
    for snap in st.session_state.council.solve(task):
        all_snapshots.append(snap)
        bud = snap.get("budget_used", 0)
        rnd = snap.get("round", 0)
        verdict = snap.get("final_verdict", "pending")
        progress.progress(min(bud / 100, 1.0),
                          text=f"Round {rnd} | Budget {bud}/100 | {verdict}")

    progress.empty()
    status_box.empty()

    if all_snapshots:
        final = all_snapshots[-1]
        st.session_state.snap = final

        # Only show entries with a real human-readable message (from _emit, not _log)
        raw_log = final.get("agent_call_log", [])
        st.session_state.all_logs = [
            e for e in raw_log if e.get("message", "").strip()
        ]

        if final.get("final_verdict") == "solved":
            st.session_state.n_solved += 1

    # ── Snapshot stats into session_state immediately so they can never be stale ──
    cs = st.session_state.council.stats()
    st.session_state.stat_avg_rounds  = cs["avg_rounds"]
    st.session_state.stat_skills      = cs["skill_library_size"]
    st.session_state.stat_gen_series  = cs["generalization_series"]
    st.session_state.stat_dsl_skills  = cs["dsl_skills"]

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
    | 🔬 **Program Inspector** | The discovered DSL rule in human-readable pseudocode |
    | 🔴 **Skeptic's Dossier** | Every falsified hypothesis — proof of depth |
    | 📉 **Generalization Curve** | Rounds-to-solve over time — the AGI proof |
    | 📚 **Skill Library** | The growing vocabulary of discovered primitives |
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(f"Train {idx + 1}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
if snap is None:
    st.info("Run the Council to see results.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏛️ Council Chamber",
    "⚡ Surprise Metric",
    "🔬 Program Inspector",
    "🔴 Skeptic's Dossier",
    "📉 Generalization Curve",
    "📚 Skill Library",
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
        panels.append((task.test_output, "Ground Truth ✓"))

        fig = _grid_fig(panels, cols=len(panels), cell_size=3.0)
        st.pyplot(fig, use_container_width=True)
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Stats row
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Initial Surprise", f"{surprise[0]:.3f}")
        sc2.metric("Final Surprise",   f"{surprise[-1]:.3f}")
        sc3.metric("Peak Surprise",    f"{max(surprise):.3f}")
        sc4.metric("Resolved?",        "✅ Yes" if surprise[-1] < 0.05 else "❌ No")

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
        st.markdown("##### Discovered Rule (DSL Program)")
        pi1, pi2, pi3, pi4 = st.columns(4)
        pi1.metric("Program",    prog_str[:30] + ("…" if len(prog_str) > 30 else ""))
        pi2.metric("MDL Score",  f"{winning_h['mdl_score']:.2f}" if winning_h and winning_h.get("mdl_score") else "—")
        pi3.metric("Causal",     winning_h.get("causal_verdict","—") if winning_h else "—")
        pi4.metric("Confidence", f"{winning_h['confidence']:.3f}" if winning_h else "—")

        st.markdown("---")
        steps = [s.strip() for s in prog_str.split("→")]
        st.markdown("**Execution steps:**")
        for i, step in enumerate(steps, 1):
            icon_map = {
                "rotate90": "🔄", "rotate180": "🔄", "rotate270": "🔄",
                "mirror_h": "↔️", "mirror_v": "↕️",
                "gravity_down": "⬇️", "gravity_up": "⬆️",
                "majority_recolor": "🎨", "sort_by_size": "📊",
                "identity": "⬜",
            }
            icon = icon_map.get(step, "▶️")
            st.markdown(
                f'<div style="background:#0d1421;border:1px solid #1e2a40;border-radius:8px;'
                f'padding:10px 16px;margin:4px 0">'
                f'<span style="color:#64748b;font-size:11px">Step {i}</span>&nbsp;&nbsp;'
                f'{icon}&nbsp;<code style="color:#7dd3fc;font-size:14px;'
                f'background:transparent;font-weight:600">{step}</code></div>',
                unsafe_allow_html=True
            )

        # Show its effect on first training pair
        st.markdown("---")
        st.markdown("**Program applied to training example:**")
        prims = [s.strip() for s in prog_str.split("→")]
        try:
            inp_ex, out_ex = task.train_pairs[0]
            pred_ex = DSL.execute(inp_ex, prims)
            match = "✅ Exact match" if np.array_equal(pred_ex, out_ex) else "❌ Mismatch"
            fig = _grid_fig([
                (inp_ex,  "Training Input"),
                (pred_ex, f"Program Output ({match})"),
                (out_ex,  "Ground Truth"),
            ], cols=3, cell_size=2.8)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
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
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
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
        st.pyplot(fig, use_container_width=True)
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
            use_container_width=True,
        )
    else:
        st.info("Run at least 1 task to populate the generalization curve.")

# ── TAB 6: SKILL LIBRARY ─────────────────────────────────────────────────────
with tab6:
    skills = st.session_state.stat_dsl_skills  # read from cached session state
    if skills:
        sl1, sl2 = st.columns(2)
        total_skills   = len(skills)
        used_skills    = sum(1 for s in skills if s.get("usage_count", 0) > 0)
        discovered     = sum(1 for s in skills if s.get("origin", "") != "BUILTIN")
        sl1.metric("Total Primitives",    total_skills)
        sl2.metric("Discovered This Session", discovered)

        # Usage bar chart
        top_skills = sorted(skills, key=lambda s: s.get("usage_count", 0), reverse=True)[:12]
        if any(s.get("usage_count", 0) > 0 for s in top_skills):
            fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#07090f")
            ax.set_facecolor("#0a0e18")
            names  = [s["name"] for s in top_skills]
            counts = [s.get("usage_count", 0) for s in top_skills]
            bar_colors = ["#6d28d9" if s.get("origin") != "BUILTIN" else "#1e40af" for s in top_skills]
            bars = ax.bar(names, counts, color=bar_colors, edgecolor="#1c2133", linewidth=0.8)
            ax.set_ylabel("Usage Count", color="#64748b", fontsize=10)
            ax.tick_params(axis="x", colors="#94a3b8", labelsize=9, rotation=30)
            ax.tick_params(axis="y", colors="#475569", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1c2133")
            legend_els = [
                mpatches.Patch(color="#6d28d9", label="Discovered this session"),
                mpatches.Patch(color="#1e40af", label="Built-in primitive"),
            ]
            ax.legend(handles=legend_els, fontsize=9, labelcolor="#94a3b8",
                      facecolor="#0d1017", edgecolor="#1c2133")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("---")
        st.markdown("**Full Library**")
        df_skills = pd.DataFrame(skills)[[
            "name", "usage_count", "description", "code", "origin"
        ]]
        st.dataframe(
            df_skills,
            hide_index=True,
            use_container_width=True,
            column_config={
                "name":        st.column_config.TextColumn("Primitive", width=130),
                "usage_count": st.column_config.NumberColumn("Uses", width=60),
                "description": st.column_config.TextColumn("Description"),
                "code":        st.column_config.TextColumn("Pseudocode", width=200),
                "origin":      st.column_config.TextColumn("Discovered In", width=120),
            }
        )
    else:
        st.info("No skills in library yet.")
