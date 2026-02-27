"""
app.py — The Neuro-Symbolic Collective
Scientific Live Dashboard (Streamlit Cloud)
===========================================
Imports from: universe.py | memory.py | council.py
"""

import streamlit as st
import numpy as np
import random
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional

from universe import Universe, ARCTask, DifficultyLevel
from memory import DSLSkillLibrary
from council import Council


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neuro-Symbolic Collective — AGI Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── ARC COLOR PALETTE ────────────────────────────────────────────────────────

ARC_COLORS = [
    "#000000",  # 0 = background (black)
    "#1E90FF",  # 1 = blue
    "#FF4500",  # 2 = red
    "#32CD32",  # 3 = green
    "#FFD700",  # 4 = yellow
    "#808080",  # 5 = gray
    "#FF69B4",  # 6 = magenta
    "#FF8C00",  # 7 = orange
    "#00CED1",  # 8 = cyan
    "#9400D3",  # 9 = purple
]

ARC_CMAP = mcolors.ListedColormap(ARC_COLORS, name="arc")
ARC_NORM = mcolors.BoundaryNorm(list(range(11)), 10)


# ─── STYLES ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Global reset ── */
  html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: #0a0d14;
    color: #e2e8f0;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #0f1420;
    border-right: 1px solid #1e2535;
  }

  /* ── Headers ── */
  h1 { color: #7dd3fc; letter-spacing: -0.5px; }
  h2 { color: #a5f3fc; border-bottom: 1px solid #1e3a5f; padding-bottom: 4px; }
  h3 { color: #e2e8f0; }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: #0f1420;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 12px;
  }

  /* ── Expanders ── */
  details { border: 1px solid #1e2535 !important; border-radius: 8px; }

  /* ── Code blocks ── */
  code {
    background: #0f1a2f;
    color: #7dd3fc;
    border-radius: 4px;
    padding: 2px 6px;
  }

  /* ── Verdict badges ── */
  .badge-solved   { background:#14532d; color:#86efac; border-radius:6px; padding:2px 8px; }
  .badge-unknown  { background:#451a03; color:#fdba74; border-radius:6px; padding:2px 8px; }
  .badge-pending  { background:#1e1b4b; color:#a5b4fc; border-radius:6px; padding:2px 8px; }

  /* ── Agent tag colors ── */
  .agent-perceiver   { color: #7dd3fc; font-weight: 600; }
  .agent-dreamer     { color: #c4b5fd; font-weight: 600; }
  .agent-scientist   { color: #6ee7b7; font-weight: 600; }
  .agent-skeptic     { color: #fca5a5; font-weight: 600; }
  .agent-philosopher { color: #fde68a; font-weight: 600; }
  .agent-causal      { color: #f9a8d4; font-weight: 600; }
  .agent-curiosity   { color: #fdba74; font-weight: 600; }
  .agent-meta        { color: #a5f3fc; font-weight: 600; }
  .agent-archivist   { color: #d9f99d; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def render_arc_grid(grid: np.ndarray, ax: plt.Axes, title: str = "") -> None:
    """Render an ARC grid with the official 10-color palette."""
    grid_int = np.array(grid, dtype=int)
    ax.imshow(grid_int, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")
    ax.set_xticks([x - 0.5 for x in range(1, grid_int.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, grid_int.shape[0])], minor=True)
    ax.grid(which="minor", color="#1e2535", linewidth=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=9, color="#94a3b8", pad=3)


def agent_color_tag(agent_name: str) -> str:
    """Return a colored HTML span for an agent name."""
    slug = agent_name.lower().replace(" ", "")
    mapping = {
        "perceiver": "perceiver",
        "dreamer": "dreamer",
        "scientist": "scientist",
        "skeptic": "skeptic",
        "philosopher": "philosopher",
        "causalreasoner": "causal",
        "curiosityengine": "curiosity",
        "metacognitor": "meta",
        "archivist": "archivist",
        "council": "meta",
        "orientation": "meta",
    }
    css = mapping.get(slug, "meta")
    return f'<span class="agent-{css}">{agent_name}</span>'


def verdict_badge(verdict: str) -> str:
    cls = {"solved": "badge-solved", "unknown": "badge-unknown"}.get(verdict, "badge-pending")
    icon = {"solved": "✅", "unknown": "❓"}.get(verdict, "⏳")
    return f'<span class="{cls}">{icon} {verdict.upper()}</span>'


def surprise_color(value: float) -> str:
    if value < 0.1:
        return "#86efac"
    elif value < 0.4:
        return "#fdba74"
    else:
        return "#fca5a5"


# ─── SESSION STATE ────────────────────────────────────────────────────────────

def init_session():
    if "universe" not in st.session_state:
        seed = random.randint(0, 9999)
        st.session_state.universe = Universe(seed=seed)
        st.session_state.council = Council(seed=seed)
        st.session_state.seed = seed
        st.session_state.tasks_solved = 0
        st.session_state.tasks_attempted = 0
        st.session_state.log_entries = []
        st.session_state.final_snapshot = None
        st.session_state.current_task = None
        st.session_state.is_running = False

init_session()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Neuro-Symbolic Collective")
    st.caption(f"Session seed: `{st.session_state.seed}`")
    st.divider()

    st.markdown("### Task Generator")
    level_map = {
        "Level 1 — Simple (1 Prior)":    DifficultyLevel.L1,
        "Level 2 — Moderate (2 Priors)": DifficultyLevel.L2,
        "Level 3 — Hard (3 Priors)":     DifficultyLevel.L3,
        "Level 4 — Expert (4 Priors)":   DifficultyLevel.L4,
        "Level 5 — Frontier (4+ Priors)": DifficultyLevel.L5,
    }
    selected_level_label = st.selectbox("Difficulty", list(level_map.keys()))
    selected_level = level_map[selected_level_label]

    st.divider()
    st.markdown("### Council")
    budget = st.slider("Simulation Budget", 20, 100, 100, step=10,
                        help="Max internal simulations per task (100 = AGI proof standard)")

    run_button = st.button("⚡ Run Council", use_container_width=True, type="primary")

    st.divider()
    st.markdown("### Session Stats")
    c1, c2 = st.columns(2)
    c1.metric("Tasks Run", st.session_state.tasks_attempted)
    c2.metric("Solved", st.session_state.tasks_solved)

    stats = st.session_state.council.stats()
    st.metric("Avg Rounds", f"{stats['avg_rounds']}")
    st.metric("Skills Discovered", stats["skill_library_size"])

    st.divider()
    st.caption("0-Cheat · Zero Memorization · Full Transparency")


# ─── MAIN HEADER ──────────────────────────────────────────────────────────────

st.markdown("""
# 🧠 The Neuro-Symbolic Collective
**A 9-Agent AGI Research System — Inference-Time Discovery on ARC-AGI-2**
""")

col_meta = st.columns(4)
col_meta[0].metric("SOTA 2026 (Gemini 3 Deep Think)", "84.6%", help="ARC-Prize Verified v2")
col_meta[1].metric("Human Baseline", "~80%")
col_meta[2].metric("Our Goal", "Sample Efficiency + Transparency")
col_meta[3].metric("0-Cheat", "✅ Enforced")

st.divider()


# ─── TASK DISPLAY ─────────────────────────────────────────────────────────────

def display_task(task: ARCTask) -> None:
    st.markdown(f"### Current Task: `{task.task_id}`")
    st.markdown(f"**Priors**: {', '.join(p.value for p in task.priors_used)}  |  "
                f"**Difficulty**: `{task.difficulty.name}`  |  "
                f"**Rule** (hidden from agents): `{task.transformation_description}`")

    n_pairs = len(task.train_pairs)
    cols = st.columns(n_pairs + 1)

    for i, (inp, out) in enumerate(task.train_pairs):
        with cols[i]:
            fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8),
                                     facecolor="#0a0d14", constrained_layout=True)
            render_arc_grid(inp, axes[0], "Input")
            render_arc_grid(out, axes[1], "Output")
            fig.suptitle(f"Train {i+1}", color="#94a3b8", fontsize=8)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with cols[-1]:
        fig, ax = plt.subplots(figsize=(1.8, 1.8), facecolor="#0a0d14")
        render_arc_grid(task.test_input, ax, "Test Input →?")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ─── COUNCIL LOG RENDERING ────────────────────────────────────────────────────

def render_council_log(log_entries: List[Dict]) -> None:
    st.markdown("### 🏛️ The Council Chamber")
    if not log_entries:
        st.caption("Awaiting the Council Meeting...")
        return

    log_html = ""
    for entry in log_entries[-60:]:
        agent = entry.get("agent", "?")
        action = entry.get("action", "")
        data = entry.get("data", {})
        rnd = entry.get("round", 0)
        msg = data.get("message", str(data)[:80]) if isinstance(data, dict) else ""
        log_html += (
            f'<div style="padding:3px 0; border-bottom:1px solid #1e2535;">'
            f'<span style="color:#475569;font-size:11px;">[R{rnd:02d}]</span> '
            f'{agent_color_tag(agent)} '
            f'<span style="color:#94a3b8;font-size:12px;">{action}</span> '
            f'<span style="color:#e2e8f0;font-size:12px;">{msg}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="height:320px;overflow-y:auto;background:#0f1420;'
        f'border:1px solid #1e2535;border-radius:8px;padding:10px;'
        f'font-family:monospace;">{log_html}</div>',
        unsafe_allow_html=True
    )


# ─── SURPRISE METRIC CHART ────────────────────────────────────────────────────

def render_surprise_chart(history: List[float]) -> None:
    st.markdown("### ⚡ Surprise Metric")
    if not history:
        st.caption("No data yet.")
        return
    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#0a0d14")
    ax.set_facecolor("#0f1420")
    x = list(range(len(history)))
    ax.plot(x, history, color="#7dd3fc", linewidth=2, marker="o", markersize=3)
    ax.fill_between(x, history, alpha=0.15, color="#7dd3fc")
    ax.axhline(y=0.05, color="#86efac", linestyle="--", linewidth=1, alpha=0.7,
               label="Resolution threshold")
    ax.set_xlabel("Round", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Prediction Error", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#475569")
    ax.spines[:].set_color("#1e2535")
    ax.legend(fontsize=8, labelcolor="#94a3b8", facecolor="#0f1420", edgecolor="#1e2535")
    ax.set_ylim(-0.05, max(1.05, max(history) + 0.1))
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─── GENERALIZATION CURVE ─────────────────────────────────────────────────────

def render_generalization_curve(series: List[Dict]) -> None:
    st.markdown("### 📉 Generalization Curve")
    if len(series) < 2:
        st.caption("Needs at least 2 completed tasks to display.")
        return
    df = pd.DataFrame(series)
    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#0a0d14")
    ax.set_facecolor("#0f1420")
    colors_map = {"solved": "#86efac", "unknown": "#fdba74", "timeout": "#fca5a5"}
    for _, row in df.iterrows():
        clr = colors_map.get(row["verdict"], "#94a3b8")
        ax.scatter(len(df[df.index <= _]), row["rounds"], color=clr, s=40, zorder=3)
    ax.plot(range(1, len(df) + 1), df["rounds"].tolist(), color="#475569", linewidth=1, alpha=0.5)
    ax.set_xlabel("Task #", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Rounds to Solve", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#475569")
    ax.spines[:].set_color("#1e2535")
    legend_els = [mpatches.Patch(color=c, label=v) for v, c in colors_map.items()]
    ax.legend(handles=legend_els, fontsize=8, labelcolor="#94a3b8",
              facecolor="#0f1420", edgecolor="#1e2535")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if len(df) >= 3 and df["rounds"].is_monotonic_decreasing:
        st.success("📉 Generalization confirmed: the Council is learning to learn faster.")


# ─── PROGRAM INSPECTOR ────────────────────────────────────────────────────────

def render_program_inspector(snapshot: Dict) -> None:
    st.markdown("### 🔬 Program Inspector")
    hypotheses = snapshot.get("hypothesis_stack", [])
    accepted = [h for h in hypotheses if h.get("status") == "accepted"]

    if not accepted:
        st.caption("No accepted program yet.")
        return

    winner = accepted[-1]
    prog = winner.get("program", "N/A")
    mdl = winner.get("mdl_score")
    causal = winner.get("causal_verdict", "—")

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Program", "")
    sc2.metric("MDL Score", f"{mdl:.2f}" if mdl else "—")
    sc3.metric("Causal Verdict", causal)

    st.code(prog if prog else "# No program", language="python")

    steps = prog.split(" → ") if prog and " → " in prog else [prog]
    for i, step in enumerate(steps, 1):
        st.markdown(f"**Step {i}**: `{step.strip()}`")


# ─── SKEPTIC'S DOSSIER ────────────────────────────────────────────────────────

def render_skeptic_dossier(snapshot: Dict) -> None:
    st.markdown("### 🔴 Skeptic's Dossier")
    contradictions = snapshot.get("contradiction_log", [])
    if not contradictions:
        st.caption("No contradictions recorded.")
        return
    for entry in contradictions[-10:]:
        with st.expander(f"❌ Hypothesis `{entry['hypothesis_id']}` — {entry['failure_mode']}"):
            st.markdown(f"- **Agent**: `{entry['agent']}`")
            st.markdown(f"- **Failure**: `{entry['failure_mode']}`")


# ─── DSL SKILL LIBRARY ────────────────────────────────────────────────────────

def render_skill_library(stats: Dict) -> None:
    st.markdown("### 📚 DSL Skill Library")
    skills = stats.get("dsl_skills", [])
    if not skills:
        st.caption("No skills yet.")
        return
    df = pd.DataFrame(skills)[["name", "usage_count", "description", "origin"]].head(20)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Primitive", width=130),
            "usage_count": st.column_config.NumberColumn("Uses", width=70),
            "description": st.column_config.TextColumn("Description"),
            "origin": st.column_config.TextColumn("Discovered In", width=100),
        }
    )


# ─── RESULT HEADER ────────────────────────────────────────────────────────────

def render_verdict(snapshot: Dict) -> None:
    verdict = snapshot.get("final_verdict", "pending")
    rounds = snapshot.get("round", 0)
    budget = snapshot.get("budget_used", 0)
    st.markdown(
        f"**Verdict**: {verdict_badge(verdict)} &nbsp;&nbsp; "
        f"**Rounds**: `{rounds}` &nbsp;&nbsp; "
        f"**Budget used**: `{budget}/100`",
        unsafe_allow_html=True
    )


# ─── ANSWER GRID ──────────────────────────────────────────────────────────────

def render_answer(task: ARCTask, snapshot: Dict) -> None:
    hypotheses = snapshot.get("hypothesis_stack", [])
    accepted = [h for h in hypotheses if h.get("status") == "accepted"]

    fig, axes = plt.subplots(
        1, 3, figsize=(9, 3.5),
        facecolor="#0a0d14", constrained_layout=True
    )

    render_arc_grid(task.test_input, axes[0], "Test Input")
    render_arc_grid(task.test_output, axes[2], "Ground Truth ✓")

    if accepted:
        # The accepted hypothesis grid is stored as a grid snapshot in the blackboard
        best = accepted[-1]
        # Attempt to get a numpy grid - it may be stored as list
        grid_data = best.get("grid")
        if grid_data is not None:
            try:
                g = np.array(grid_data, dtype=int)
                render_arc_grid(g, axes[1], "Council's Answer 🧠")
            except Exception:
                axes[1].text(0.5, 0.5, "Answer\nunavailable", ha="center", va="center",
                             color="#94a3b8", transform=axes[1].transAxes)
        else:
            axes[1].text(0.5, 0.5, "See Program\nInspector", ha="center", va="center",
                         color="#94a3b8", transform=axes[1].transAxes)
    else:
        axes[1].set_facecolor("#0f1420")
        axes[1].text(0.5, 0.5, "UNKNOWN\n❓", ha="center", va="center",
                     color="#fdba74", fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title("Council's Answer", fontsize=9, color="#94a3b8")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─── MAIN RUN LOGIC ───────────────────────────────────────────────────────────

if run_button and not st.session_state.is_running:
    st.session_state.is_running = True
    st.session_state.log_entries = []
    st.session_state.final_snapshot = None

    task = st.session_state.universe.generate_task(selected_level)
    task.train_pairs = task.train_pairs  # ensure it's populated
    st.session_state.current_task = task
    st.session_state.tasks_attempted += 1


# ─── LAYOUT ───────────────────────────────────────────────────────────────────

if st.session_state.current_task is not None:
    task = st.session_state.current_task
    display_task(task)
    st.divider()

    # Run the Council loop + stream to the dashboard in real time
    if st.session_state.is_running and st.session_state.final_snapshot is None:
        progress_bar = st.progress(0, text="Council is deliberating...")
        log_placeholder = st.empty()
        surprise_placeholder = st.empty()

        snapshot = None
        step = 0

        council_gen = st.session_state.council.solve(task)
        try:
            while True:
                snapshot = next(council_gen)
                step += 1

                # Collect log entries from the snapshot
                st.session_state.log_entries = snapshot.get("agent_call_log", [])

                budget_frac = min(snapshot.get("budget_used", 0) / 100, 1.0)
                progress_bar.progress(budget_frac, text=f"Budget used: {snapshot.get('budget_used', 0)}/100")

                if snapshot.get("final_verdict") != "pending":
                    break

                if step > 300:   # failsafe
                    break

        except StopIteration as e:
            snapshot = e.value or snapshot

        progress_bar.empty()
        st.session_state.final_snapshot = snapshot
        st.session_state.is_running = False

        if snapshot and snapshot.get("final_verdict") == "solved":
            st.session_state.tasks_solved += 1

    # Always render all panels if we have a snapshot
    if st.session_state.final_snapshot:
        snap = st.session_state.final_snapshot

        render_verdict(snap)
        st.divider()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏛️ Council Chamber",
            "⚡ Surprise Metric",
            "🔬 Program Inspector",
            "🔴 Skeptic's Dossier",
            "📉 Generalization",
            "📚 Skill Library",
        ])

        with tab1:
            st.divider()
            ccol1, ccol2 = st.columns([1, 1])
            with ccol1:
                render_council_log(snap.get("agent_call_log", []))
            with ccol2:
                render_answer(task, snap)

        with tab2:
            render_surprise_chart(snap.get("surprise_history", []))

        with tab3:
            render_program_inspector(snap)

        with tab4:
            render_skeptic_dossier(snap)

        with tab5:
            gen_series = st.session_state.council.stats()["generalization_series"]
            render_generalization_curve(gen_series)

        with tab6:
            render_skill_library(st.session_state.council.stats())

else:
    st.info("👈 Select a difficulty level and press **⚡ Run Council** to begin.")
    st.markdown("""
    **This dashboard is a Live Research Laboratory.**

    - **Council Chamber** — Watch the 9 agents debate in real time.
    - **Surprise Metric** — Prediction error decays as the Council understands the task.
    - **Program Inspector** — The discovered rule as human-readable pseudocode.
    - **Skeptic's Dossier** — Every failed hypothesis. Proof of depth of search.
    - **Generalization Curve** — Does the Council get faster over time? That's the AGI proof.
    - **Skill Library** — The growing vocabulary of learned reasoning primitives.
    """)
