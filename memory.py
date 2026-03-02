"""
memory.py — The Shared Free Energy Substrate
=============================================
Contains:
  • Blackboard  — the shared, structured working memory for the Council
  • EpisodeMemory — persistent vector store across tasks (Agent 9 — Archivist)
  • LatentSkillLibrary — growing library of learned reusable latent vectors
  • SurpriseTracker — real-time prediction error monitor (for Curiosity Engine)
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


# ─── ENUMS & CONSTANTS ────────────────────────────────────────────────────────

class HypothesisStatus(Enum):
    PENDING    = "pending"
    TESTING    = "testing"
    FALSIFIED  = "falsified"
    CAUSAL_LAW = "causal_law"
    COINCIDENCE = "coincidence"
    ACCEPTED   = "accepted"


class ProgramVerdict(Enum):
    PASS       = "pass"
    FAIL       = "fail"
    UNKNOWN    = "unknown"


MAX_HYPOTHESIS_STACK = 50
MAX_CONTRADICTION_LOG = 100
MAX_EPISODE_MEMORY = 500


# ─── HYPOTHESIS ───────────────────────────────────────────────────────────────

@dataclass
class Hypothesis:
    """A single imagined output grid produced by the Dreamer."""
    id: str
    grid: np.ndarray
    confidence: float           # dreamer's prediction confidence [0, 1]
    status: HypothesisStatus = HypothesisStatus.PENDING
    program: Optional[str] = None          # Latent z-vector summary from the Scientist
    program_mdl: Optional[float] = None   # Number of active dimensions (sparsity)
    causal_verdict: Optional[str] = None  # CAUSAL_LAW or COINCIDENCE
    contradiction_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "program": self.program,
            "mdl_score": round(self.program_mdl, 4) if self.program_mdl else None,
            "causal_verdict": self.causal_verdict,
            "contradiction_count": self.contradiction_count,
            "grid": self.grid.tolist() if self.grid is not None else None,
        }


# ─── CONTRADICTION LOG ────────────────────────────────────────────────────────

@dataclass
class ContradictionEntry:
    """A falsification event issued by the Skeptic or Causal Reasoner."""
    hypothesis_id: str
    counter_example_input: np.ndarray
    produced_output: np.ndarray
    expected_output: np.ndarray
    failure_mode: str         # e.g. "wrong_color", "wrong_shape", "causal_break"
    agent: str                # "Skeptic" or "CausalReasoner"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "failure_mode": self.failure_mode,
            "agent": self.agent,
            "timestamp": round(self.timestamp, 2),
        }


# ─── WORLD STATE ──────────────────────────────────────────────────────────────

@dataclass
class WorldState:
    """
    The Perceiver's structured representation of the current task grid.
    All agents reason about this, never the raw pixels.
    """
    objects: List[Dict]     # list of {"id", "color", "cells", "bbox", "size"}
    grid_shape: Tuple[int, int]
    background_color: int = 0
    philosopher_revision: int = 0   # how many times the Philosopher has revised this

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def color_set(self) -> List[int]:
        return sorted(set(o["color"] for o in self.objects))

    @property
    def majority_color(self) -> Optional[int]:
        if not self.objects:
            return None
        by_color: Dict[int, int] = {}
        for obj in self.objects:
            c = obj["color"]
            by_color[c] = by_color.get(c, 0) + obj["size"]
        return max(by_color, key=by_color.get)

    @property
    def minority_color(self) -> Optional[int]:
        if not self.objects:
            return None
        by_color: Dict[int, int] = {}
        for obj in self.objects:
            c = obj["color"]
            by_color[c] = by_color.get(c, 0) + obj["size"]
        return min(by_color, key=by_color.get)

    def to_dict(self) -> Dict:
        return {
            "object_count": self.object_count,
            "grid_shape": self.grid_shape,
            "color_set": self.color_set,
            "majority_color": self.majority_color,
            "minority_color": self.minority_color,
            "philosopher_revision": self.philosopher_revision,
            "objects": [
                {
                    "id": o["id"],
                    "color": o["color"],
                    "size": o["size"],
                    "bbox": o["bbox"],
                }
                for o in self.objects
            ],
        }


# ─── BLACKBOARD ───────────────────────────────────────────────────────────────

class Blackboard:
    """
    The Shared Free Energy Substrate.
    The common medium through which all 9 Council agents communicate.
    Every read/write is logged for full transparency.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.round: int = 0
        self.budget_used: int = 0
        self.budget_max: int = 100

        # ── Core state ──
        self.world_state: Optional[WorldState] = None
        self.hypothesis_stack: List[Hypothesis] = []
        self.contradiction_log: List[ContradictionEntry] = []
        self.meeting_agenda: List[str] = []        # ordered list of next speakers
        self.final_answer: Optional[np.ndarray] = None
        self.final_verdict: str = "pending"        # "solved", "unknown", "timeout"

        # ── Metrics ──
        self.surprise_history: List[float] = []   # per-round mean prediction error
        self.agent_call_log: List[Dict] = []       # full audit log

        # ── Prior Art from Archivist ──
        self.prior_art_hints: List[Dict] = []

    # ── World State ─────────────────────────────────────────────────────────

    def set_world_state(self, ws: WorldState, agent: str = "Perceiver") -> None:
        self.world_state = ws
        self._log(agent, "set_world_state", {"objects": ws.object_count})

    # ── Hypothesis Management ────────────────────────────────────────────────

    def push_hypothesis(self, grid: np.ndarray, confidence: float, agent: str = "Dreamer") -> Hypothesis:
        h_id = f"H{len(self.hypothesis_stack):03d}_{hashlib.md5(grid.tobytes()).hexdigest()[:6]}"
        h = Hypothesis(id=h_id, grid=grid.copy(), confidence=confidence)
        self.hypothesis_stack.append(h)
        if len(self.hypothesis_stack) > MAX_HYPOTHESIS_STACK:
            self.hypothesis_stack = self.hypothesis_stack[-MAX_HYPOTHESIS_STACK:]
        self.budget_used += 1
        self._log(agent, "push_hypothesis", {"id": h_id, "confidence": confidence})
        return h

    def get_top_hypothesis(self) -> Optional[Hypothesis]:
        pending = [h for h in self.hypothesis_stack if h.status == HypothesisStatus.PENDING]
        if not pending:
            return None
        return max(pending, key=lambda h: h.confidence)

    def update_hypothesis(self, h_id: str, **kwargs) -> None:
        for h in self.hypothesis_stack:
            if h.id == h_id:
                for k, v in kwargs.items():
                    setattr(h, k, v)
                break

    # ── Contradiction Log ────────────────────────────────────────────────────

    def add_contradiction(self, entry: ContradictionEntry) -> None:
        self.contradiction_log.append(entry)
        if len(self.contradiction_log) > MAX_CONTRADICTION_LOG:
            self.contradiction_log = self.contradiction_log[-MAX_CONTRADICTION_LOG:]
        h = self._find_hypothesis(entry.hypothesis_id)
        if h:
            h.contradiction_count += 1
            h.status = HypothesisStatus.FALSIFIED
        self._log(entry.agent, "contradiction", {"failure_mode": entry.failure_mode})

    # ── Surprise Tracking ────────────────────────────────────────────────────

    def record_surprise(self, error: float, agent: str = "CuriosityEngine") -> None:
        self.surprise_history.append(error)
        self._log(agent, "surprise_recorded", {"error": round(error, 4)})

    @property
    def current_surprise(self) -> float:
        return self.surprise_history[-1] if self.surprise_history else 1.0

    @property
    def surprise_is_plateauing(self) -> bool:
        """Returns True if the last 3 surprise values are within 5% of each other."""
        if len(self.surprise_history) < 3:
            return False
        recents = self.surprise_history[-3:]
        return (max(recents) - min(recents)) < 0.05

    @property
    def surprise_resolved(self) -> bool:
        return self.current_surprise < 0.05

    # ── Meeting Control ──────────────────────────────────────────────────────

    def set_agenda(self, agenda: List[str]) -> None:
        self.meeting_agenda = agenda
        self._log("Metacognitor", "set_agenda", {"speakers": agenda})

    def advance_round(self) -> None:
        self.round += 1

    def set_prior_art(self, hints: List[Dict]) -> None:
        self.prior_art_hints = hints
        self._log("Archivist", "prior_art_injected", {"n_hints": len(hints)})

    # ── Final Answer ─────────────────────────────────────────────────────────

    def declare_answer(self, grid: Optional[np.ndarray], verdict: str, agent: str) -> None:
        self.final_answer = grid
        self.final_verdict = verdict
        self._log(agent, "declare_answer", {"verdict": verdict})

    # ── Budget ───────────────────────────────────────────────────────────────

    @property
    def budget_remaining(self) -> int:
        return self.budget_max - self.budget_used

    @property
    def budget_critical(self) -> bool:
        return self.budget_remaining <= 10

    # ── Serialization ────────────────────────────────────────────────────────

    def snapshot(self) -> Dict:
        """Return a full snapshot of the Blackboard for the Dashboard."""
        return {
            "task_id": self.task_id,
            "round": self.round,
            "budget_used": self.budget_used,
            "budget_remaining": self.budget_remaining,
            "final_verdict": self.final_verdict,
            "surprise_history": self.surprise_history,
            "world_state": self.world_state.to_dict() if self.world_state else None,
            "hypothesis_stack": [h.to_dict() for h in self.hypothesis_stack],
            "contradiction_log": [c.to_dict() for c in self.contradiction_log],
            "prior_art_hints": self.prior_art_hints,
            "agent_call_log": self.agent_call_log[-200:],  # cap for streaming
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _log(self, agent: str, action: str, data: Dict) -> None:
        self.agent_call_log.append({
            "round": self.round,
            "agent": agent,
            "action": action,
            "data": data,
            "timestamp": round(time.time(), 3),
        })

    def _find_hypothesis(self, h_id: str) -> Optional[Hypothesis]:
        for h in self.hypothesis_stack:
            if h.id == h_id:
                return h
        return None


# ─── LATENT SKILL LIBRARY ───────────────────────────────────────────────────────

@dataclass
class LatentSkill:
    """A reusable transformation vector discovered by the Archivist."""
    name: str
    description: str
    z_vector: List[float]       # The transformation vector in latent space
    origin_task_id: str
    usage_count: int = 1
    success_rate: float = 1.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "origin": self.origin_task_id,
            "usage_count": self.usage_count,
            "success_rate": round(self.success_rate, 3),
        }


class LatentSkillLibrary:
    """
    The growing vocabulary of the Council's reasoning.
    Skills are now continuous z-vectors discovered dynamically.
    """

    def __init__(self):
        self._skills: Dict[str, LatentSkill] = {}

    def add_skill(self, skill: LatentSkill) -> None:
        if skill.name in self._skills:
            self._skills[skill.name].usage_count += 1
        else:
            self._skills[skill.name] = skill

    def get_all(self) -> List[LatentSkill]:
        return sorted(self._skills.values(), key=lambda s: s.usage_count, reverse=True)

    def get_hints_for(self, description_keywords: List[str]) -> List[LatentSkill]:
        """Return skills whose descriptions match any of the keywords."""
        hits = []
        for skill in self._skills.values():
            if any(kw.lower() in skill.description.lower() for kw in description_keywords):
                hits.append(skill)
        return hits

    def to_dict(self) -> List[Dict]:
        return [s.to_dict() for s in self.get_all()]


# ─── EPISODE MEMORY (Archivist's long-term store) ─────────────────────────────

@dataclass
class EpisodeRecord:
    """A complete record of one solved (or attempted) task."""
    task_id: str
    task_fingerprint: str
    priors_used: List[str]
    winning_program: Optional[str] # Human-readable string or z_vector summary
    causal_label: str          # "CAUSAL_LAW" | "COINCIDENCE" | "UNKNOWN"
    rounds_to_solve: int
    budget_used: int
    surprise_arc: List[float]  # the full surprise history for this task
    verdict: str               # "solved" | "unknown" | "timeout"
    winning_z: Optional[List[float]] = None # Discovered latent vector
    embedding: Optional[np.ndarray] = None   # for nearest-neighbor retrieval
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "priors": self.priors_used,
            "program": self.winning_program,
            "causal_label": self.causal_label,
            "rounds": self.rounds_to_solve,
            "budget": self.budget_used,
            "verdict": self.verdict,
            "final_surprise": round(self.surprise_arc[-1], 4) if self.surprise_arc else None,
        }


class EpisodeMemory:
    """
    The Archivist's persistent long-term store.
    Stores every episode (solved or attempted) and supports nearest-neighbor retrieval
    for Prior Art hint generation.
    """

    def __init__(self):
        self._episodes: List[EpisodeRecord] = []

    def store(self, record: EpisodeRecord) -> None:
        self._episodes.append(record)
        if len(self._episodes) > MAX_EPISODE_MEMORY:
            # Remove the oldest, lowest-value episodes first
            self._episodes = sorted(
                self._episodes,
                key=lambda e: (e.verdict == "solved", -e.rounds_to_solve)
            )[-MAX_EPISODE_MEMORY:]

    def retrieve_similar(
        self,
        priors: List[str],
        k: int = 3,
    ) -> List[EpisodeRecord]:
        """
        Retrieve K most similar past episodes by Prior overlap.
        Falls back to most recent if no overlap found.
        """
        if not self._episodes:
            return []

        def overlap(ep: EpisodeRecord) -> int:
            return len(set(ep.priors_used) & set(priors))

        scored = sorted(self._episodes, key=overlap, reverse=True)
        return scored[:k]

    def get_generalization_series(self) -> List[Dict]:
        """Return rounds-to-solve over time, for the Generalization Curve."""
        return [
            {"task_id": ep.task_id, "rounds": ep.rounds_to_solve, "verdict": ep.verdict}
            for ep in self._episodes
        ]

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    @property
    def solved_count(self) -> int:
        return sum(1 for ep in self._episodes if ep.verdict == "solved")

    @property
    def avg_rounds(self) -> float:
        solved = [ep for ep in self._episodes if ep.verdict == "solved"]
        if not solved:
            return 0.0
        return sum(ep.rounds_to_solve for ep in solved) / len(solved)

    def to_dict(self) -> List[Dict]:
        return [ep.to_dict() for ep in self._episodes]


# ─── SURPRISE TRACKER ─────────────────────────────────────────────────────────

class SurpriseTracker:
    """
    The Curiosity Engine's instrument.
    Tracks prediction error per round using grid-level pixel MSE.
    """

    def __init__(self):
        self._history: List[float] = []

    def compute(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Compute normalized prediction error between two grids."""
        if predicted.shape != actual.shape:
            return 1.0
        mse = float(np.mean((predicted.astype(float) - actual.astype(float)) ** 2))
        max_val = max(predicted.max(), actual.max(), 1)
        normalized = mse / (max_val ** 2)
        self._history.append(normalized)
        return normalized

    @property
    def current(self) -> float:
        return self._history[-1] if self._history else 1.0

    @property
    def history(self) -> List[float]:
        return list(self._history)

    @property
    def is_resolved(self) -> bool:
        return self.current < 0.05

    @property
    def is_plateauing(self) -> bool:
        if len(self._history) < 3:
            return False
        r = self._history[-3:]
        return (max(r) - min(r)) < 0.05


# ─── SELF-TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bb = Blackboard("DEMO_TASK")
    ws = WorldState(
        objects=[
            {"id": 0, "color": 1, "cells": [(0, 0), (0, 1)], "bbox": (0, 0, 0, 1), "size": 2},
            {"id": 1, "color": 3, "cells": [(2, 2)], "bbox": (2, 2, 2, 2), "size": 1},
        ],
        grid_shape=(5, 5),
    )
    bb.set_world_state(ws)
    g = np.zeros((5, 5), dtype=np.int32)
    h = bb.push_hypothesis(g, confidence=0.8)
    print("Blackboard snapshot keys:", list(bb.snapshot().keys()))

    lib = LatentSkillLibrary()
    print(f"\nLatent Library — {len(lib.get_all())} skills loaded.")

    mem = EpisodeMemory()
    from universe import ARCTask, DifficultyLevel, Universe # universe import correction
    u = Universe(seed=7)
    task = u.generate_task(DifficultyLevel.L1)
    
    # Mocking a latent skill
    z_mock = [0.1] * 64
    mem.store(EpisodeRecord(
        task_id=task.task_id,
        task_fingerprint=task.fingerprint,
        priors_used=[p.value for p in task.priors_used],
        winning_program="z[8 active dims]",
        winning_z=z_mock,
        causal_label="CAUSAL_LAW",
        rounds_to_solve=4,
        budget_used=12,
        surprise_arc=[0.9, 0.6, 0.3, 0.04],
        verdict="solved",
    ))
    similar = mem.retrieve_similar([p.value for p in task.priors_used])
    print(f"\nEpisode Memory — {mem.total_episodes} episodes, "
          f"retrieved {len(similar)} similar.")
    print("✓ memory.py self-test passed.")
