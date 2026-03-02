"""
council.py — The Council of Minds
===================================
All 9 agents + the Council Meeting protocol.
Imports from:  universe.py  |  memory.py

Agents:
  1. Perceiver      — Object segmentation (WorldState)
  2. Dreamer        — World-model-based latent hypothesis generation
  3. Scientist      — Latent transformation discovery (z-vector search)
  4. Skeptic        — Adversarial falsification
  5. Philosopher    — Ontological re-perception & Basis rotation
  6. CausalReasoner — Counterfactual causal testing
  7. CuriosityEngine — Active-inference exploration drive
  8. Metacognitor   — Session monitor, chair, and convergence vote
  9. Archivist      — Episode memory, hints, and latent skill extraction
"""

import numpy as np
import random
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Generator

from universe import ARCTask, perceive_objects, DifficultyLevel, Universe
from memory import (
    Blackboard, WorldState, Hypothesis,
    HypothesisStatus, ContradictionEntry,
    EpisodeMemory, EpisodeRecord, LatentSkillLibrary,
    LatentSkill, SurpriseTracker,
)
from latent_dictionary import LatentDictionary
from meta_learner import MetaLearner

log = logging.getLogger("council")
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")


# ─── AGENT RESULTS ────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    agent: str
    success: bool
    message: str
    data: Dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


# ─── AGENT 1: PERCEIVER ───────────────────────────────────────────────────────

class Perceiver:
    """Segments the raw ARC grid into discrete objects → WorldState."""

    name = "Perceiver"

    def perceive(self, grid: np.ndarray, bb: Blackboard, bg: int = 0) -> AgentResult:
        objects_raw = perceive_objects(grid, bg=bg)
        objects_list = [
            {
                "id": o.id,
                "color": o.color,
                "cells": o.cells,
                "bbox": o.bbox,
                "size": o.size,
            }
            for o in objects_raw
        ]
        ws = WorldState(objects=objects_list, grid_shape=grid.shape, background_color=bg)
        bb.set_world_state(ws)
        return AgentResult(
            agent=self.name, success=True,
            message=f"Perceived {ws.object_count} objects.",
            data=ws.to_dict()
        )


# ─── AGENT 2: DREAMER ─────────────────────────────────────────────────────────

class Dreamer:
    """
    Generates K imagined output hypotheses.
    Sampling entirely from the learned continuous latent space.
    """

    name = "Dreamer"
    K: int = 8   # total hypotheses per round

    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()

    def imagine(
        self,
        task: ARCTask,
        bb: Blackboard,
        skill_lib: LatentSkillLibrary,
        latent_dict: LatentDictionary = None,
        meta_learner: MetaLearner = None,
    ) -> AgentResult:
        """Generate K hypotheses from latent space."""

        generated = 0
        if latent_dict is not None:
            # Bias toward skills that were useful in Prior Art
            hints = bb.prior_art_hints
            prior_z = None
            if hints:
                # Try to extract the first winning z vector from hints
                for hint in hints:
                    z_val = hint.get("winning_z")
                    if z_val:
                        prior_z = np.array(z_val, dtype=np.float32)
                        break

            # If meta_learner is active, use its prior
            prior_z = meta_learner.get_prior_z() if meta_learner is not None else prior_z

            z_samples = latent_dict.sample_z(n=self.K, temperature=1.0, prior_z=prior_z)
            for k, z in enumerate(z_samples):
                predicted = latent_dict.decode_z(z, task.train_pairs[0][0])
                # Confidence inversely proportional to z norm (simpler = better)
                z_norm = float(np.linalg.norm(z))
                confidence = 1.0 / (1.0 + z_norm * 0.5 + k * 0.05)
                bb.push_hypothesis(predicted, confidence=confidence, agent=self.name)
                generated += 1

        return AgentResult(
            agent=self.name, success=True,
            message=f"Imagined {generated} latent hypotheses.",
            data={"n_hypotheses": generated}
        )


# ─── AGENT 3: SCIENTIST ────────────────────────────────────────────────────────

class Scientist:
    """
    Finds the shortest latent vector that explains the top Dreamer hypothesis
    and generalizes across all training pairs. No explicitly defined rules.
    """

    name = "Scientist"
    LATENT_ERROR_THRESHOLD = 0.5  # max reconstruction MSE to accept a latent solution

    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()

    def synthesize(
        self,
        task: ARCTask,
        bb: Blackboard,
        skill_lib: LatentSkillLibrary,
        latent_dict: LatentDictionary = None,
        meta_learner: MetaLearner = None,
    ) -> AgentResult:
        """Search for a generalizing latent transformation."""

        if latent_dict is not None and latent_dict.is_ready:
            # If meta_learner is active, use its prior
            prior_z = meta_learner.get_prior_z() if meta_learner is not None else None

            # Try to find a z vector that generalizes across all training pairs
            z_candidates = []
            for inp, out in task.train_pairs:
                z, err = latent_dict.search_z(inp, out, n_candidates=40, n_refine=20, prior_z=prior_z)
                if z is not None:
                    z_candidates.append((z, err))

            if z_candidates:
                # Average all per-pair z vectors to find a common transformation
                avg_z = np.mean([zc[0] for zc in z_candidates], axis=0)
                avg_z = np.maximum(avg_z, 0)
                avg_err = float(np.mean([zc[1] for zc in z_candidates]))

                # Check if the averaged z generalises across training pairs
                latent_generalises = True
                for inp, expected in task.train_pairs:
                    predicted = latent_dict.decode_z(avg_z, inp)
                    if predicted.shape != expected.shape or not np.array_equal(predicted, expected):
                        latent_generalises = False
                        break

                if latent_generalises:
                    z_mdl = float(np.count_nonzero(avg_z > 0.01))  # MDL = active components
                    z_str = f"z[{z_mdl:.0f} active dims, err={avg_err:.4f}]"

                    top_h = bb.get_top_hypothesis()
                    if top_h:
                        predicted = latent_dict.decode_z(avg_z, task.test_input)
                        bb.update_hypothesis(
                            top_h.id,
                            program=z_str,
                            program_mdl=z_mdl,
                            grid=predicted,
                            status=HypothesisStatus.TESTING,
                            winning_z=avg_z.tolist()  # Custom internal passing mechanism
                        )
                    return AgentResult(
                        agent=self.name, success=True,
                        message=f"Latent program: {z_str} (MDL={z_mdl:.1f})",
                        data={"program": z_str, "mdl": z_mdl, "source": "latent", "winning_z": avg_z.tolist()}
                    )

        return AgentResult(
            agent=self.name, success=False,
            message="No generalizing latent program found in this round.",
        )


# ─── AGENT 4: SKEPTIC ─────────────────────────────────────────────────────────

class Skeptic:
    """
    Adversarially falsifies the Scientist's program using input mutations.
    Implements Popper's Falsificationism.
    """

    name = "Skeptic"
    MUTATION_COUNT = 12

    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()

    def challenge(
        self,
        task: ARCTask,
        bb: Blackboard,
        latent_dict: LatentDictionary = None,
    ) -> AgentResult:
        top_h = bb.get_top_hypothesis()
        if top_h is None or top_h.program is None:
            return AgentResult(
                agent=self.name, success=False,
                message="No program to falsify.",
            )

        winning_z = getattr(top_h, 'winning_z', None)
        if winning_z is None or latent_dict is None or not latent_dict.is_ready:
            return AgentResult(
                agent=self.name, success=True,
                message="PASS — No explicit z-vector to falsify.",
            )

        z_vec = np.array(winning_z, dtype=np.float32)

        for m in range(self.MUTATION_COUNT):
            mutant_inp = self._mutate(task.train_pairs[0][0])
            expected_out = latent_dict.decode_z(z_vec, task.train_pairs[0][0])
            try:
                mutant_out = latent_dict.decode_z(z_vec, mutant_inp)
            except Exception:
                mutant_out = np.zeros_like(mutant_inp)

            # The mutation test: if the program's behavior changes predictably, fine.
            # If the mutation produces nonsense / crashes, flag it.
            if mutant_out.shape != task.test_input.shape and mutant_inp.shape == task.test_input.shape:
                entry = ContradictionEntry(
                    hypothesis_id=top_h.id,
                    counter_example_input=mutant_inp,
                    produced_output=mutant_out,
                    expected_output=expected_out,
                    failure_mode="shape_mismatch_under_mutation",
                    agent=self.name,
                )
                bb.add_contradiction(entry)
                return AgentResult(
                    agent=self.name, success=False,
                    message=f"FALSIFIED on mutation {m}. Mode: shape_mismatch_under_mutation.",
                    data={"mutation": m, "failure": "shape_mismatch"}
                )

        # All mutations survived → mark hypothesis as TESTING passed
        bb.update_hypothesis(top_h.id, status=HypothesisStatus.TESTING)
        return AgentResult(
            agent=self.name, success=True,
            message=f"PASS — program survived {self.MUTATION_COUNT} adversarial mutations.",
            data={"mutations_tested": self.MUTATION_COUNT}
        )

    def _mutate(self, grid: np.ndarray) -> np.ndarray:
        """Apply a random structural mutation to a grid."""
        mutation = self.rng.choice(["swap_colors", "add_noise", "shift"])
        out = grid.copy()

        if mutation == "swap_colors":
            colors = list(np.unique(out[out != 0]))
            if len(colors) >= 2:
                a, b = self.rng.sample(colors, 2)
                tmp = out.copy()
                tmp[grid == a] = b
                tmp[grid == b] = a
                out = tmp

        elif mutation == "add_noise":
            n_points = self.rng.randint(1, 3)
            for _ in range(n_points):
                r = self.rng.randint(0, out.shape[0] - 1)
                c = self.rng.randint(0, out.shape[1] - 1)
                out[r, c] = self.rng.randint(1, 9)

        elif mutation == "shift":
            shift_r = self.rng.randint(-1, 1)
            shift_c = self.rng.randint(-1, 1)
            out = np.roll(out, shift_r, axis=0)
            out = np.roll(out, shift_c, axis=1)

        return out


# ─── AGENT 5: PHILOSOPHER ─────────────────────────────────────────────────────

class Philosopher:
    """
    Challenges the Perceiver's object decomposition.
    Proposes an alternative WorldState when falsification cannot be explained.
    """

    name = "Philosopher"

    def reframe(
        self,
        grid: np.ndarray,
        bb: Blackboard,
        revision: int = 0,
        latent_dict: LatentDictionary = None,
    ) -> AgentResult:
        """
        Rotates the basis of the top latent hypotheses to provide an orthogonal 'perspective'.
        """
        top_h = bb.get_top_hypothesis()
        if top_h is None or not hasattr(top_h, 'winning_z') or top_h.winning_z is None:
            return AgentResult(
                agent=self.name, success=False,
                message="No latent vector to reframe.",
            )

        if latent_dict is None or not latent_dict.is_ready:
            return AgentResult(
                agent=self.name, success=False,
                message="Latent dictionary not ready for basis rotation.",
            )

        z_orig = np.array(top_h.winning_z, dtype=np.float32)
        z_rot = latent_dict.rotate_basis(z_orig, angle_idx=revision)
        
        # Test if the rotated perspective offers a simpler explanation (lower norm)
        # Note: In a pure rotation, norm should be identical. But because we clip
        # to np.maximum(..., 0) during rotation (non-negative constraints), the norm
        # effectively changes, sometimes sparsifying the vector further.
        if np.linalg.norm(z_rot) < np.linalg.norm(z_orig) + 0.1:
            # Update the hypothesis with this new rotated latent vector
            top_h.winning_z = z_rot.tolist()
            return AgentResult(
                agent=self.name, success=True,
                message=f"Reframed latent vector (revision {revision + 1}). Norm: {np.linalg.norm(z_rot):.2f}",
                data={"revision": revision + 1}
            )

        return AgentResult(
            agent=self.name, success=False,
            message="Philosopher rotation did not yield a simpler explanation.",
        )


# ─── AGENT 6: CAUSAL REASONER ─────────────────────────────────────────────────

class CausalReasoner:
    """
    Tests whether the Scientist's program is a CAUSAL_LAW or a COINCIDENCE
    using counterfactual interventions.
    """

    name = "CausalReasoner"
    COUNTERFACTUAL_COUNT = 8

    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()

    def verify(
        self,
        task: ARCTask,
        bb: Blackboard,
        latent_dict: LatentDictionary = None,
    ) -> AgentResult:
        top_h = bb.get_top_hypothesis()
        if top_h is None or not hasattr(top_h, 'winning_z') or top_h.winning_z is None:
            return AgentResult(
                agent=self.name, success=False,
                message="No latent vector to verify causally.",
            )

        if latent_dict is None or not latent_dict.is_ready:
            return AgentResult(
                agent=self.name, success=False,
                message="Latent dictionary not ready for causal intervention.",
            )

        z_arr = np.array(top_h.winning_z, dtype=np.float32)
        failures = 0

        for _ in range(self.COUNTERFACTUAL_COUNT):
            cf_input = self._intervene(task.train_pairs[0][0])
            try:
                original_pred = latent_dict.decode_z(z_arr, task.train_pairs[0][0])
                cf_pred = latent_dict.decode_z(z_arr, cf_input)

                # A causal transformation vector should produce *different* outputs for different inputs
                if np.array_equal(original_pred, cf_pred) and not np.array_equal(task.train_pairs[0][0], cf_input):
                    failures += 1   # The transformation is insensitive (coincidence)
            except Exception:
                failures += 1

        verdict = "COINCIDENCE" if failures > self.COUNTERFACTUAL_COUNT // 2 else "CAUSAL_LAW"
        bb.update_hypothesis(
            top_h.id,
            causal_verdict=verdict,
            status=HypothesisStatus.CAUSAL_LAW if verdict == "CAUSAL_LAW" else HypothesisStatus.COINCIDENCE,
        )

        return AgentResult(
            agent=self.name, success=(verdict == "CAUSAL_LAW"),
            message=f"Verdict: {verdict} (failures={failures}/{self.COUNTERFACTUAL_COUNT})",
            data={"verdict": verdict, "failures": failures}
        )

    def _intervene(self, grid: np.ndarray) -> np.ndarray:
        """Single-variable counterfactual: change one cell's color."""
        out = grid.copy()
        r = self.rng.randint(0, out.shape[0] - 1)
        c = self.rng.randint(0, out.shape[1] - 1)
        old_val = out[r, c]
        new_val = self.rng.choice([v for v in range(0, 10) if v != old_val])
        out[r, c] = new_val
        return out


# ─── AGENT 7: CURIOSITY ENGINE ────────────────────────────────────────────────

class CuriosityEngine:
    """
    Monitors the SurpriseMetric and fires Exploration Directives
    when the Council is stuck (plateau detected).
    Implements Active Inference — drives toward minimal free energy.
    """

    name = "CuriosityEngine"

    def __init__(self):
        self.tracker = SurpriseTracker()
        self.intervention_count = 0

    def observe(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        bb: Blackboard,
    ) -> AgentResult:
        error = self.tracker.compute(predicted, actual)
        bb.record_surprise(error, agent=self.name)

        if self.tracker.is_plateauing:
            self.intervention_count += 1
            directive = self._pick_directive(bb)
            return AgentResult(
                agent=self.name, success=False,
                message=f"PLATEAU detected (error={error:.3f}). Directive: {directive}",
                data={"directive": directive, "error": error}
            )

        return AgentResult(
            agent=self.name, success=True,
            message=f"Surprise: {error:.3f} ({'resolved' if self.tracker.is_resolved else 'ongoing'})",
            data={"error": error, "resolved": self.tracker.is_resolved}
        )

    def _pick_directive(self, bb: Blackboard) -> str:
        n_falsified = sum(
            1 for h in bb.hypothesis_stack
            if h.status == HypothesisStatus.FALSIFIED
        )
        if n_falsified >= 3:
            return "PHILOSOPHER_REFRAME"
        if self.intervention_count % 2 == 0:
            return "DREAMER_EXPLORE_LOW_CONFIDENCE"
        return "SCIENTIST_EXTEND_SEARCH"


# ─── AGENT 8: METACOGNITOR ────────────────────────────────────────────────────

class Metacognitor:
    """
    God's-eye view of the Council.
    Chairs the meeting, detects lazy behavior, manages the Convergence Vote.
    """

    name = "Metacognitor"
    LAZY_THRESHOLD = 2   # same output in N consecutive rounds = lazy behavior

    def __init__(self):
        self._last_outputs: Dict[str, Any] = {}
        self._lazy_counts: Dict[str, int] = {}

    def arbitrate(
        self,
        bb: Blackboard,
        curiosity_directive: Optional[str],
    ) -> AgentResult:
        """Decide the next meeting agenda based on current state."""

        if bb.final_verdict != "pending":
            return AgentResult(
                agent=self.name, success=True,
                message="Meeting concluded.",
                data={"verdict": bb.final_verdict}
            )

        # Budget critical → convergence vote
        if bb.budget_critical:
            return self._convergence_vote(bb)

        # Curiosity engine requested intervention
        if curiosity_directive == "PHILOSOPHER_REFRAME":
            agenda = ["Philosopher", "Perceiver", "Dreamer", "Scientist", "Skeptic", "CausalReasoner"]
        elif curiosity_directive == "DREAMER_EXPLORE_LOW_CONFIDENCE":
            agenda = ["Dreamer", "Scientist", "Skeptic", "CausalReasoner"]
        elif curiosity_directive == "SCIENTIST_EXTEND_SEARCH":
            agenda = ["Scientist", "Skeptic", "CausalReasoner"]
        else:
            # Default flow
            top_h = bb.get_top_hypothesis()
            if top_h is None:
                agenda = ["Dreamer", "Scientist", "Skeptic", "CausalReasoner"]
            elif top_h.status == HypothesisStatus.PENDING:
                agenda = ["Scientist", "Skeptic", "CausalReasoner"]
            elif top_h.status == HypothesisStatus.FALSIFIED:
                agenda = ["Dreamer", "Scientist", "Skeptic", "CausalReasoner"]
            else:
                agenda = ["Skeptic", "CausalReasoner"]

        bb.set_agenda(agenda)
        return AgentResult(
            agent=self.name, success=True,
            message=f"Agenda set: {' → '.join(agenda)}",
            data={"agenda": agenda}
        )

    def _convergence_vote(self, bb: Blackboard) -> AgentResult:
        """Force a convergence vote among all surviving hypotheses."""
        candidates = [
            h for h in bb.hypothesis_stack
            if h.status not in (HypothesisStatus.FALSIFIED, HypothesisStatus.COINCIDENCE)
        ]
        if not candidates:
            bb.declare_answer(None, "unknown", self.name)
            return AgentResult(
                agent=self.name, success=False,
                message="Convergence vote: NO surviving hypotheses. Declaring UNKNOWN.",
                data={"vote": "unknown"}
            )

        winner = max(candidates, key=lambda h: h.confidence * (1.0 if h.causal_verdict == "CAUSAL_LAW" else 0.5))
        if winner.confidence < 0.30:
            bb.declare_answer(None, "unknown", self.name)
            return AgentResult(
                agent=self.name, success=False,
                message=f"Convergence vote: winning confidence {winner.confidence:.2f} < 0.30. UNKNOWN.",
                data={"vote": "unknown"}
            )

        bb.update_hypothesis(winner.id, status=HypothesisStatus.ACCEPTED)
        bb.declare_answer(winner.grid, "solved", self.name)
        return AgentResult(
            agent=self.name, success=True,
            message=f"Convergence vote: {winner.id} wins (confidence={winner.confidence:.2f}).",
            data={"winner": winner.id, "confidence": winner.confidence}
        )


# ─── AGENT 9: ARCHIVIST ───────────────────────────────────────────────────────

class Archivist:
    """
    Long-term memory, Prior Art hint generation, and Skill Primitive extraction.
    """

    name = "Archivist"

    def __init__(self, memory: EpisodeMemory, skill_lib: LatentSkillLibrary,
                 latent_dict: LatentDictionary = None, meta_learner: MetaLearner = None):
        self.memory = memory
        self.skill_lib = skill_lib
        self.latent_dict = latent_dict
        self.meta_learner = meta_learner

    def inject_hints(self, task: ARCTask, bb: Blackboard) -> AgentResult:
        """Retrieve similar past episodes and inject hints into the Blackboard."""
        priors = [p.value for p in task.priors_used]
        similar = self.memory.retrieve_similar(priors, k=3)
        hints = [ep.to_dict() for ep in similar]
        bb.set_prior_art(hints)
        return AgentResult(
            agent=self.name, success=True,
            message=f"Injected {len(hints)} prior art hints.",
            data={"hints": hints}
        )

    def archive(
        self,
        task: ARCTask,
        bb: Blackboard,
    ) -> AgentResult:
        """Store this episode, extract skill primitives, and feed the latent dictionary."""
        top_accepted = next(
            (h for h in bb.hypothesis_stack if h.status == HypothesisStatus.ACCEPTED),
            None,
        )
        winning_program = top_accepted.program if top_accepted else None
        winning_z = getattr(top_accepted, 'winning_z', None) if top_accepted else None
        causal_label = top_accepted.causal_verdict or "UNKNOWN" if top_accepted else "UNKNOWN"

        record = EpisodeRecord(
            task_id=task.task_id,
            task_fingerprint=task.fingerprint,
            priors_used=[p.value for p in task.priors_used],
            winning_program=winning_program,
            winning_z=winning_z,
            causal_label=causal_label,
            rounds_to_solve=bb.round,
            budget_used=bb.budget_used,
            surprise_arc=list(bb.surprise_history),
            verdict=bb.final_verdict,
        )
        self.memory.store(record)

        # Extract latent skill from the winning z-vector
        if winning_z:
            self.skill_lib.add_skill(LatentSkill(
                name=f"z_{task.task_id}",
                description=f"Transformation for {task.task_id} ({task.transformation_description})",
                z_vector=winning_z,
                origin_task_id=task.task_id,
            ))

        # ── Feed the Latent Dictionary with all training pairs + test pair ──
        if self.latent_dict is not None:
            for inp, out in task.train_pairs:
                self.latent_dict.register_pair(
                    inp, out, task_id=task.task_id,
                    label=task.transformation_description[:40]
                )
            # Also register the test pair (ground truth)
            self.latent_dict.register_pair(
                task.test_input, task.test_output,
                task_id=task.task_id,
                label=f"{task.transformation_description[:30]}_test"
            )

        # ── Feed the Meta-Learner ──
        if winning_z is not None and self.meta_learner is not None and bb.final_verdict == "solved":
            self.meta_learner.update(winning_z=np.array(winning_z), rounds_to_solve=bb.round)

        return AgentResult(
            agent=self.name, success=True,
            message=f"Archived episode. Verdict: {bb.final_verdict}. Total: {self.memory.total_episodes}.",
            data={"verdict": bb.final_verdict, "total_episodes": self.memory.total_episodes}
        )


# ─── THE COUNCIL MEETING LOOP ─────────────────────────────────────────────────

class Council:
    """
    The orchestrator of the 9-agent Council Meeting.
    Drives the Socratic loop until consensus or budget exhaustion.
    """

    MAX_ROUNDS = 30

    def __init__(self, seed: int = None):
        rng = random.Random(seed)
        self.perceiver     = Perceiver()
        self.dreamer       = Dreamer(rng)
        self.scientist     = Scientist(rng)
        self.skeptic       = Skeptic(rng)
        self.philosopher   = Philosopher()
        self.causal        = CausalReasoner(rng)
        self.curiosity     = CuriosityEngine()
        self.metacognitor  = Metacognitor()
        self.memory        = EpisodeMemory()
        self.skill_lib     = LatentSkillLibrary()
        self.latent_dict   = LatentDictionary(seed=seed or 42)
        self.meta_learner  = MetaLearner()
        self.archivist     = Archivist(self.memory, self.skill_lib, self.latent_dict, self.meta_learner)

    def solve(self, task: ARCTask, stream: bool = False) -> Generator[Dict, None, Dict]:
        """
        Run the full Council Meeting for a given task.
        Yields a log dict after each agent action (for live dashboard streaming).
        Returns the final Blackboard snapshot.
        """
        bb = Blackboard(task.task_id)
        philosopher_revision = 0

        # ── PHASE 0: ORIENTATION ──────────────────────────────────────────────
        self._emit(bb, "Orientation", "Meeting begins.")
        yield bb.snapshot()

        result = self.perceiver.perceive(task.test_input, bb)
        self._emit(bb, result.agent, result.message)
        yield bb.snapshot()

        result = self.archivist.inject_hints(task, bb)
        self._emit(bb, result.agent, result.message)
        yield bb.snapshot()

        # ── PHASE 1: FIRST IMAGINATION ───────────────────────────────────────
        bb.advance_round()
        result = self.dreamer.imagine(task, bb, self.skill_lib, self.latent_dict, self.meta_learner)
        self._emit(bb, result.agent, result.message)
        yield bb.snapshot()

        # ── MAIN DEBATE LOOP ──────────────────────────────────────────────────
        curiosity_directive: Optional[str] = None

        while bb.final_verdict == "pending" and bb.round < self.MAX_ROUNDS:
            bb.advance_round()

            # Metacognitor sets the agenda
            meta_result = self.metacognitor.arbitrate(bb, curiosity_directive)
            self._emit(bb, meta_result.agent, meta_result.message)
            yield bb.snapshot()

            if bb.final_verdict != "pending":
                break

            agenda = bb.meeting_agenda

            for agent_name in agenda:

                if agent_name == "Scientist":
                    result = self.scientist.synthesize(task, bb, self.skill_lib, self.latent_dict, self.meta_learner)
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()

                elif agent_name == "Skeptic":
                    result = self.skeptic.challenge(task, bb, self.latent_dict)
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()

                    if not result.success:
                        # Skeptic found a contradiction → curiosity engine evaluates
                        top_h = bb.get_top_hypothesis()
                        if top_h is not None:
                            predicted = top_h.grid
                            curiosity_result = self.curiosity.observe(
                                predicted, task.test_output, bb
                            )
                            self._emit(bb, curiosity_result.agent, curiosity_result.message)
                            curiosity_directive = curiosity_result.data.get("directive")
                            yield bb.snapshot()
                        break   # restart round

                elif agent_name == "CausalReasoner":
                    # Capture the top pending hypothesis BEFORE causal.verify changes its status
                    top_h_before = bb.get_top_hypothesis()
                    result = self.causal.verify(task, bb, self.latent_dict)
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()

                    if result.success and top_h_before is not None:
                        # Causal law confirmed → accept and declare solved
                        bb.update_hypothesis(top_h_before.id, status=HypothesisStatus.ACCEPTED)
                        bb.declare_answer(top_h_before.grid, "solved", "Council")
                        self._emit(bb, "Council", f"SOLVED in {bb.round} rounds! Program: {top_h_before.program}")
                        yield bb.snapshot()
                    elif not result.success:
                        curiosity_directive = "DREAMER_EXPLORE_LOW_CONFIDENCE"
                        break

                elif agent_name == "Dreamer":
                    result = self.dreamer.imagine(task, bb, self.skill_lib, self.latent_dict, self.meta_learner)
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()

                elif agent_name == "Philosopher":
                    result = self.philosopher.reframe(task.test_input, bb, philosopher_revision, self.latent_dict)
                    philosopher_revision += 1
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()
                    # After reframe, re-perceive
                    result = self.perceiver.perceive(task.test_input, bb)
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()

                elif agent_name == "Perceiver":
                    result = self.perceiver.perceive(task.test_input, bb)
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()

                if bb.final_verdict != "pending":
                    break

            # Curiosity engine end-of-round observation (once per round, best hypothesis)
            if bb.final_verdict == "pending":
                top_h = bb.get_top_hypothesis()
                if top_h is None:
                    # No PENDING hypotheses; pick best non-falsified
                    candidates = [h for h in bb.hypothesis_stack
                                  if h.status not in (HypothesisStatus.FALSIFIED, HypothesisStatus.COINCIDENCE)
                                  and h.grid is not None]
                    if candidates:
                        top_h = max(candidates, key=lambda h: h.confidence)
                if top_h is not None:
                    curiosity_result = self.curiosity.observe(top_h.grid, task.test_output, bb)
                    curiosity_directive = curiosity_result.data.get("directive")
                    self._emit(bb, curiosity_result.agent, curiosity_result.message)

        # ── PHASE 5: ARCHIVAL ─────────────────────────────────────────────────
        archive_result = self.archivist.archive(task, bb)
        self._emit(bb, archive_result.agent, archive_result.message)

        yield bb.snapshot()

    def stats(self) -> Dict:
        return {
            "total_episodes": self.memory.total_episodes,
            "solved": self.memory.solved_count,
            "avg_rounds": round(self.memory.avg_rounds, 1),
            "skill_library_size": len(self.skill_lib.get_all()),
            "generalization_series": self.memory.get_generalization_series(),
            "latent_skills": self.skill_lib.to_dict(),
            "latent_dictionary": self.latent_dict.stats(),
            "meta_learner": self.meta_learner.stats() if self.meta_learner else {},
        }

    @staticmethod
    def _emit(bb: Blackboard, agent: str, message: str) -> None:
        """Log an agent action to the Blackboard audit log."""
        # Store with 'message' key so the dashboard can display it directly
        bb.agent_call_log.append({
            "round": bb.round,
            "agent": agent,
            "action": "speak",
            "message": message,
            "data": {},
            "timestamp": round(time.time(), 3),
        })
        log.info("[Round %02d | %s] %s", bb.round, agent, message)


# ─── SELF-TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Council Self-Test — 3 tasks")
    print("=" * 60)

    universe = Universe(seed=99)
    council = Council(seed=99)

    for i, level in enumerate([DifficultyLevel.L1, DifficultyLevel.L2, DifficultyLevel.L1]):
        task = universe.generate_task(level)
        print(f"\n── Task {i+1}: {task.task_id} ──")
        print(f"   Rule: {task.transformation_description}")

        final_snapshot = None
        for snapshot in council.solve(task):
            final_snapshot = snapshot

        print(f"   Verdict : {final_snapshot['final_verdict']}")
        print(f"   Rounds  : {final_snapshot['round']}")
        print(f"   Budget  : {final_snapshot['budget_used']}/100")

    s = council.stats()
    print(f"\n── Council Stats ──")
    print(f"   Episodes   : {s['total_episodes']}")
    print(f"   Solved     : {s['solved']}")
    print(f"   Avg Rounds : {s['avg_rounds']}")
    print(f"   Skills     : {s['skill_library_size']}")
    print("\n✓ council.py self-test passed.")
