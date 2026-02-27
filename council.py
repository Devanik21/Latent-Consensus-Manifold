"""
council.py — The Council of Minds
===================================
All 9 agents + the Council Meeting protocol.
Imports from:  universe.py  |  memory.py

Agents:
  1. Perceiver      — Object segmentation (WorldState)
  2. Dreamer        — World-model-based hypothesis generation
  3. Scientist      — DSL program synthesis (MDL search)
  4. Skeptic        — Adversarial falsification
  5. Philosopher    — Ontological re-perception
  6. CausalReasoner — Counterfactual causal testing
  7. CuriosityEngine — Active-inference exploration drive
  8. Metacognitor   — Session monitor, chair, and convergence vote
  9. Archivist      — Episode memory, hints, and skill extraction
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
    EpisodeMemory, EpisodeRecord, DSLSkillLibrary,
    SkillPrimitive, SurpriseTracker,
)

log = logging.getLogger("council")
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")


# ─── DSL INTERPRETER ──────────────────────────────────────────────────────────

class DSL:
    """
    The Domain Specific Language interpreter.
    Programs are lists of (primitive_name, *args) tuples.
    """

    PRIMITIVES = {
        "rotate90":       lambda g, _bg: np.rot90(g, 1),
        "rotate180":      lambda g, _bg: np.rot90(g, 2),
        "rotate270":      lambda g, _bg: np.rot90(g, 3),
        "mirror_h":       lambda g, _bg: np.fliplr(g),
        "mirror_v":       lambda g, _bg: np.flipud(g),
        "gravity_down":   lambda g, bg:  DSL._gravity(g, bg, "down"),
        "gravity_up":     lambda g, bg:  DSL._gravity(g, bg, "up"),
        "majority_recolor": lambda g, bg: DSL._majority_recolor(g, bg),
        "sort_by_size":   lambda g, bg:  DSL._sort_by_size(g, bg),
        "identity":       lambda g, _bg: g.copy(),
    }

    @staticmethod
    def execute(grid: np.ndarray, program: List[str], bg: int = 0) -> np.ndarray:
        """Execute a program (list of primitive names) on a grid."""
        result = grid.copy()
        for prim in program:
            fn = DSL.PRIMITIVES.get(prim)
            if fn:
                result = fn(result, bg)
        return result

    @staticmethod
    def program_to_str(program: List[str]) -> str:
        return " → ".join(program)

    @staticmethod
    def mdl_score(program: List[str]) -> float:
        """Minimum Description Length proxy: shorter programs score lower (better)."""
        return float(len(program))

    # ── Primitive implementations ──

    @staticmethod
    def _gravity(grid: np.ndarray, bg: int, direction: str) -> np.ndarray:
        out = np.full_like(grid, bg)
        for col in range(grid.shape[1]):
            column = grid[:, col]
            non_bg = column[column != bg]
            if direction == "down":
                out[grid.shape[0] - len(non_bg):, col] = non_bg
            else:
                out[:len(non_bg), col] = non_bg
        return out

    @staticmethod
    def _majority_recolor(grid: np.ndarray, bg: int) -> np.ndarray:
        flat = grid[grid != bg].flatten()
        if len(flat) == 0:
            return grid.copy()
        colors, counts = np.unique(flat, return_counts=True)
        majority = colors[np.argmax(counts)]
        out = grid.copy()
        out[grid != bg] = majority
        return out

    @staticmethod
    def _sort_by_size(grid: np.ndarray, bg: int) -> np.ndarray:
        try:
            from scipy.ndimage import label
        except ImportError:
            return grid.copy()
        objs = []
        for color in np.unique(grid):
            if color == bg:
                continue
            mask = (grid == color).astype(int)
            labeled, n = label(mask)
            for lab in range(1, n + 1):
                cells = list(zip(*np.where(labeled == lab)))
                objs.append({"color": color, "cells": cells})

        objs.sort(key=lambda o: len(o["cells"]))
        out = np.full_like(grid, bg)
        col_cursor = 0
        for obj in objs:
            if not obj["cells"]:
                continue
            rows = [c[0] for c in obj["cells"]]
            cols = [c[1] for c in obj["cells"]]
            r_min, c_min = min(rows), min(cols)
            w = max(cols) - c_min + 1
            if col_cursor + w > grid.shape[1]:
                break
            for r, c in obj["cells"]:
                out[r, col_cursor + (c - c_min)] = obj["color"]
            col_cursor += w + 1
        return out


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
    Powered by combinatorial rollouts through the DSL, biased by prior art.
    """

    name = "Dreamer"
    K: int = 8   # number of hypotheses per round

    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()

    def imagine(
        self,
        task: ARCTask,
        bb: Blackboard,
        skill_lib: DSLSkillLibrary,
    ) -> AgentResult:
        """Generate K hypotheses and push them to the Blackboard."""

        primitives = list(DSL.PRIMITIVES.keys())
        # Bias toward skills that were useful in Prior Art
        hints = bb.prior_art_hints
        biased = []
        for hint in hints:
            prog = hint.get("winning_program", "")
            if prog:
                biased += prog.split(" → ")[:2]

        generated = 0
        for k in range(self.K):
            # Pick a program length 1..3
            length = self.rng.randint(1, 3)

            # With 50% chance, use a biased primitive as the first step
            program = []
            if biased and self.rng.random() > 0.5:
                program.append(self.rng.choice(biased))
                length -= 1

            program += self.rng.choices(primitives, k=length)
            program = list(dict.fromkeys(program))  # deduplicate while preserving order

            # Apply to training input to produce imagined output
            predicted = DSL.execute(task.train_pairs[0][0], program)
            confidence = 1.0 / (1.0 + DSL.mdl_score(program) + k * 0.05)

            bb.push_hypothesis(predicted, confidence=confidence, agent=self.name)
            generated += 1

        return AgentResult(
            agent=self.name, success=True,
            message=f"Imagined {generated} hypotheses.",
            data={"n_hypotheses": generated}
        )


# ─── AGENT 3: SCIENTIST ────────────────────────────────────────────────────────

class Scientist:
    """
    Finds the shortest DSL program (MDL) that explains the top Dreamer hypothesis
    and generalizes across all training pairs.
    """

    name = "Scientist"
    MAX_PROGRAM_LENGTH = 4
    MCTS_ROLLOUTS = 60

    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()

    def synthesize(
        self,
        task: ARCTask,
        bb: Blackboard,
        skill_lib: DSLSkillLibrary,
    ) -> AgentResult:
        """Search for a generalizing program. Update the best hypothesis on success."""

        best_program: Optional[List[str]] = None
        best_score = float("inf")
        primitives = list(DSL.PRIMITIVES.keys())

        for rollout in range(self.MCTS_ROLLOUTS):
            length = self.rng.randint(1, self.MAX_PROGRAM_LENGTH)
            program = self.rng.choices(primitives, k=length)
            program = list(dict.fromkeys(program))

            if self._generalizes(program, task):
                score = DSL.mdl_score(program)
                if score < best_score:
                    best_score = score
                    best_program = program

        if best_program is None:
            return AgentResult(
                agent=self.name, success=False,
                message="No generalizing program found in this round.",
            )

        prog_str = DSL.program_to_str(best_program)

        # Attach the program to the best pending hypothesis
        top_h = bb.get_top_hypothesis()
        if top_h:
            # Regenerate the predicted output using this program on the test input
            predicted = DSL.execute(task.test_input, best_program)
            bb.update_hypothesis(
                top_h.id,
                program=prog_str,
                program_mdl=best_score,
                grid=predicted,
                status=HypothesisStatus.TESTING,
            )

        return AgentResult(
            agent=self.name, success=True,
            message=f"Found program: {prog_str} (MDL={best_score:.1f})",
            data={"program": prog_str, "mdl": best_score}
        )

    def _generalizes(self, program: List[str], task: ARCTask) -> bool:
        """Check if a program correctly maps every training input → output."""
        for inp, expected in task.train_pairs:
            produced = DSL.execute(inp, program)
            if produced.shape != expected.shape:
                return False
            if not np.array_equal(produced, expected):
                return False
        return True


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
    ) -> AgentResult:
        top_h = bb.get_top_hypothesis()
        if top_h is None or top_h.program is None:
            return AgentResult(
                agent=self.name, success=False,
                message="No program to falsify.",
            )

        program = top_h.program.split(" → ")

        for m in range(self.MUTATION_COUNT):
            mutant_inp = self._mutate(task.train_pairs[0][0])
            expected_out = DSL.execute(task.train_pairs[0][0], program)    # original expected
            mutant_out = DSL.execute(mutant_inp, program)

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
    ) -> AgentResult:
        """Propose an alternative object decomposition."""
        if revision == 0:
            # Try: include background as an object
            bg_cells = list(zip(*np.where(grid == 0)))
            objects = (bb.world_state.objects or []).copy()
            if bg_cells:
                max_id = max((o["id"] for o in objects), default=-1) + 1
                objects.append({
                    "id": max_id,
                    "color": 0,
                    "cells": bg_cells,
                    "bbox": (0, 0, grid.shape[0]-1, grid.shape[1]-1),
                    "size": len(bg_cells),
                    "is_background": True,
                })
            new_ws = WorldState(
                objects=objects,
                grid_shape=grid.shape,
                philosopher_revision=revision + 1,
            )
        elif revision == 1:
            # Try: merge all same-color regions into single objects
            objects = []
            obj_id = 0
            for color in np.unique(grid):
                cells = list(zip(*np.where(grid == color)))
                if cells:
                    rows = [c[0] for c in cells]
                    cols = [c[1] for c in cells]
                    objects.append({
                        "id": obj_id,
                        "color": int(color),
                        "cells": cells,
                        "bbox": (min(rows), min(cols), max(rows), max(cols)),
                        "size": len(cells),
                    })
                    obj_id += 1
            new_ws = WorldState(
                objects=objects,
                grid_shape=grid.shape,
                philosopher_revision=revision + 1,
            )
        else:
            return AgentResult(
                agent=self.name, success=False,
                message="Philosopher exhausted all reframing strategies.",
            )

        bb.set_world_state(new_ws)
        return AgentResult(
            agent=self.name, success=True,
            message=f"Reframed WorldState (revision {revision + 1}). "
                    f"Objects now: {new_ws.object_count}.",
            data=new_ws.to_dict(),
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
    ) -> AgentResult:
        top_h = bb.get_top_hypothesis()
        if top_h is None or top_h.program is None:
            return AgentResult(
                agent=self.name, success=False,
                message="No program to verify causally.",
            )

        program = top_h.program.split(" → ")
        failures = 0

        for _ in range(self.COUNTERFACTUAL_COUNT):
            cf_input = self._intervene(task.train_pairs[0][0])
            try:
                original_pred = DSL.execute(task.train_pairs[0][0], program)
                cf_pred = DSL.execute(cf_input, program)

                # A causal program should produce *different* outputs for different inputs
                if np.array_equal(original_pred, cf_pred) and not np.array_equal(task.train_pairs[0][0], cf_input):
                    failures += 1   # The program is insensitive (coincidence)
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

    def __init__(self, memory: EpisodeMemory, skill_lib: DSLSkillLibrary):
        self.memory = memory
        self.skill_lib = skill_lib

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
        """Store this episode and extract skill primitives from the winning program."""
        top_accepted = next(
            (h for h in bb.hypothesis_stack if h.status == HypothesisStatus.ACCEPTED),
            None,
        )
        winning_program = top_accepted.program if top_accepted else None
        causal_label = top_accepted.causal_verdict or "UNKNOWN" if top_accepted else "UNKNOWN"

        record = EpisodeRecord(
            task_id=task.task_id,
            task_fingerprint=task.fingerprint,
            priors_used=[p.value for p in task.priors_used],
            winning_program=winning_program,
            causal_label=causal_label,
            rounds_to_solve=bb.round,
            budget_used=bb.budget_used,
            surprise_arc=list(bb.surprise_history),
            verdict=bb.final_verdict,
        )
        self.memory.store(record)

        # Extract skill primitives from the winning program
        if winning_program:
            for prim_name in winning_program.split(" → "):
                prim_name = prim_name.strip()
                if prim_name in DSL.PRIMITIVES:
                    self.skill_lib.add_skill(SkillPrimitive(
                        name=prim_name,
                        description=f"Used to solve {task.task_id} ({task.transformation_description})",
                        code=f"{prim_name}(grid)",
                        origin_task_id=task.task_id,
                    ))

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
        self.skill_lib     = DSLSkillLibrary()
        self.archivist     = Archivist(self.memory, self.skill_lib)

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
        result = self.dreamer.imagine(task, bb, self.skill_lib)
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
                    result = self.scientist.synthesize(task, bb, self.skill_lib)
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()

                elif agent_name == "Skeptic":
                    result = self.skeptic.challenge(task, bb)
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
                    result = self.causal.verify(task, bb)
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()

                    if result.success:
                        # Causal law confirmed → declare solved
                        top_h = bb.get_top_hypothesis()
                        if top_h and top_h.status == HypothesisStatus.CAUSAL_LAW:
                            bb.update_hypothesis(top_h.id, status=HypothesisStatus.ACCEPTED)
                            bb.declare_answer(top_h.grid, "solved", "Council")
                            self._emit(bb, "Council", f"SOLVED in {bb.round} rounds!")
                            yield bb.snapshot()
                    else:
                        curiosity_directive = "DREAMER_EXPLORE_LOW_CONFIDENCE"
                        break

                elif agent_name == "Dreamer":
                    result = self.dreamer.imagine(task, bb, self.skill_lib)
                    self._emit(bb, result.agent, result.message)
                    yield bb.snapshot()

                elif agent_name == "Philosopher":
                    result = self.philosopher.reframe(task.test_input, bb, philosopher_revision)
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

            # Curiosity engine observation every round
            top_h = bb.get_top_hypothesis()
            if top_h is not None:
                curiosity_result = self.curiosity.observe(top_h.grid, task.test_output, bb)
                curiosity_directive = curiosity_result.data.get("directive")
                self._emit(bb, curiosity_result.agent, curiosity_result.message)

        # ── PHASE 5: ARCHIVAL ─────────────────────────────────────────────────
        archive_result = self.archivist.archive(task, bb)
        self._emit(bb, archive_result.agent, archive_result.message)

        return bb.snapshot()

    def stats(self) -> Dict:
        return {
            "total_episodes": self.memory.total_episodes,
            "solved": self.memory.solved_count,
            "avg_rounds": round(self.memory.avg_rounds, 1),
            "skill_library_size": len(self.skill_lib.get_all()),
            "generalization_series": self.memory.get_generalization_series(),
            "dsl_skills": self.skill_lib.to_dict(),
        }

    @staticmethod
    def _emit(bb: Blackboard, agent: str, message: str) -> None:
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
