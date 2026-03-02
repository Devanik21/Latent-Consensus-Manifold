"""
universe.py — The Procedural ARC-AGI-2 Task Generator
======================================================
Zero-Cheat: Every task is synthesized from scratch using Core Knowledge Priors.
No two tasks share the same transformation fingerprint in a session.
"""

import numpy as np
import hashlib
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from enum import Enum


# ─── CORE ENUMS ────────────────────────────────────────────────────────────────

class Prior(Enum):
    # Domain A (Spatial / Geometric)
    SYMMETRY      = "symmetry"
    
    # Domain B (Topological / Physical)
    OBJECTNESS    = "objectness"
    CONTAINMENT   = "containment"
    GRAVITY       = "gravity"
    
    # Domain C (Abstract / Logical / Relational)
    NUMEROSITY    = "numerosity"
    CAUSALITY     = "causality"
    GOAL          = "goal_directedness"


class TaskDomain(Enum):
    A_SPATIAL    = "A_Spatial"
    B_TOPOLOGICAL = "B_Topological"
    C_ABSTRACT   = "C_Abstract"


class DifficultyLevel(Enum):
    L1 = 1  # 1 prior
    L2 = 2  # 2 priors
    L3 = 3  # 3 priors
    L4 = 4  # 4 priors
    L5 = 5  # 4+ chained priors


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

@dataclass
class GridObject:
    """A discrete object detected in an ARC grid."""
    id: int
    color: int
    cells: List[Tuple[int, int]]   # list of (row, col)
    bbox: Tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max)
    size: int
    is_background: bool = False

    @property
    def centroid(self) -> Tuple[float, float]:
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        return (sum(rows) / len(rows), sum(cols) / len(cols))


@dataclass
class ARCTask:
    """A single ARC-AGI-2 task with training examples and a test pair."""
    task_id: str
    fingerprint: str
    priors_used: List[Prior]
    difficulty: DifficultyLevel
    train_pairs: List[Tuple[np.ndarray, np.ndarray]]  # [(input, output), ...]
    test_input: np.ndarray
    test_output: np.ndarray   # ground truth, hidden from agents
    transformation_description: str   # human-readable rule for the dashboard

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "fingerprint": self.fingerprint,
            "priors": [p.value for p in self.priors_used],
            "difficulty": self.difficulty.value,
            "train": [
                {"input": inp.tolist(), "output": out.tolist()}
                for inp, out in self.train_pairs
            ],
            "test_input": self.test_input.tolist(),
            "description": self.transformation_description,
        }


# ─── TRANSFORMATION PRIMITIVES ────────────────────────────────────────────────

class GridTransforms:
    """All atomic grid transformation functions used by the generator."""

    @staticmethod
    def rotate90(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=1)

    @staticmethod
    def rotate180(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2)

    @staticmethod
    def rotate270(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=3)

    @staticmethod
    def mirror_h(grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)

    @staticmethod
    def mirror_v(grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)

    @staticmethod
    def recolor(grid: np.ndarray, from_color: int, to_color: int) -> np.ndarray:
        out = grid.copy()
        out[out == from_color] = to_color
        return out

    @staticmethod
    def gravity_down(grid: np.ndarray, bg: int = 0) -> np.ndarray:
        """Drop all non-background cells downward within each column."""
        out = np.full_like(grid, bg)
        for col in range(grid.shape[1]):
            column = grid[:, col]
            non_bg = column[column != bg]
            out[grid.shape[0] - len(non_bg):, col] = non_bg
        return out

    @staticmethod
    def gravity_up(grid: np.ndarray, bg: int = 0) -> np.ndarray:
        out = np.full_like(grid, bg)
        for col in range(grid.shape[1]):
            column = grid[:, col]
            non_bg = column[column != bg]
            out[:len(non_bg), col] = non_bg
        return out

    @staticmethod
    def fill_enclosed(grid: np.ndarray, fill_color: int, bg: int = 0) -> np.ndarray:
        """Flood-fill holes enclosed by a border of non-background cells."""
        from scipy.ndimage import label
        out = grid.copy()
        mask = (grid == bg).astype(int)
        labeled, n = label(mask)
        # The background region touching the border is label at corners
        border_labels = set()
        for r in [0, grid.shape[0] - 1]:
            for c in range(grid.shape[1]):
                if labeled[r, c] > 0:
                    border_labels.add(labeled[r, c])
        for c in [0, grid.shape[1] - 1]:
            for r in range(grid.shape[0]):
                if labeled[r, c] > 0:
                    border_labels.add(labeled[r, c])
        for lab in range(1, n + 1):
            if lab not in border_labels:
                out[labeled == lab] = fill_color
        return out

    @staticmethod
    def scale_object(grid: np.ndarray, bg: int = 0, factor: int = 2) -> np.ndarray:
        """Scale all non-background cells by factor (pixel art style)."""
        rows, cols = grid.shape
        out = np.full((rows * factor, cols * factor), bg, dtype=grid.dtype)
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != bg:
                    out[r*factor:(r+1)*factor, c*factor:(c+1)*factor] = grid[r, c]
        return out

    @staticmethod
    def majority_recolor(grid: np.ndarray, bg: int = 0) -> np.ndarray:
        """Recolor: the majority non-bg color overwrites all non-bg cells."""
        flat = grid[grid != bg].flatten()
        if len(flat) == 0:
            return grid.copy()
        colors, counts = np.unique(flat, return_counts=True)
        majority = colors[np.argmax(counts)]
        out = grid.copy()
        out[grid != bg] = majority
        return out

    @staticmethod
    def sort_objects_by_size(grid: np.ndarray, bg: int = 0) -> np.ndarray:
        """Sort objects left-to-right by ascending cell count."""
        objects = GridTransforms._extract_objects(grid, bg)
        if not objects:
            return grid.copy()
        objects_sorted = sorted(objects, key=lambda o: len(o["cells"]))
        out = np.full_like(grid, bg)
        # place them sequentially from left
        col_cursor = 0
        for obj in objects_sorted:
            r_min = min(c[0] for c in obj["cells"])
            c_min = min(c[1] for c in obj["cells"])
            r_max = max(c[0] for c in obj["cells"])
            c_max = max(c[1] for c in obj["cells"])
            h = r_max - r_min + 1
            w = c_max - c_min + 1
            if col_cursor + w > grid.shape[1]:
                break
            for (r, c) in obj["cells"]:
                out[r + 0, col_cursor + (c - c_min)] = obj["color"]
            col_cursor += w + 1
        return out

    @staticmethod
    def _extract_objects(grid: np.ndarray, bg: int = 0) -> List[Dict]:
        from scipy.ndimage import label
        objects = []
        for color in np.unique(grid):
            if color == bg:
                continue
            mask = (grid == color).astype(int)
            labeled, n = label(mask)
            for lab in range(1, n + 1):
                cells = list(zip(*np.where(labeled == lab)))
                objects.append({"color": color, "cells": cells})
        return objects


# ─── GENERATOR ────────────────────────────────────────────────────────────────

class Universe:
    """
    The Procedural ARC-AGI-2 Task Generator.

    Generates mathematically unique tasks using Core Knowledge Priors.
    Maintains a session fingerprint log to guarantee zero-shot uniqueness.
    """

    COLORS = list(range(1, 10))   # 1-9, 0 = background
    GRID_SIZES = [(5, 5), (6, 6), (8, 8), (10, 10), (12, 12), (15, 15)]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._fingerprint_log: set = set()
        self._task_counter: int = 0

        # Register all prior-based generators
        self._prior_generators: Dict[Prior, Callable] = {
            Prior.SYMMETRY:     self._gen_symmetry_task,
            Prior.OBJECTNESS:   self._gen_objectness_task,
            Prior.NUMEROSITY:   self._gen_numerosity_task,
            Prior.GRAVITY:      self._gen_gravity_task,
            Prior.CONTAINMENT:  self._gen_containment_task,
            Prior.CAUSALITY:    self._gen_causality_task,
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_task(
        self,
        level: DifficultyLevel = DifficultyLevel.L1,
        domain: Optional[TaskDomain] = None
    ) -> ARCTask:
        """Generate a unique ARC task at the specified difficulty level and domain."""
        n_priors = level.value
        
        # Filter priors based on requested domain
        all_priors = list(self._prior_generators.keys())
        if domain == TaskDomain.A_SPATIAL:
            all_priors = [Prior.SYMMETRY]
        elif domain == TaskDomain.B_TOPOLOGICAL:
            all_priors = [Prior.OBJECTNESS, Prior.CONTAINMENT, Prior.GRAVITY]
        elif domain == TaskDomain.C_ABSTRACT:
            all_priors = [Prior.NUMEROSITY, Prior.CAUSALITY, Prior.GOAL]
            
        chosen_priors = self.rng.sample(all_priors, min(n_priors, len(all_priors)))

        for attempt in range(100):
            task = self._compose_task(chosen_priors, level)
            if task.fingerprint not in self._fingerprint_log:
                self._fingerprint_log.add(task.fingerprint)
                self._task_counter += 1
                return task

        raise RuntimeError(
            "Could not generate a unique task after 100 attempts. "
            "Try a higher difficulty or a new Universe seed."
        )

    def generate_curriculum(
        self,
        n_tasks: int = 10,
        start_level: DifficultyLevel = DifficultyLevel.L1,
        max_level: DifficultyLevel = DifficultyLevel.L5,
    ) -> List[ARCTask]:
        """Generate a curriculum of tasks that progressively increase in difficulty."""
        tasks = []
        level_values = list(DifficultyLevel)
        start_idx = level_values.index(start_level)
        max_idx = level_values.index(max_level)

        for i in range(n_tasks):
            level_idx = min(start_idx + (i // max(1, n_tasks // (max_idx - start_idx + 1))), max_idx)
            tasks.append(self.generate_task(level_values[level_idx]))
        return tasks

    @property
    def session_task_count(self) -> int:
        return self._task_counter

    # ── Task Composition ───────────────────────────────────────────────────

    def _compose_task(
        self, priors: List[Prior], level: DifficultyLevel
    ) -> ARCTask:
        """Compose a task by chaining transformations from multiple priors."""
        grid_h, grid_w = self.rng.choice(self.GRID_SIZES)
        n_train = 3

        # Build a base grid template + transformation chain
        base_fn, transform_fn, description = self._build_transform_chain(priors, grid_h, grid_w)

        train_pairs = []
        for _ in range(n_train):
            inp = base_fn()
            out = transform_fn(inp)
            train_pairs.append((inp.copy(), out.copy()))

        test_inp = base_fn()
        test_out = transform_fn(test_inp)

        # Fingerprint = hash of (description + all training outputs)
        fp_data = description + "".join(str(o.tolist()) for _, o in train_pairs)
        fingerprint = hashlib.sha256(fp_data.encode()).hexdigest()[:16]

        task_id = f"T{self._task_counter:04d}_{fingerprint}"

        return ARCTask(
            task_id=task_id,
            fingerprint=fingerprint,
            priors_used=priors,
            difficulty=level,
            train_pairs=train_pairs,
            test_input=test_inp,
            test_output=test_out,
            transformation_description=description,
        )

    def _build_transform_chain(
        self, priors: List[Prior], h: int, w: int
    ) -> Tuple[Callable, Callable, str]:
        """Build a randomized (base_grid_factory, transform, description) chain."""
        bg = 0
        parts = []
        transforms = []

        for prior in priors:
            if prior == Prior.SYMMETRY:
                op = self.rng.choice(["rotate90", "rotate180", "mirror_h", "mirror_v"])
                fn = getattr(GridTransforms, op)
                transforms.append(fn)
                parts.append(f"apply {op} to all objects")

            elif prior == Prior.GRAVITY:
                direction = self.rng.choice(["down", "up"])
                fn = GridTransforms.gravity_down if direction == "down" else GridTransforms.gravity_up
                transforms.append(fn)
                parts.append(f"gravity pulls all cells {direction}")

            elif prior == Prior.NUMEROSITY:
                transforms.append(GridTransforms.majority_recolor)
                parts.append("recolor everything to the majority color")

            elif prior == Prior.CONTAINMENT:
                color = self.rng.choice(self.COLORS)
                fn = lambda g, c=color: GridTransforms.fill_enclosed(g, fill_color=c)
                transforms.append(fn)
                parts.append(f"fill all enclosed holes with color {color}")

            elif prior == Prior.CAUSALITY:
                from_c = self.rng.choice(self.COLORS[:5])
                to_c = self.rng.choice(self.COLORS[5:])
                fn = lambda g, fc=from_c, tc=to_c: GridTransforms.recolor(g, fc, tc)
                transforms.append(fn)
                parts.append(f"color {from_c} causes all cells to become color {to_c}")

            elif prior == Prior.OBJECTNESS:
                transforms.append(GridTransforms.sort_objects_by_size)
                parts.append("sort objects left-to-right by ascending size")

        # Compose transforms into a single function call chain
        def composed_transform(grid: np.ndarray) -> np.ndarray:
            result = grid.copy()
            for fn in transforms:
                result = fn(result)
            return result

        def make_base_grid() -> np.ndarray:
            return self._make_sparse_grid(h, w, bg)

        description = " → ".join(parts) if parts else "identity"
        return make_base_grid, composed_transform, description

    # ── Grid Factories ─────────────────────────────────────────────────────

    def _make_sparse_grid(
        self,
        h: int,
        w: int,
        bg: int = 0,
        n_objects: Optional[int] = None,
    ) -> np.ndarray:
        """Create a sparse grid with a few colored rectangular/point objects."""
        grid = np.full((h, w), bg, dtype=np.int32)
        n = n_objects or self.rng.randint(2, 5)
        used_colors = self.rng.sample(self.COLORS, min(n, len(self.COLORS)))

        for color in used_colors:
            obj_h = self.rng.randint(1, max(1, h // 3))
            obj_w = self.rng.randint(1, max(1, w // 3))
            r = self.rng.randint(0, h - obj_h)
            c = self.rng.randint(0, w - obj_w)
            grid[r:r+obj_h, c:c+obj_w] = color

        return grid

    # ── Prior Generators (named, for the registry) ─────────────────────────

    def _gen_symmetry_task(self, h: int, w: int) -> Tuple[Callable, Callable, str]:
        fn = self.rng.choice([
            GridTransforms.rotate90, GridTransforms.mirror_h, GridTransforms.mirror_v
        ])
        return lambda: self._make_sparse_grid(h, w), fn, f"apply {fn.__name__}"

    def _gen_objectness_task(self, h: int, w: int) -> Tuple[Callable, Callable, str]:
        return lambda: self._make_sparse_grid(h, w), GridTransforms.sort_objects_by_size, "sort objects by size"

    def _gen_numerosity_task(self, h: int, w: int) -> Tuple[Callable, Callable, str]:
        return lambda: self._make_sparse_grid(h, w), GridTransforms.majority_recolor, "majority recolor"

    def _gen_gravity_task(self, h: int, w: int) -> Tuple[Callable, Callable, str]:
        return lambda: self._make_sparse_grid(h, w), GridTransforms.gravity_down, "gravity down"

    def _gen_containment_task(self, h: int, w: int) -> Tuple[Callable, Callable, str]:
        color = self.rng.choice(self.COLORS)
        return (
            lambda: self._make_sparse_grid(h, w),
            lambda g: GridTransforms.fill_enclosed(g, color),
            f"fill enclosed with {color}",
        )

    def _gen_causality_task(self, h: int, w: int) -> Tuple[Callable, Callable, str]:
        fc, tc = self.rng.sample(self.COLORS, 2)
        return (
            lambda: self._make_sparse_grid(h, w),
            lambda g: GridTransforms.recolor(g, fc, tc),
            f"recolor {fc} → {tc}",
        )


# ─── OBJECT PERCEPTION UTILITIES (used by Agent 1 — the Perceiver) ────────────

def perceive_objects(grid: np.ndarray, bg: int = 0) -> List[GridObject]:
    """
    Segment a grid into discrete GridObject instances.
    Uses 4-connectivity flood-fill labeling.
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        raise ImportError("scipy is required. Run: pip install scipy")

    objects = []
    obj_id = 0

    for color in np.unique(grid):
        if color == bg:
            continue
        mask = (grid == color).astype(int)
        labeled, n = label(mask)

        for lab in range(1, n + 1):
            cell_positions = [
                (int(r), int(c)) for r, c in zip(*np.where(labeled == lab))
            ]
            rows = [p[0] for p in cell_positions]
            cols = [p[1] for p in cell_positions]
            bbox = (min(rows), min(cols), max(rows), max(cols))

            objects.append(GridObject(
                id=obj_id,
                color=int(color),
                cells=cell_positions,
                bbox=bbox,
                size=len(cell_positions),
            ))
            obj_id += 1

    return objects


def grid_fingerprint(grid: np.ndarray) -> str:
    """Compute a short fingerprint of a numpy grid for deduplication."""
    return hashlib.md5(grid.tobytes()).hexdigest()[:12]


# ─── SELF-TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Universe Self-Test")
    print("=" * 60)

    universe = Universe(seed=42)

    for level in DifficultyLevel:
        task = universe.generate_task(level)
        print(f"\n[{level.name}] {task.task_id}")
        print(f"  Priors:   {[p.value for p in task.priors_used]}")
        print(f"  Rule:     {task.transformation_description}")
        print(f"  Grid:     {task.test_input.shape}")
        print(f"  Train ex: {len(task.train_pairs)}")

        objects = perceive_objects(task.test_input)
        print(f"  Objects in test_input: {len(objects)}")

    print(f"\n✓ Generated {universe.session_task_count} unique tasks. No fingerprint collisions.")
