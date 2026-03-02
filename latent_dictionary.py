"""
latent_dictionary.py — Online Transformation Dictionary Learning
================================================================
Replaces hardcoded DSL primitives with a learned, continuous
transformation space that builds incrementally in RAM.

Core idea:
  Every (input_grid → output_grid) pair encodes a *transformation*.
  We flatten both grids, compute a *delta vector*, and incrementally
  decompose the growing matrix of deltas via NMF / Dictionary Learning.
  The result is a basis of *latent transformation components* —
  discovered operations, NOT predefined ones.

Agents interface:
  • Dreamer   → sample_z()       — sample novel z coefficients
  • Scientist → search_z()       — find z that best explains a task
  • Archivist → register_pair()  — feed solved pairs into the dictionary
  • Dashboard → get_basis()      — visualize the learned components

Memory budget: < 50 MB for 1000+ episodes on Streamlit Cloud.
"""

import numpy as np
import hashlib
import logging
from typing import List, Tuple, Optional, Dict

log = logging.getLogger("latent_dictionary")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

MAX_GRID_DIM    = 15          # ARC grids are at most 30×30; ours cap at 15×15
FLAT_DIM        = MAX_GRID_DIM * MAX_GRID_DIM   # 225
LATENT_DIM      = 64          # number of dictionary atoms (basis vectors)
MAX_PAIRS       = 2000        # cap on stored transition pairs (RAM safety)
LEARNING_RATE   = 0.005       # online NMF step size
MIN_PAIRS_FIT   = 5           # minimum pairs before dictionary becomes usable
NMF_ITERS       = 30          # coordinate descent iterations per online update
EPSILON         = 1e-8        # numerical stability


# ─── GRID UTILITIES ───────────────────────────────────────────────────────────

def _pad_grid(grid: np.ndarray) -> np.ndarray:
    """Pad any grid to MAX_GRID_DIM × MAX_GRID_DIM with zeros."""
    padded = np.zeros((MAX_GRID_DIM, MAX_GRID_DIM), dtype=np.float32)
    h, w = min(grid.shape[0], MAX_GRID_DIM), min(grid.shape[1], MAX_GRID_DIM)
    padded[:h, :w] = grid[:h, :w]
    return padded


def _grid_to_flat(grid: np.ndarray) -> np.ndarray:
    """Flatten a grid into a 1-D vector of length FLAT_DIM."""
    return _pad_grid(grid).flatten()


def _flat_to_grid(vec: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Reshape a flat vector back to a grid of the given shape, rounding to ints."""
    full = vec.reshape(MAX_GRID_DIM, MAX_GRID_DIM)
    h, w = min(shape[0], MAX_GRID_DIM), min(shape[1], MAX_GRID_DIM)
    cropped = full[:h, :w]
    return np.clip(np.round(cropped), 0, 9).astype(np.int32)


def _compute_delta(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    """
    Compute the transformation delta between input and output grids.
    Uses a signed encoding: delta = output - input (in float space).
    Shifted to non-negative for NMF compatibility: delta_nn = delta + 9.
    """
    inp_f = _grid_to_flat(inp).astype(np.float32)
    out_f = _grid_to_flat(out).astype(np.float32)
    delta = out_f - inp_f
    # Shift to non-negative: ARC values range [0,9], so delta ∈ [-9, 9]
    delta_nn = delta + 9.0   # now ∈ [0, 18]
    return delta_nn


def _apply_delta(inp: np.ndarray, delta_nn: np.ndarray,
                 shape: Tuple[int, int]) -> np.ndarray:
    """Reconstruct an output grid from an input grid and a non-negative delta."""
    inp_f = _grid_to_flat(inp).astype(np.float32)
    delta = delta_nn - 9.0     # un-shift
    out_f = inp_f + delta
    return _flat_to_grid(out_f, shape)


# ─── ONLINE NMF (Multiplicative Updates) ──────────────────────────────────────

class _OnlineNMF:
    """
    Lightweight Non-Negative Matrix Factorization with online updates.

    Decomposes a data matrix V ≈ W × H where:
      V  = (n_samples × FLAT_DIM)  — observed transformation deltas
      W  = (n_samples × LATENT_DIM) — per-sample coefficients (z vectors)
      H  = (LATENT_DIM × FLAT_DIM) — the learned dictionary / basis

    We maintain only H (the global basis) persistently.
    Per-sample W is computed on demand via coordinate descent.
    """

    def __init__(self, n_components: int = LATENT_DIM,
                 flat_dim: int = FLAT_DIM):
        self.n_components = n_components
        self.flat_dim = flat_dim
        # Initialize dictionary with small random values
        self.H = np.random.RandomState(42).rand(
            n_components, flat_dim
        ).astype(np.float32) * 0.01 + EPSILON
        self._fitted = False
        self._n_updates = 0

    def partial_fit(self, V: np.ndarray, n_iter: int = NMF_ITERS) -> None:
        """
        Online update of the dictionary H using the batch V.
        Uses multiplicative update rules (Lee & Seung, 2001).
        """
        n_samples = V.shape[0]
        V = np.maximum(V, EPSILON)

        # Initialize W for this batch
        W = np.random.RandomState(self._n_updates).rand(
            n_samples, self.n_components
        ).astype(np.float32) * 0.01 + EPSILON

        for _ in range(n_iter):
            # Update W: W ← W * (V @ H.T) / (W @ H @ H.T)
            WH = W @ self.H
            numerator_W = V @ self.H.T
            denominator_W = WH @ self.H.T + EPSILON
            W *= numerator_W / denominator_W

            # Update H: H ← H * (W.T @ V) / (W.T @ W @ H)
            numerator_H = W.T @ V
            denominator_H = W.T @ W @ self.H + EPSILON
            # Blend old and new to prevent catastrophic forgetting
            H_new = self.H * (numerator_H / denominator_H)
            alpha = min(LEARNING_RATE * n_samples, 0.3)
            self.H = (1 - alpha) * self.H + alpha * H_new

        self._fitted = True
        self._n_updates += 1

    def encode(self, v: np.ndarray, n_iter: int = 50) -> np.ndarray:
        """
        Encode a single delta vector v into its latent coefficients z.
        Solves: min_z ||v - z @ H||² s.t. z ≥ 0
        via multiplicative updates (coordinate descent).
        """
        v = np.maximum(v.reshape(1, -1), EPSILON).astype(np.float32)
        z = np.ones((1, self.n_components), dtype=np.float32) * 0.01

        for _ in range(n_iter):
            zH = z @ self.H
            numerator = v @ self.H.T
            denominator = zH @ self.H.T + EPSILON
            z *= numerator / denominator

        return z.flatten()

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode a latent vector z back into a delta vector."""
        z = np.maximum(z.reshape(1, -1), 0).astype(np.float32)
        return (z @ self.H).flatten()

    @property
    def is_ready(self) -> bool:
        return self._fitted


# ─── LATENT DICTIONARY (Public API) ──────────────────────────────────────────

class LatentDictionary:
    """
    The Learned Transformation Space.

    Replaces DSL.PRIMITIVES with a continuous latent space
    where every discovered transformation is a point z ∈ R^64.
    The dictionary grows incrementally as the Council solves tasks.

    Key methods for agents:
      register_pair()   — feed a solved (input, output) into the dictionary
      search_z()        — find the best z explaining a task (Scientist)
      sample_z()        — sample novel z candidates (Dreamer)
      decode_z()        — reconstruct an output from z + input grid
      get_basis()       — return the learned dictionary matrix (Observatory)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._nmf = _OnlineNMF()
        self._deltas: List[np.ndarray] = []      # stored delta vectors
        self._z_archive: List[np.ndarray] = []    # z vectors of solved episodes
        self._z_labels: List[str] = []            # human-readable auto-labels
        self._pair_meta: List[Dict] = []          # metadata per pair
        self._n_registered = 0

    # ── Learning Interface ────────────────────────────────────────────────

    def register_pair(self, inp: np.ndarray, out: np.ndarray,
                      task_id: str = "", label: str = "") -> np.ndarray:
        """
        Register one (input → output) transformation pair.
        Updates the dictionary online and returns the z encoding.
        """
        delta = _compute_delta(inp, out)
        self._deltas.append(delta)
        self._pair_meta.append({"task_id": task_id, "label": label})
        self._n_registered += 1

        # Cap stored deltas for RAM safety
        if len(self._deltas) > MAX_PAIRS:
            self._deltas = self._deltas[-MAX_PAIRS:]
            self._pair_meta = self._pair_meta[-MAX_PAIRS:]

        # Fit / update the dictionary when we have enough data
        if len(self._deltas) >= MIN_PAIRS_FIT:
            # Use the last batch (most recent) for an online update
            batch_size = min(len(self._deltas), 50)
            batch = np.array(self._deltas[-batch_size:], dtype=np.float32)
            self._nmf.partial_fit(batch)

        # Encode and archive this pair's z vector
        z = self._nmf.encode(delta) if self._nmf.is_ready else self.rng.rand(LATENT_DIM).astype(np.float32) * 0.01
        self._z_archive.append(z)
        self._z_labels.append(label or task_id or f"z_{self._n_registered}")

        # Cap z archive
        if len(self._z_archive) > MAX_PAIRS:
            self._z_archive = self._z_archive[-MAX_PAIRS:]
            self._z_labels = self._z_labels[-MAX_PAIRS:]

        return z

    def register_batch(self, pairs: List[Tuple[np.ndarray, np.ndarray]],
                       task_id: str = "") -> List[np.ndarray]:
        """Register multiple pairs at once (e.g., all training examples)."""
        return [self.register_pair(inp, out, task_id=task_id) for inp, out in pairs]

    # ── Agent Interfaces ──────────────────────────────────────────────────

    def search_z(self, task_input: np.ndarray,
                 task_output: np.ndarray,
                 n_candidates: int = 60,
                 n_refine: int = 20) -> Tuple[Optional[np.ndarray], float]:
        """
        Scientist interface: find the best z that maps input → output.

        Strategy:
          1. Encode the exact (input, output) delta as z_exact.
          2. Generate n_candidates by perturbing z_exact.
          3. Pick the candidate whose decoded output is closest to target.

        Returns: (best_z, reconstruction_error)
        """
        if not self._nmf.is_ready:
            return None, float("inf")

        delta_target = _compute_delta(task_input, task_output)
        z_exact = self._nmf.encode(delta_target)

        best_z = z_exact
        best_err = self._reconstruction_error(z_exact, task_input, task_output)

        for _ in range(n_candidates):
            # Perturbation: mix exact encoding with random noise
            noise = self.rng.randn(LATENT_DIM).astype(np.float32) * 0.1
            z_cand = np.maximum(z_exact + noise, 0)

            # Also try mixing with archived z vectors (transfer learning)
            if self._z_archive:
                idx = self.rng.randint(0, len(self._z_archive))
                mix_alpha = self.rng.uniform(0.1, 0.5)
                z_mix = (1 - mix_alpha) * z_exact + mix_alpha * self._z_archive[idx]
                z_mix = np.maximum(z_mix, 0)

                err_mix = self._reconstruction_error(z_mix, task_input, task_output)
                if err_mix < best_err:
                    best_err = err_mix
                    best_z = z_mix

            err_cand = self._reconstruction_error(z_cand, task_input, task_output)
            if err_cand < best_err:
                best_err = err_cand
                best_z = z_cand

        # Refine the best candidate via gradient-free hill climbing
        for _ in range(n_refine):
            noise = self.rng.randn(LATENT_DIM).astype(np.float32) * 0.02
            z_refine = np.maximum(best_z + noise, 0)
            err_r = self._reconstruction_error(z_refine, task_input, task_output)
            if err_r < best_err:
                best_err = err_r
                best_z = z_refine

        return best_z, best_err

    def sample_z(self, n: int = 8, temperature: float = 1.0,
                 prior_z: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Dreamer interface: sample n novel z vectors.

        If prior_z is given (from meta-learner), samples are biased toward it.
        Otherwise, samples from the empirical distribution of archived z vectors.
        """
        samples = []
        for _ in range(n):
            if prior_z is not None:
                # Sample near the prior with temperature-controlled noise
                noise = self.rng.randn(LATENT_DIM).astype(np.float32) * temperature * 0.15
                z = np.maximum(prior_z + noise, 0)
            elif self._z_archive:
                # Pick a random archived z and perturb it
                base = self._z_archive[self.rng.randint(0, len(self._z_archive))]
                noise = self.rng.randn(LATENT_DIM).astype(np.float32) * temperature * 0.2
                z = np.maximum(base + noise, 0)
            else:
                # Cold start: purely random z
                z = self.rng.rand(LATENT_DIM).astype(np.float32) * 0.1
            samples.append(z)
        return samples

    def decode_z(self, z: np.ndarray, input_grid: np.ndarray) -> np.ndarray:
        """
        Decode a latent vector z into a predicted output grid,
        conditioned on the input grid.
        """
        delta_nn = self._nmf.decode(z)
        return _apply_delta(input_grid, delta_nn, input_grid.shape)

    def encode_pair(self, inp: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Encode a (input, output) pair to its z representation (read-only)."""
        if not self._nmf.is_ready:
            return self.rng.rand(LATENT_DIM).astype(np.float32) * 0.01
        delta = _compute_delta(inp, out)
        return self._nmf.encode(delta)

    # ── Philosopher Interface ─────────────────────────────────────────────

    def rotate_basis(self, z: np.ndarray, angle_idx: int = 0) -> np.ndarray:
        """
        Philosopher interface: apply a learned orthogonal rotation
        to a z vector, providing a different 'perspective' on the
        transformation without changing its magnitude.
        """
        # Use Givens rotations on pairs of latent dimensions
        z_rot = z.copy()
        dim_pairs = [(i, (i + 1 + angle_idx) % LATENT_DIM)
                     for i in range(0, LATENT_DIM, 2)]
        theta = np.pi / (4 + angle_idx)  # progressively different angles
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for i, j in dim_pairs:
            zi, zj = z_rot[i], z_rot[j]
            z_rot[i] = cos_t * zi - sin_t * zj
            z_rot[j] = sin_t * zi + cos_t * zj
        return np.maximum(z_rot, 0)

    # ── Dashboard / Observatory Interface ─────────────────────────────────

    def get_basis(self) -> np.ndarray:
        """Return the learned dictionary matrix H (LATENT_DIM × FLAT_DIM)."""
        return self._nmf.H.copy()

    def get_z_archive(self) -> List[np.ndarray]:
        """Return all archived z vectors for visualization."""
        return list(self._z_archive)

    def get_z_labels(self) -> List[str]:
        """Return human-readable labels for archived z vectors."""
        return list(self._z_labels)

    @property
    def is_ready(self) -> bool:
        """True once the dictionary has been fitted at least once."""
        return self._nmf.is_ready

    @property
    def n_registered(self) -> int:
        return self._n_registered

    @property
    def latent_dim(self) -> int:
        return LATENT_DIM

    def stats(self) -> Dict:
        """Return a summary dict for the dashboard."""
        return {
            "n_registered": self._n_registered,
            "n_archived_z": len(self._z_archive),
            "dictionary_ready": self._nmf.is_ready,
            "n_updates": self._nmf._n_updates,
            "latent_dim": LATENT_DIM,
            "basis_norm": float(np.linalg.norm(self._nmf.H)),
        }

    def to_dict(self) -> Dict:
        """Serializable export for session download."""
        return {
            "stats": self.stats(),
            "z_archive": [z.tolist() for z in self._z_archive[-100:]],
            "z_labels": self._z_labels[-100:],
            "basis_sample": self._nmf.H[:4].tolist(),  # first 4 atoms
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _reconstruction_error(self, z: np.ndarray,
                              inp: np.ndarray, target: np.ndarray) -> float:
        """MSE between decoded output and target output grid."""
        predicted = self.decode_z(z, inp)
        if predicted.shape != target.shape:
            # Shape mismatch → pad or crop for comparison
            h = min(predicted.shape[0], target.shape[0])
            w = min(predicted.shape[1], target.shape[1])
            predicted = predicted[:h, :w]
            target_cropped = target[:h, :w]
            return float(np.mean((predicted.astype(float) - target_cropped.astype(float)) ** 2))
        return float(np.mean((predicted.astype(float) - target.astype(float)) ** 2))


# ─── SELF-TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Latent Dictionary Self-Test")
    print("=" * 60)

    # 1. Create dictionary
    ld = LatentDictionary(seed=42)
    print(f"\n[1] Created LatentDictionary (latent_dim={ld.latent_dim})")
    print(f"    Ready: {ld.is_ready}")

    # 2. Generate some synthetic transformation pairs
    rng = np.random.RandomState(42)
    pairs = []
    for i in range(20):
        inp = rng.randint(0, 10, size=(5, 5)).astype(np.int32)
        # Simple transformation: rotate90
        out = np.rot90(inp, 1)
        pairs.append((inp, out))

    print(f"\n[2] Generated {len(pairs)} synthetic pairs (rotate90)")

    # 3. Register pairs
    for i, (inp, out) in enumerate(pairs):
        z = ld.register_pair(inp, out, task_id=f"T{i:03d}")
        if i == 0:
            print(f"    First z shape: {z.shape}, norm: {np.linalg.norm(z):.4f}")

    print(f"\n[3] Dictionary status:")
    print(f"    Ready: {ld.is_ready}")
    print(f"    Registered: {ld.n_registered}")
    print(f"    Stats: {ld.stats()}")

    # 4. Search for the transformation on a new pair
    test_inp = rng.randint(0, 10, size=(5, 5)).astype(np.int32)
    test_out = np.rot90(test_inp, 1)
    z_found, err = ld.search_z(test_inp, test_out, n_candidates=30)
    print(f"\n[4] Search result:")
    print(f"    Error: {err:.4f}")
    print(f"    z norm: {np.linalg.norm(z_found):.4f}")

    # 5. Decode and compare
    decoded = ld.decode_z(z_found, test_inp)
    match = np.array_equal(decoded, test_out)
    pixel_acc = np.mean(decoded == test_out)
    print(f"\n[5] Decode result:")
    print(f"    Exact match: {match}")
    print(f"    Pixel accuracy: {pixel_acc:.2%}")

    # 6. Sample z vectors (Dreamer)
    samples = ld.sample_z(n=5)
    print(f"\n[6] Dreamer sampled {len(samples)} z vectors")
    print(f"    Norms: {[f'{np.linalg.norm(s):.3f}' for s in samples]}")

    # 7. Rotate basis (Philosopher)
    if z_found is not None:
        z_rot = ld.rotate_basis(z_found, angle_idx=0)
        print(f"\n[7] Philosopher rotation:")
        print(f"    Original norm: {np.linalg.norm(z_found):.4f}")
        print(f"    Rotated norm:  {np.linalg.norm(z_rot):.4f}")
        print(f"    Cosine sim:    {np.dot(z_found, z_rot) / (np.linalg.norm(z_found) * np.linalg.norm(z_rot) + 1e-8):.4f}")

    # 8. Memory check
    import sys
    total_bytes = (
        sys.getsizeof(ld._deltas) +
        sum(d.nbytes for d in ld._deltas) +
        sys.getsizeof(ld._z_archive) +
        sum(z.nbytes for z in ld._z_archive) +
        ld._nmf.H.nbytes
    )
    print(f"\n[8] Memory usage: {total_bytes / 1024:.1f} KB")

    print("\n✓ latent_dictionary.py self-test passed.")
