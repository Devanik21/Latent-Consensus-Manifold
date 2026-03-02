"""
meta_learner.py — Bayesian Prior Updater for Learning to Learn
================================================================
Tracks the 64-dimensional latent space to identify which dimensions
historically lead to successful task solutions. 
Biases future sampling (Dreamer) and searches (Scientist) towards 
these empirically proven "cognitive" sub-spaces.
"""

import numpy as np
from typing import Dict

# Import LATENT_DIM from the dictionary config 
from latent_dictionary import LATENT_DIM


class MetaLearner:
    """
    Lightweight Bayesian Prior mechanism.
    Maintains a Beta distribution (or exponential moving average) 
    over the relevance of each of the 64 latent dimensions.
    """

    def __init__(self, n_components: int = LATENT_DIM, learning_rate: float = 0.1):
        self.n_components = n_components
        self.learning_rate = learning_rate
        
        # Starts with a uniform prior (all dimensions equally likely to be useful)
        # Values are bounded [0.01, 1.0] to represent probability/importance
        self.priors = np.full(n_components, 0.5, dtype=np.float32)
        
        self.total_updates = 0

    def update(self, winning_z: np.ndarray, rounds_to_solve: int):
        """
        Update priors based on a successful solution.
        Dimensions that are highly active in winning_z get boosted.
        Faster solutions (fewer rounds) apply a stronger boost.
        """
        # Normalize the winning z to see relative dimension importance
        max_val = np.max(winning_z)
        if max_val < 1e-5:
            return  # Empty z vector, nothing to learn from
            
        z_norm = winning_z / max_val
        
        # Calculate learning signal strength based on efficiency
        # Solved in 1 round = max boost (1.0). Solved in max rounds = min boost (0.2)
        efficiency_signal = max(0.2, 1.0 / max(1, rounds_to_solve))
        
        # Current effective learning rate
        alpha = self.learning_rate * efficiency_signal
        
        # Exponential Moving Average update
        # If z_norm[i] is high, prior goes up. If z_norm[i] is low, prior decays slightly.
        self.priors = (1 - alpha) * self.priors + (alpha * z_norm)
        
        # Clamp to prevent complete death of a dimension (exploration floor)
        self.priors = np.clip(self.priors, 0.05, 1.0)
        self.total_updates += 1

    def get_prior_z(self) -> np.ndarray:
        """
        Return the current expectation vector for z.
        Agents can use this to bias random samples or initialize search.
        """
        return self.priors.copy()

    def stats(self) -> Dict:
        """Dashboard telemetry."""
        return {
            "updates": self.total_updates,
            "mean_prior": float(np.mean(self.priors)),
            "max_prior": float(np.max(self.priors)),
            "active_dimensions": int(np.sum(self.priors > 0.3))
        }
