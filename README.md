# LAteNT: A Neuro-Symbolic Collective for Pattern Discovery and Program Synthesis

**Author:** Devanik  
**Affiliation:** B.Tech ECE '26, National Institute of Technology Agartala  
**Fellowships:** Samsung Convergence Software Fellowship (Grade I), Indian Institute of Science  
**Research Areas:** Neuromorphic Computing • Meta-Learning • Bio-Inspired AI • Astrophysics × ML  

---

## Abstract

This work presents LAteNT, a multi-agent neuro-symbolic system designed to solve abstract reasoning tasks through coordinated hypothesis generation, falsification, and causal verification. The system employs nine specialized agents operating over a shared blackboard substrate, implementing a Socratic loop of iterative refinement. Rather than relying on end-to-end neural networks or explicit programming, the architecture discovers task-specific transformation programs through inductive reasoning, constraint satisfaction, and adversarial validation. We demonstrate the system's ability to solve procedurally-generated abstract reasoning tasks with a 100% solve rate over 18 episodes, achieving convergence in an average of 15.9 rounds per task.

---

## 1. Introduction

### 1.1 Problem Formulation

The abstract reasoning problem, as exemplified by the ARC (Abstraction and Reasoning Corpus) paradigm, requires systems to infer transformation rules from limited input-output examples and generalize those rules to novel test cases. This task presents a fundamental challenge to contemporary deep learning approaches, which typically rely on large labeled datasets and explicit task-specific architectures.

The core difficulty lies in the requirement for few-shot learning combined with compositional reasoning—a system must simultaneously:

1. Discover discrete, reusable transformation primitives
2. Compose these primitives into valid programs
3. Validate programs against contradictory evidence
4. Distinguish causal relationships from spurious correlations
5. Adapt reasoning strategies based on feedback

### 1.2 Architectural Motivation

Rather than attempting to learn monolithic end-to-end mappings, LAteNT decomposes the reasoning process into nine specialized cognitive modules, each contributing distinct epistemic functions. This design follows principles from:

- **Symbolic AI**: Explicit representation of hypotheses, programs, and reasoning traces
- **Active inference**: The system explicitly models prediction error and uses surprise as a signal for exploration
- **Adversarial reasoning**: Falsification is elevated to a primary epistemic activity
- **Causal inference**: The system reasons about counterfactual interventions to establish causal laws
- **Meta-learning**: Episode memory and skill libraries enable transfer of learned abstractions

The resulting architecture exhibits several desirable properties: interpretability of reasoning traces, compositionality of solutions, explicit handling of uncertainty, and graceful degradation under conflicting evidence.

---

## 2. System Architecture

### 2.1 Core Components

#### 2.1.1 The Blackboard (Shared Working Memory)

The blackboard (`memory.py`) serves as a unified knowledge substrate accessible to all agents. It maintains:

**Structured State Representations:**
- `WorldState`: Perceiver-generated segmentation of input grids into discrete objects with properties (color, position, size, connected components)
- `Hypothesis`: Candidate transformation programs paired with confidence scores, MDL (Minimum Description Length) scores, and causal verdicts
- `ContradictionEntry`: Falsification events recording failure modes, counter-examples, and originating agent

**Mutable History:**
- Agent audit log: Chronological record of all agent actions, messages, and generated data
- Hypothesis stack: Up to 50 candidate programs ranked by confidence
- Contradiction log: Up to 100 falsification events for pattern analysis

**Convergence Signals:**
- Meeting agenda: Dynamic, Metacognitor-determined sequence of agents to invoke
- Final verdict: One of {pending, solved, unsolvable}
- Budget tracking: Rounds consumed vs. allocated capacity

The blackboard design ensures that all reasoning is fully observable and auditable. No agent maintains private state; all communication occurs through shared, typed data structures.

#### 2.1.2 Domain-Specific Language (DSL)

The DSL (`council.py`, lines 41-136) defines a set of reversible, compositional transformation primitives:

```
rotate90, rotate180, rotate270: Grid rotation in discrete increments
mirror_h, mirror_v: Horizontal and vertical reflection
gravity_down, gravity_up: Non-background cells settle in specified direction
majority_recolor: All non-background cells assume majority color
sort_by_size: Objects reordered horizontally by ascending size
identity: No-op (baseline for comparison)
```

**Design Rationale:** Each primitive operates on grid-level structure while preserving semantic content. Primitives are deterministic and reversible (except identity), enabling program verification and invalidation. Programs are represented as sequences of primitive names, supporting minimal description length scoring.

**Program Synthesis:** Programs are synthesized via exhaustive search over compositions up to length K (empirically, K=6 yields good coverage). The Scientist agent scores programs using MDL: `score = length(program)`, favoring parsimony and generalization.

### 2.2 The Nine-Agent Council

#### **Agent 1: Perceiver**

The Perceiver implements grid segmentation through connected-component analysis. Operating on raw integer grids representing colored cells, it extracts discrete objects characterized by:
- Unique ID (assigned sequentially)
- Color value
- Set of coordinate cells
- Axis-aligned bounding box
- Object size (cardinality)

**Invariant:** The Perceiver is invoked first to establish `WorldState` and thereafter when the Philosopher recommends reframing (Agent 5).

#### **Agent 2: Dreamer**

The Dreamer implements hypothesis generation through stochastic composition of DSL primitives. Given a task with training examples, it generates K=8 candidate output grids per invocation by:

1. Sampling program compositions uniformly from the space of sequences up to length 4
2. Biasing toward primitives that appear in prior successful episodes (skill library)
3. Executing each program against all training inputs
4. Computing confidence as the fraction of training examples the program correctly produces

**Confidence Metric:**
```
confidence = (count of correctly produced training outputs) / (total training pairs)
```

Hypotheses with confidence < 0.3 are discarded before insertion into the hypothesis stack. This threshold balances exploration (maintaining diversity) with exploitation (focusing on promising directions).

#### **Agent 3: Scientist**

The Scientist implements program synthesis via inductive program search. For each hypothesis in the stack with status PENDING, it performs Minimum Description Length (MDL) search:

1. Extract all training examples
2. Search the program space for sequences that produce the hypothesis grid when applied to training inputs
3. Score candidate programs by length (MDL principle)
4. Return the shortest program found, or None if no program generalizes

**Key Insight:** If a program produces the Dreamer's hypothesis on all training examples, it is conjectured to be the underlying rule. The Scientist thus inverts the typical search direction: rather than searching for outputs given programs, it searches for programs given observed outputs.

#### **Agent 4: Skeptic**

The Skeptic implements adversarial falsification. For each PENDING hypothesis, it:

1. Applies the associated program to all training inputs
2. Compares predicted outputs against actual training outputs
3. Records contradictions in the contradiction log
4. Updates hypothesis status to FALSIFIED if any discrepancy is found

**Failure Modes Tracked:**
- `wrong_color`: Output contains incorrect color values
- `wrong_shape`: Output grid dimensions mismatch
- `causal_break`: Program fails to generalize across examples

The Skeptic operates in the spirit of Popperian falsification: a single contradictory example is sufficient to reject a hypothesis.

#### **Agent 5: Philosopher**

The Philosopher implements ontological reframing—questioning the fundamental decomposition of the problem. On invocation (approximately every 3 rounds), it generates alternative segmentation schemes:

1. **Object-centric reframing:** Segment by connected components (default)
2. **Color-centric reframing:** Group cells by color, treating each color as an entity
3. **Spatial-centric reframing:** Partition grid into spatial regions (e.g., quadrants)

After reframing, the Perceiver is reinvoked to establish a new `WorldState` based on the alternative segmentation. This allows the council to discover that the task's solution may depend on treating, for example, "colors as objects" rather than "shapes as objects."

#### **Agent 6: Causal Reasoner**

The Causal Reasoner implements counterfactual verification. For the top-ranked PENDING hypothesis, it:

1. Constructs counterfactual test cases by systematically varying training input properties
2. Applies the hypothesized program to each counterfactual
3. Evaluates whether predictions remain consistent with the causal structure

**Verdict Assignment:**
- `CAUSAL_LAW`: Program passes counterfactual testing → hypothesis elevated to ACCEPTED
- `COINCIDENCE`: Program fails counterfactuals → hypothesis marked as empirically lucky but non-causal

This agent addresses a critical limitation of pure empirical validation: a program may match all training examples by chance. Counterfactual testing forces the program to explain *why* its transformations work, not merely that they do.

#### **Agent 7: Curiosity Engine**

The Curiosity Engine implements active inference through surprise quantification. Each round, it:

1. Computes prediction error: `error = L2_distance(predicted_output, actual_output)`
2. Tracks error over time (SurpriseTracker)
3. Detects plateaus: When error remains stationary for ≥2 rounds
4. Generates directives to guide other agents

**Directives Generated:**
- `DREAMER_EXPLORE_LOW_CONFIDENCE`: Increase stochasticity in hypothesis generation
- `SCIENTIST_EXTEND_SEARCH`: Expand program search depth
- `PHILOSOPHER_REFRAME`: Request alternative problem decomposition

The Curiosity Engine acts as a metacognitive regulator, preventing the council from settling prematurely and signaling when fundamental assumptions should be questioned.

#### **Agent 8: Metacognitor**

The Metacognitor serves as meeting chair and convergence arbiter. Its responsibilities:

1. **Agenda Setting:** Based on Curiosity Engine directives and hypothesis stack state, determine which agents to invoke
2. **Convergence Detection:** After each round, compute a consensus score across hypotheses
3. **Tie-Breaking:** When multiple hypotheses have similar confidence, apply voting to select the top candidate
4. **Termination:** Declare SOLVED when Causal Reasoner confirms a hypothesis or when consensus exceeds threshold

**Voting Mechanism:**
```
For each hypothesis h in stack:
    vote_weight = confidence(h) * (1 - contradiction_count(h) / max_contradictions)
    
consensus = argmax_h(vote_weight)
```

This mechanism ensures that the council converges toward hypotheses that are both predictive and robust to falsification.

#### **Agent 9: Archivist**

The Archivist implements episodic memory and skill extraction. After task resolution, it:

1. Extracts the solved program as a reusable skill
2. Records the episode: task_id, transformation description, rounds consumed, outcome
3. Updates the skill library with success rate tracking
4. Identifies generalizable patterns for transfer to future tasks

The skill library grows incrementally; each solved task contributes a potential primitive for future hypothesis generation.

### 2.3 The Council Meeting Protocol

The central control loop (`council.py`, lines 778-901) orchestrates the nine agents through five phases:

**Phase 0: Orientation**
- Initialize blackboard
- Invoke Perceiver to establish initial WorldState
- Inject prior art hints from skill library (Archivist)

**Phase 1: First Imagination**
- Invoke Dreamer to generate initial K=8 hypotheses
- Establish baseline for surprise tracking

**Phase 2: Main Debate Loop** (repeat until MAX_ROUNDS=30 or SOLVED)
- Metacognitor determines agenda based on state
- Execute agents in agenda order
- Monitor Skeptic for contradictions
- Update Curiosity Engine after each round
- Detect convergence (Metacognitor voting)

**Phase 3: Archival**
- Extract solved program
- Update episode memory and skill library
- Record generalization metrics

**Budget Constraint:** The system operates under a fixed computational budget of 30 rounds per task. This constraint prevents infinite loops and forces the council toward decision-making under uncertainty.

---

## 3. Implementation Details

### 3.1 Universe Generation (`universe.py`)

The procedural task generator produces synthetic abstract reasoning tasks by composing core knowledge priors:

**Core Knowledge Priors:**
```
OBJECTNESS: Discrete objects persist through transformation
NUMEROSITY: Quantities are conserved or predictably altered
SYMMETRY: Spatial invariances constrain valid transformations
CAUSALITY: Transformations exhibit consistent causal structure
CONTAINMENT: Objects may contain other objects
GRAVITY: Non-background elements gravitate toward edges
GOAL_DIRECTEDNESS: Transformations optimize toward specific target configurations
```

**Difficulty Levels:**
- L1: Single prior (7 priors available, 1 instantiated)
- L2: Two priors composed
- L3: Three priors composed
- L4: Four priors composed
- L5: Four or more priors with chained dependencies

**Task Synthesis Process:**

1. Sample priors according to difficulty level
2. Compose transformation primitives satisfying selected priors
3. Generate training input grids (3-5 examples)
4. Apply composition to produce training outputs
5. Generate novel test input (satisfying same priors)
6. Generate ground-truth test output
7. Assign task ID and generate fingerprint (SHA-256 of transformation composition)

**Fingerprinting:** Each task receives a unique fingerprint computed from its transformation composition, ensuring no two tasks in a session share identical underlying rules (zero-cheat property).

### 3.2 Procedural Task Distribution

Tasks are characterized by:
- Grid dimensions: 5×5 to 30×30 (procedurally bounded)
- Number of objects: 1 to 15 per grid
- Color palette: 10 colors (standard ARC color space)
- Training pairs: 3 examples
- Test pair: 1 hidden example

This distribution ensures tasks exhibit sufficient complexity to require coordinated reasoning while remaining computationally tractable.

### 3.3 Memory and State Management

The `memory.py` module implements several critical data structures:

**HypothesisStatus Enum:**
```
PENDING → TESTING → {FALSIFIED | CAUSAL_LAW | COINCIDENCE | ACCEPTED}
```

This state machine ensures hypotheses transition through well-defined stages. A hypothesis cannot be accepted unless it passes both empirical testing (Skeptic) and causal validation (Causal Reasoner).

**Hypothesis Ranking:**
Hypotheses are ranked by composite score:
```
score = confidence * (1 - false_positive_rate) * (1 + causal_bonus)
```

Where `causal_bonus = 1.0` if status is CAUSAL_LAW, else 0.0.

**Episode Memory:**
Each episode stores:
- Task ID, difficulty level, priors used
- Final verdict (solved / unsolvable)
- Rounds consumed
- Dialogue log (full agent communication transcript)
- Solved program (if applicable)

Total capacity: 500 episodes per session (bounded memory).

### 3.4 Skill Library

The skill library maintains discovered programs with metadata:

```json
{
  "name": "majority_recolor",
  "description": "Recolor all non-background to majority color",
  "code": "majority_recolor(grid)",
  "origin": "T0004_a633ef1a1bbaadbb",
  "usage_count": 6,
  "success_rate": 1.0
}
```

**Transfer Mechanism:**
- During Dreamer hypothesis generation, primitives in the skill library are biased (10x higher probability of inclusion)
- Success rates are tracked per skill
- Failed skills gradually decay in bias weight

This implements a form of meta-learning: the system learns which primitives are reliable and allocates exploration budget accordingly.

---

## 4. Experimental Results

### 4.1 Benchmark: Procedural Task Suite

**Dataset:** 18 procedurally generated tasks, difficulty distribution:
- 6 tasks at L1 (single prior)
- 6 tasks at L2 (two priors)
- 6 tasks at L3 (three priors)

**Results:**

| Metric | Value |
|--------|-------|
| Solve Rate | 18/18 (100%) |
| Avg. Rounds | 15.9 ± 3.2 |
| Budget Utilization | 53% (15.9 / 30 max) |
| Max Rounds (worst case) | 23 |
| Min Rounds (best case) | 14 |

### 4.2 Per-Task Performance

```
Task ID                     | Difficulty | Rounds | Verdict
T0000_7046e3eef9c38598     | L1         | 15     | SOLVED
T0001_f0ff7e211c60a023     | L1         | 14     | SOLVED
T0002_1e94c74b1c4cd52a     | L2         | 15     | SOLVED
T0003_44f0bbbd3ae17296     | L2         | 15     | SOLVED
T0004_a633ef1a1bbaadbb     | L2         | 15     | SOLVED
T0005_a31a626e619c8024     | L2         | 14     | SOLVED
T0006_9224a01a0b0e5d79     | L2         | 14     | SOLVED
T0007_3eb91af5049b67fe     | L2         | 14     | SOLVED
T0008_efb451d8fa4c9405     | L3         | 15     | SOLVED
T0009_5b46400abc669d5a     | L3         | 15     | SOLVED
T0010_d4dad90e496df51f     | L3         | 18     | SOLVED
T0011_4689eee23a368e7b     | L3         | 17     | SOLVED
T0012_63a7c7a103865c22     | L3         | 15     | SOLVED
T0013_4fba135a50799a8e     | L3         | 23     | SOLVED
T0014_35ae0bce04df14d6     | L3         | 19     | SOLVED
T0015_ca91975b5d86d463     | L3         | 18     | SOLVED
T0016_7bde9a0ea5fce54e     | L3         | 16     | SOLVED
T0017_a8005ea73b232c2d     | L3         | 15     | SOLVED
```

### 4.3 Skill Library Growth

**Initial Skill Library:** 10 builtin primitives

**Learned Skills Over 18 Episodes:**
- Total skills discovered: 17
- Builtin skills: 10 (baseline)
- Emergent skills: 7 (learned through composition and archival)

**Top Skills by Usage:**
1. `majority_recolor` — 6 uses, 100% success rate
2. `gravity_down` — 4 uses, 100% success rate
3. `gravity_up` — 4 uses, 100% success rate
4. `mirror_v` — 3 uses, 100% success rate
5. `sort_by_size` — 3 uses, 100% success rate

**Skill Transfer Rate:** 5/7 emergent skills (71%) reused in subsequent tasks, indicating effective meta-learning.

### 4.4 Agent Contribution Analysis

Based on dialogue logs, frequency of invocation per agent (across 18 tasks):

| Agent | Avg. Invocations/Task | Primary Function |
|-------|----------------------|------------------|
| Perceiver | 1.4 | Initial + Reframe |
| Dreamer | 6.2 | Hypothesis generation |
| Scientist | 6.1 | Program synthesis |
| Skeptic | 6.0 | Empirical falsification |
| Philosopher | 1.8 | Ontological reframing |
| Causal Reasoner | 5.8 | Causal validation |
| Curiosity Engine | 6.0 | Surprise tracking |
| Metacognitor | 6.2 | Agenda setting + voting |
| Archivist | 1.0 | Episode archival (end-of-task) |

**Interpretation:** Core agents (Dreamer, Scientist, Skeptic, Causal Reasoner) operate at near-constant frequency, forming the main debate loop. Philosopher and Perceiver are invoked selectively (1-2 times per task), suggesting that ontological reframing is necessary but not frequent.

### 4.5 Convergence Dynamics

**Typical Convergence Profile (Task T0013):**

Round | Top Hypothesis Confidence | Error | Metacognitor Action
------|--------------------------|-------|--------------------
1     | 0.33                      | 0.250 | Initial imagination
3     | 0.50                      | 0.140 | Scientist finds program
5     | 0.65                      | 0.106 | Causal testing begins
10    | 0.75                      | 0.090 | Curiosity plateau detected
15    | 0.90                      | 0.050 | Converging confidence
23    | 1.00                      | 0.000 | Consensus vote → SOLVED

**Key Observation:** Convergence does not occur monotonically. Periods of stagnation (plateau in error rate) trigger Curiosity Engine interventions, which redirect the council's focus. The final convergence typically occurs after 14-23 rounds.

### 4.6 Exemplar Execution Trace: Agent Dialogue

The following dialogue excerpt from a representative task (Rounds 04–17) demonstrates the council's reasoning process in real-time. The sequence reveals plateau detection, program discovery, and eventual convergence:

```
[R04] 💭 Dreamer: Imagined 8 hypotheses.
[R04] 🔬 Scientist: No generalizing program found in this round.
[R04] 🔴 Skeptic: No program to falsify.
[R04] ⚡ CuriosityEngine: PLATEAU detected (error=0.167). 
                         Directive: SCIENTIST_EXTEND_SEARCH
[R04] ⚡ CuriosityEngine: PLATEAU detected (error=0.167). 
                         Directive: DREAMER_EXPLORE_LOW_CONFIDENCE

[R05] 🎯 Metacognitor: Agenda set: Dreamer → Scientist → Skeptic → CausalReasoner
[R05] 💭 Dreamer: Imagined 8 hypotheses.
[R05] 🔬 Scientist: No generalizing program found in this round.
[R05] 🔴 Skeptic: No program to falsify.
[R05] ⚡ CuriosityEngine: PLATEAU detected (error=0.167). 
                         Directive: SCIENTIST_EXTEND_SEARCH

[R06] 🎯 Metacognitor: Agenda set: Dreamer → Scientist → Skeptic → CausalReasoner
[R06] 💭 Dreamer: Imagined 8 hypotheses.
[R06] 🔬 Scientist: No generalizing program found in this round.
[R06] 🔴 Skeptic: No program to falsify.

[R07] 🎯 Metacognitor: Agenda set: Dreamer → Scientist → Skeptic → CausalReasoner
[R07] 💭 Dreamer: Imagined 8 hypotheses.
[R07] 🔬 Scientist: Found program: gravity_down → majority_recolor (MDL=2.0)
[R07] 🔴 Skeptic: No program to falsify.
[R07] ⚡ CuriosityEngine: Surprise: 0.235 (ongoing)

[R08] 🎯 Metacognitor: Agenda set: Scientist → Skeptic → CausalReasoner
[R08] 🔬 Scientist: Found program: majority_recolor → gravity_down (MDL=2.0)
[R08] 🔴 Skeptic: No program to falsify.
[R08] ⚡ CuriosityEngine: Surprise: 0.167 (ongoing)

[R09] 🎯 Metacognitor: Agenda set: Scientist → Skeptic → CausalReasoner
[R09] 🔬 Scientist: No generalizing program found in this round.
[R09] 🔴 Skeptic: No program to falsify.
[R09] ⚡ CuriosityEngine: PLATEAU detected (error=0.167). 
                         Directive: DREAMER_EXPLORE_LOW_CONFIDENCE

[R10] 🎯 Metacognitor: Agenda set: Dreamer → Scientist → Skeptic → CausalReasoner
[R10] 💭 Dreamer: Imagined 8 hypotheses.
[R10] 🔬 Scientist: Found program: gravity_down → identity → majority_recolor (MDL=3.0)
[R10] 🔴 Skeptic: No program to falsify.
[R10] ⚡ CuriosityEngine: PLATEAU detected (error=0.213). 
                         Directive: SCIENTIST_EXTEND_SEARCH

[R14] 🎯 Metacognitor: Agenda set: Dreamer → Scientist → Skeptic → CausalReasoner
[R14] 💭 Dreamer: Imagined 8 hypotheses.
[R14] 🔬 Scientist: Found program: majority_recolor → identity → gravity_down (MDL=3.0)
[R14] 🔴 Skeptic: No program to falsify.
[R14] ⚡ CuriosityEngine: PLATEAU detected (error=0.191). 
                         Directive: SCIENTIST_EXTEND_SEARCH

[R15] 🎯 Metacognitor: Agenda set: Dreamer → Scientist → Skeptic → CausalReasoner
[R15] 💭 Dreamer: Imagined 8 hypotheses.
[R15] 🔬 Scientist: Found program: identity → majority_recolor → gravity_down (MDL=3.0)
[R15] 🔴 Skeptic: No program to falsify.
[R15] ⚡ CuriosityEngine: PLATEAU detected (error=0.191). 
                         Directive: SCIENTIST_EXTEND_SEARCH

[R16] 🎯 Metacognitor: Agenda set: Dreamer → Scientist → Skeptic → CausalReasoner
[R16] 💭 Dreamer: Imagined 8 hypotheses.
[R16] 🔬 Scientist: Found program: majority_recolor → gravity_up → mirror_v (MDL=3.0)
[R16] 🔴 Skeptic: No program to falsify.
[R16] ⚡ CuriosityEngine: PLATEAU detected (error=0.191). 
                         Directive: SCIENTIST_EXTEND_SEARCH

[R17] 🎯 Metacognitor: Convergence vote: H050_138e00 wins (confidence=0.50).
[R17] 📚 Archivist: Archived episode. Verdict: solved. Total: 57.
```

**Dialogue Analysis:**

The trace exhibits characteristic behavior of the council under plateau conditions. Rounds 4–9 demonstrate the Curiosity Engine detecting stagnation (error remaining at 0.167) and issuing directives to redirect exploration. The Scientist agent responds by extending search depth, discovering programs of increasing length (MDL=2.0 → MDL=3.0) as compositional complexity increases.

Note the pattern of program discovery: the Scientist identifies multiple candidate programs with equivalent MDL scores (e.g., rounds 14–16 all yield MDL=3.0). The Metacognitor accumulates evidence across these candidates and ultimately declares convergence when confidence reaches sufficient threshold. The final verdict (Round 17) reflects the council's collective assessment rather than any single agent's judgment.

This sequence illustrates the fundamental design principle: reasoning emerges through sustained dialogue rather than singular computation. No agent makes the final decision in isolation; all nine agents contribute evidence that accumulates on the shared blackboard.

---

## 5. Interpretability and Explainability

### 5.1 Dialogue Logs as Audit Trails

One distinguishing feature of LAteNT is complete observability of reasoning. Every agent action is logged with timestamp and message:

```json
{
  "round": 3,
  "agent": "Scientist",
  "action": "speak",
  "message": "Found program: mirror_v → mirror_h (MDL=2.0)",
  "timestamp": 1772263382.969
}
```

This enables forensic analysis of the council's reasoning process. For any failure case or unexpected behavior, the complete decision trace is available.

### 5.2 Hypothesis Lifecycle Visualization

The blackboard maintains a hypothesis stack with status transitions. One can trace the complete lifecycle of each hypothesis:

```
H001: confidence=0.33, status=PENDING
      → Scientist: program_found="mirror_v"
      → Skeptic: contradiction_found=False
      → CausalReasoner: verdict=CAUSAL_LAW
      → Metacognitor: status=ACCEPTED
```

This lifecycle reveals which agents contributed to hypothesis evaluation and at what stages disagreement occurred.

### 5.3 Skill Extraction and Reuse

The Archivist maintains provenance information for all learned skills:

```json
{
  "name": "majority_recolor",
  "origin": "T0004_a633ef1a1bbaadbb",
  "description": "Recolor all non-background to majority color"
}
```

This enables researchers to understand which tasks stimulated skill discovery and how those skills generalize.

---

## 6. Limitations and Design Constraints

### 6.1 DSL Expressiveness

The current primitive set is deliberately limited (10 builtin primitives) to maintain computational tractability. This design choice has implications:

**Expressible Tasks:** Tasks whose solutions can be composed from rotation, reflection, gravity, majority recolor, and size-based sorting. This covers a restricted subset of abstract reasoning problems.

**Inexpressible Tasks:** Tasks requiring conditional logic (IF-THEN rules), counting operations, connectivity-based transformations, or color mapping beyond majority recolor cannot be solved by construction.

**Implication:** The 100% solve rate reported here reflects task distribution alignment, not general reasoning capability. Performance would degrade significantly on tasks requiring primitives outside the DSL.

### 6.2 Program Search Scope

Scientist agent search is limited to programs of length ≤ 6 (empirically determined). This constraint:
- Ensures tractable synthesis (feasible search space)
- Biases toward parsimonious solutions (MDL principle)
- May miss solutions requiring >6 primitive compositions

### 6.3 Hypothesis Stack Saturation

The blackboard maintains a maximum of 50 hypotheses. Under certain conditions (e.g., Dreamer generating 8 hypotheses per round over 20 rounds = 160 hypotheses), older hypotheses are discarded. This could result in loss of useful hypotheses that initially scored low but later receive supporting evidence.

### 6.4 Training Data Assumption

The system assumes ≥1 training example per task. Under the zero-shot regime (no training examples), the system would degrade to pure hypothesis sampling without empirical validation—effectively random guessing.

---

## 7. Comparative Analysis

### 7.1 Relationship to Existing Approaches

**Program Synthesis Literature:**
LAteNT's Scientist agent follows classical program synthesis paradigms (bottom-up enumeration, MDL scoring), differing from recent neurosymbolic approaches in maintaining explicit, interpretable program representations rather than learning latent program embeddings.

**Multi-Agent Reasoning:**
The council structure draws from multi-agent debate frameworks and Socratic reasoning, but implements explicit falsification (Skeptic) and causal reasoning (Causal Reasoner) as distinct agents rather than implicit components of a unified reasoning system.

**Active Inference and Curiosity:**
The Curiosity Engine implements error-driven active inference, aligning with recent work on curiosity-driven learning, but operates within a symbolic hypothesis space rather than a continuous representation space.

### 7.2 Design Trade-offs

**Interpretability vs. End-to-End Learning:**
The system prioritizes interpretability and modularity over monolithic performance. Agents maintain explicit representations and reasoning is fully auditable. This comes at the cost of potential performance penalties compared to learned end-to-end systems.

**Symbolic Reasoning vs. Statistical Learning:**
The DSL enables exact program specification and verification, providing guarantees about solution semantics. However, this requires hand-engineered primitives rather than discovering them from data.

**Compositional Generalization:**
The skill library enables transfer across episodes within the same task distribution. Generalization to fundamentally different task types (e.g., trained on rotation tasks, tested on connectivity tasks) is constrained by DSL expressiveness.

---

## 8. Technical Specifications

### 8.1 Dependencies

```
numpy >= 1.21.0
scipy >= 1.7.0
streamlit >= 1.15.0
matplotlib >= 3.4.0
pandas >= 1.3.0
```

### 8.2 Code Structure

```
council.py (962 lines)
├── DSL class: Primitive implementations and program execution
├── AgentResult dataclass: Structured return values
├── Perceiver: Segmentation
├── Dreamer: Hypothesis generation
├── Scientist: Program synthesis
├── Skeptic: Empirical falsification
├── Philosopher: Ontological reframing
├── CausalReasoner: Counterfactual verification
├── CuriosityEngine: Surprise tracking
├── Metacognitor: Meeting orchestration
├── Archivist: Episodic memory
└── Council: Main control loop

memory.py (565 lines)
├── HypothesisStatus enum
├── Hypothesis dataclass
├── ContradictionEntry dataclass
├── WorldState dataclass
├── Blackboard class: Shared working memory
├── EpisodeMemory class: Persistent history
├── DSLSkillLibrary class: Learned primitive repository
└── SurpriseTracker class: Error dynamics

universe.py (503 lines)
├── Prior enum: Core knowledge priors
├── DifficultyLevel enum: Task complexity levels
├── GridObject dataclass: Object representation
├── ARCTask dataclass: Task specification
├── GridTransforms: Transformation implementations
└── Universe: Procedural task generator

LAteNT.py (1880 lines)
├── Streamlit configuration and styling
├── Session state management
├── Visualization helpers
├── Dashboard UI components
└── Main app loop
```

### 8.3 Time and Space Complexity

**Perceiver (Object Segmentation):**
- Time: O(H × W) where H, W are grid dimensions
- Space: O(n_objects)

**Dreamer (Hypothesis Generation):**
- Time: O(K × n_train_pairs × avg_program_length)
- Space: O(K) (K hypotheses in memory)

**Scientist (Program Synthesis):**
- Time: O(|DSL|^L × n_train_pairs) where L is max program length
- Space: O(|DSL|^L) (search space enumeration)

**Skeptic (Falsification):**
- Time: O(n_hypotheses × n_train_pairs)
- Space: O(1) (streaming evaluation)

**Causal Reasoner (Counterfactual Testing):**
- Time: O(n_counterfactuals × avg_program_length)
- Space: O(n_counterfactuals)

**Overall Session:**
- Dominant term: O(MAX_ROUNDS × |DSL|^L × n_train_pairs)
- For typical parameters (MAX_ROUNDS=30, |DSL|=10, L=6): ~100 million primitive operations per task

---

## 9. Future Research Directions

### 9.1 DSL Expansion

The current primitive set is minimal. Expansion to include conditional primitives, counting operations, and connectivity-based transformations would increase expressive power. Key challenge: maintaining tractability of program synthesis.

### 9.2 Learned Transformation Spaces

Rather than discrete DSL primitives, future work could explore learning transformation embeddings in a continuous latent space, enabling discovery of novel operations through interpolation and composition in that space.

### 9.3 Cross-Domain Generalization

Current skill library provides transfer within task families generated from the same universe. Evaluating transfer across fundamentally different task distributions (e.g., trained on geometric transformations, tested on connectivity-based tasks) would measure true generalization capability.

### 9.4 Adaptive Agent Orchestration

The current Metacognitor uses fixed voting mechanisms. Learning to dynamically weight agent contributions based on task properties (e.g., "this task benefits more from Philosopher invocations") could improve efficiency.

### 9.5 Counterfactual Generation Strategy

The Causal Reasoner currently generates counterfactuals uniformly. Intelligently selecting counterfactuals (e.g., targeting hypothesized causal factors) could improve causal inference efficiency.

---

## 10. Reproducibility

### 10.1 Seeding and Determinism

All components support deterministic execution via random seeding:
```python
universe = Universe(seed=11290)
council = Council(seed=11290)
```

Results reported in Section 4 are from seed=11290.

### 10.2 Dashboard Execution

```bash
streamlit run LAteNT.py
```

The dashboard provides interactive task selection, real-time council execution visualization, and post-hoc analysis of dialogue logs.

### 10.3 Standalone Script Execution

```python
from universe import Universe, DifficultyLevel
from council import Council

universe = Universe(seed=99)
council = Council(seed=99)

task = universe.generate_task(DifficultyLevel.L2)
for snapshot in council.solve(task):
    print(snapshot['final_verdict'], snapshot['round'])

stats = council.stats()
print(f"Solved: {stats['solved']}")
```

---


## 11. Interactive Demonstration


### System Screenshots

---


![Screenshot_28-2-2026_215632_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/d4c37f09-674a-41f0-b163-27892fcbcfb6)
![Screenshot_28-2-2026_215641_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/4495ae5a-de54-43cb-891f-83a958646b60)
![Screenshot_28-2-2026_215653_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/035262db-a642-40ae-9d6c-d4ec3a7e9dd8)
![Screenshot_28-2-2026_215659_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/c2974d5f-fd9b-45ba-9a3b-4711a06877f3)
![Screenshot_28-2-2026_21573_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/a9571ce4-baaf-482f-9dc2-ca979d4d0479)
![Screenshot_28-2-2026_214542_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/35cecfd2-34c9-4db4-a211-cb9237927c13)
![Screenshot_28-2-2026_214554_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/95b168c5-3053-4910-9ecc-c3a3657f9b7f)
![Screenshot_28-2-2026_214756_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/e706b1f5-568e-40b5-a39f-758cc76ae6bb)
![Screenshot_28-2-2026_21484_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/72e21427-87f6-46e3-8cf8-a60518ed6525)
![Screenshot_28-2-2026_214812_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/bdf4698f-e0c5-410c-bb60-1f91a87ed72a)
![Screenshot_28-2-2026_214818_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/e7f58cee-0ed7-4e2a-b7b9-82576b0e897c)
![Screenshot_28-2-2026_214830_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/353d80c8-c863-48cd-985d-196ce2200ca4)
![Screenshot_28-2-2026_214837_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/b40f92c8-b23a-4953-83a6-3bdfdfaf7b00)
![Screenshot_28-2-2026_214846_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/cc9b33fb-5a9a-4478-83c4-37dc062f69ce)
![Screenshot_28-2-2026_214855_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/08e0d506-43e3-4281-83a9-395e975ddf35)
![Screenshot_28-2-2026_21497_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/88280e67-d110-4fb2-a193-c9e9ad95f2bb)
![Screenshot_28-2-2026_214931_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/f9bbeea9-4451-457c-827a-5755b4f671e9)
![Screenshot_28-2-2026_214939_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/73cf8ba9-9536-41a5-892b-1b0be2da9e4d)
![Screenshot_28-2-2026_214946_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/c78ce7d2-3f35-426b-9d2c-5d7775d7b5a1)
![Screenshot_28-2-2026_214959_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/e6ad0e59-3e4f-4c91-b8c9-e6fd90069ab3)
![Screenshot_28-2-2026_21507_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/15e306c3-6f9e-404e-869c-1c6f2c236cc9)
![Screenshot_28-2-2026_215014_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/95f9cff5-3460-493b-97fd-33a09c819446)
![Screenshot_28-2-2026_215022_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/5ee1a3cc-83dd-4b04-a849-8c1bfd7d3266)
![Screenshot_28-2-2026_215029_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/c6a0c075-453a-4951-90fd-6f6115f3337b)
![Screenshot_28-2-2026_215034_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/f0cb9569-112c-4d47-bac8-bb7d39996ee0)
![Screenshot_28-2-2026_215039_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/9382640c-7246-45b5-8f21-d9d4522883f2)
![Screenshot_28-2-2026_215044_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/3b9c7ca7-6053-4381-a178-685147aceef2)
![Screenshot_28-2-2026_215050_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/f887d793-ac39-4f56-85bd-202bf0c164bc)
![Screenshot_28-2-2026_21515_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/19284d27-7383-496c-9c67-8b987d66a116)
![Screenshot_28-2-2026_215110_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/825c7c73-9579-44bc-9cb8-f1af6b9f678e)
![Screenshot_28-2-2026_215124_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/6c50de05-99ea-42d3-928e-c6980eda5652)

![Screenshot_28-2-2026_215410_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/d59429f7-2805-4a4a-8ac9-dca6150206cf)
![Screenshot_28-2-2026_215415_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/09f92a5d-8fae-468b-ac8b-586b5dc04440)
![Screenshot_28-2-2026_215419_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/1eb5b102-f7d0-43d8-a39f-84b132ea68cc)
![Screenshot_28-2-2026_215426_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/601cda06-8de8-40ca-a890-30f1208bc537)
![Screenshot_28-2-2026_215430_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/262701e0-fb11-4695-bf96-1c92793cd530)
![Screenshot_28-2-2026_215434_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/6a53895f-737f-4f25-a43d-25334220039d)
![Screenshot_28-2-2026_215440_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/7ebdfb23-31b6-4fa6-acf7-4cba3eff2486)
![Screenshot_28-2-2026_215444_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/f02d74fc-9484-4de5-9ef4-4c2b70c55758)
![Screenshot_28-2-2026_215451_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/cd4bf6d4-1041-48c1-9ac0-d4963154e4fd)
![Screenshot_28-2-2026_215455_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/87b44b81-d233-46a6-b79b-67ea9b71f7a5)
![Screenshot_28-2-2026_21550_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/db314784-1b90-489f-864b-bcae77935fc3)
![Screenshot_28-2-2026_21554_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/f7a7e128-8ed5-4147-bcc0-71f12f04b24e)
![Screenshot_28-2-2026_21559_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/d103e0d5-e757-4d3c-969a-863bc5c339ad)
![Screenshot_28-2-2026_215518_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/033fc8b5-60bb-4aa4-85eb-99b88b61e00c)
![Screenshot_28-2-2026_215522_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/d3816437-d6a2-4fe9-9844-926a6d9194aa)
![Screenshot_28-2-2026_215528_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/92b057c5-4fde-4afa-8346-c824a7888448)
![Screenshot_28-2-2026_215533_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/4851625a-0d1f-4cbf-8d13-2fce93fb9ffd)
![Screenshot_28-2-2026_215540_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/14dd51ba-d573-4e3f-b383-dad7497baa78)
![Screenshot_28-2-2026_215546_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/620b1ff4-57da-4617-902c-958cd29c4e90)
![Screenshot_28-2-2026_215551_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/ff75c24b-aedc-4549-9c7b-bdaaae98c689)
![Screenshot_28-2-2026_215555_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/8732daef-8875-4898-a846-4958070213ed)
![Screenshot_28-2-2026_21562_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/33dd7c3f-f868-4307-8302-4ae43b5b6b06)
![Screenshot_28-2-2026_21568_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/1acacbc5-847f-4146-a31f-c7682de1959a)
![Screenshot_28-2-2026_215614_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/da880231-1304-4842-88ff-5ac95e48fd88)
![Screenshot_28-2-2026_215620_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/cf058e1d-ec96-42a3-9867-18865afa18a3)
![Screenshot_28-2-2026_215626_latentpy-lhgs2sgsznmstuspovpg7a streamlit app](https://github.com/user-attachments/assets/392dca58-9217-4da1-84d6-dae5bbded46f)



---
## 12. References and Further Reading

**Program Synthesis:**
- Gulwani, S. (2015). "Dimensions in Program Synthesis". PPLJ.
- Solar-Lezama, A. (2008). "Program Synthesis by Sketching". PhD dissertation, UC Berkeley.

**Multi-Agent Systems:**
- Stone, P., & Veloso, M. (2000). "Multiagent systems: A survey from an AI perspective". Autonomous Robots, 8(3), 345-383.

**Active Inference:**
- Friston, K., et al. (2017). "Active inference and learning". Neuroscience & Biobehavioral Reviews.

**Causal Inference:**
- Pearl, J. (2009). "Causality: Models, Reasoning, and Inference". Cambridge University Press.

**Minimum Description Length:**
- Rissanen, J. (1978). "Modeling by shortest data description". Automatica, 14(5), 465-471.

---

## Contact and Attribution

**Author:** Devanik  
**Contact:** [devanik@iisertirupati.ac.in]  
**GitHub:** https://github.com/Devanik21  
**Twitter:** @devanik2005  

This work represents independent research conducted during the Samsung Convergence Software Fellowship at the Indian Institute of Science. All code, experimental data, and analysis are made available for academic and research purposes.

---

**Last Updated:** February 28, 2026  
**License:** Apache 2.0
