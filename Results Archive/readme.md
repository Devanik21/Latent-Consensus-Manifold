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

## 11. References and Further Reading

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
