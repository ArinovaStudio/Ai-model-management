# 04a. Pipeline Stages & Engines

The AI Planning Agent executes a **sequential pipeline** containing 8 dedicated engine modules. Each stage depends on the outputs generated and persisted by the preceding stage.

---

## 1. Pipeline Stages Detail

| Step | Engine File | Input Context | Persistence Output | Core Logic / Mathematical Heuristics |
|---|---|---|---|---|
| **1** | `analysis_engine.py` | Prompt description, title, start date, budget, and deadline. | `ProjectInfo`, `FeatureRequest` lists. | Extracts core requirement criteria and builds an initial feature scope. |
| **2** | `intelligence_engine.py` | Selected features & tech stack. | `ProjectIntelligence` specs. | Suggests code modularity blueprints, DB schema plans, and architectural style (monolith vs microservices). |
| **3** | `wbs_engine.py` | Project architecture & features. | `GroupingTask` (milestones) & `Task` lists. | Breaks down requirement features into milestones and actionable tasks. |
| **4** | `estimation_engine.py` | Tasks & Milestones list. | Hourly estimate fields, sorting indices (`order`). | Estimates complexity scores, effort hours, and calculates chronological task sequences. |
| **5** | `resource_engine.py` | Estimates and target deadlines. | `ResourceAllocation` records. | Evaluates team resource types and calculates required developer headcounts. |
| **6** | `schedule_engine.py` | Tasks, resource allocations, and dates. | `scheduledStart`, `scheduledEnd`, Sprints, and Milestones. | Maps dependencies, assigns task gantt timelines, and fits tasks into sprints. |
| **7** | `validation_engine.py` | Completed schedule and deadlines. | `TimelineValidation` checks. | Checks for resource overload and flags tasks that violate the deadline. |
| **8** | `gap_engine.py` | Schedule validation failures. | `GapReport` markdown summary. | Generates a final plan assessment showing capacity issues or timeline gaps. |

---

## 2. Key Scheduling Mathematical Formulas

### Sizing Project Duration:
If the user does **not** specify a hard deadline in `ProjectInfo`, the scheduling engine calculates a default duration:
$$\text{Expected Duration (Days)} = \text{ceil}\left(\frac{\text{Total Task Effort Hours}}{\text{Available Developers} \times \text{Productive Hours per Day}}\right)$$
* `Productive Hours per Day` defaults to `6.0` (accounts for overhead/meetings).
* `Available Developers` defaults to a baseline headcount of `2`.

### Sprint Partitioning:
Sprints are partitioned using `sprint_duration_days` (defaulting to `14` days). Sprints are assigned dynamically:
$$\text{Sprint Number} = \text{floor}\left(\frac{\text{Task Scheduled Start Date} - \text{Project Start Date}}{14}\right) + 1$$
If a milestone’s task timelines slip beyond the soft deadline, the engine raises warnings which are flagged in the `validation_engine`.
