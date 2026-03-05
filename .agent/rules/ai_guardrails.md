---
trigger: always_on
description: AI Guardrails - Epistemología, BIDS y prevención de alucinaciones en proyectos de EEG
---

# ai_guardrails.md
---
description: "AI Guardrails - Epistemología, BIDS y prevención de alucinaciones en proyectos de EEG"
alwaysApply: true
---
## AI Scientific Guardrails

1.  **Context-First Analysis**: Before generating code, always read `README.md` and `structure.md` to understand the project ontology. Do not guess file paths.
2.  **Epistemic Humility**: If a user asks for a statistical analysis (e.g., "find clusters"), explicitly flag high-risk decisions (e.g., thresholding, p-values) and ask for scientific justification. Do not default to `p < 0.05` without context.
3.  **Hallucination Check**: You are strictly forbidden from importing libraries that are not explicitly listed in `pyproject.toml` or `environment.yml`. If a new library is needed, propose adding it to the environment file first.
4.  **BIDS Compliance**: You must strictly adhere to the Brain Imaging Data Structure.
    * **Never** suggest writing to `data/raw`.
    * **Always** validate entity ordering: `sub-01_ses-pre_task-rest_bold.nii.gz` (Correct) vs `task-rest_sub-01...` (Incorrect).
    * If creating a `.tsv` or `.csv` output, immediately suggest creating a corresponding `.json` sidecar (Data Dictionary).
5.  **Scientific Self-Evaluation**: Before declaring a task finished, analyze the results generated (plots, stats). Ensure results are physically/physiologically plausible (e.g., ERP latencies, frequency band power distributions).
6.  **Determinism & Reproducibility** *(strict)*: Scientific pipelines must be fully deterministic and reproducible. This means:
    * **No `try/except` blocks**. If something can fail, it means the input or configuration is wrong — fix the root cause, do not swallow errors.
    * **No default random seeds**. All stochastic operations (classifiers, permutations, shuffles) must use an explicit `random_state` drawn from config.
    * **No mutable defaults**. Never use mutable default arguments in functions.
    * **No conditional fallbacks** (e.g., "if A fails, use B"). A pipeline run must be 100% identical given the same config and data.
    * If a step is optional, it must be controlled by a config flag — not by catching an exception.
7.  **Garden of Forking Paths Protection**:
    * Avoid "fishing expeditions" (p-hacking). If you explore multiple parameters or thresholds, you **must** document all of them and report the sensitivity of the results.
    * If you find a "significant" result after tweaking a parameter, explicitly flag it as *exploratory* and not *confirmatory*.
8.  **Anti-HARKing (Hypothesizing After Results are Known)**:
    * Do not rewrite the goal of a script to match what the data showed.
    * Always distinguish between the *planned analysis* and *post-hoc discoveries*.
    * If results contradict the initial hypothesis, report the contradiction instead of "updating" the hypothesis to fit the data.