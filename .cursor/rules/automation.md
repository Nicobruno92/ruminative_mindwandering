---
trigger: model_decision
description: Automation - Local testing, SLURM batch jobs and automated result evaluation
---

---
description: "Automation - Local testing, SLURM batch jobs and automated result evaluation"
alwaysApply: true
---
## Automation & Iteration

### Rules
- **Fast Iteration**: Always provide/maintain scripts for fast local testing (e.g., small datasets, low number of permutations) to ensure code logic is sound before launching heavy cluster jobs.
- **SLURM Integration**: Document any pipeline that requires multi-step cluster execution (STAGES). Ensure `sbatch` scripts are clearly linked to their Python counterparts.
- **Automated Validation**: Pipelines must include a step where results (metrics, distribution plots) are generated and evaluated. The model must review these results before declaring a stage "verified".
- **Computational Reproducibility**: 
    - Any results generated must be accompanied by the exact software environment (versions) and random seeds used.
    - Log `pip freeze`, `conda list`, and the full `config.yaml` in the results output folder for every run.
- **Reproducibility**: Any batch run must log the exact `config.yaml` used into the results directory.

### Checklist
- [ ] Fast local test script exists and passes.
- [ ] Results folder contains the `used_config.yaml`.
- [ ] Performance metrics (AUC, Balanced Accuracy) have been reviewed and documented.
- [ ] SLURM scripts are updated to match Python entrypoint changes.