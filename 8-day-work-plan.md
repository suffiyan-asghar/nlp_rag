# Assignment 3: 8-Day Work Plan (2 Members)

This plan is tailored for a 2-member team working on an NLP assignment.

## 1. Prerequisites (Complete Before Day 1)

### Academic/Project Prerequisites

- Read the full assignment instructions and grading rubric.
- Confirm deliverables: code, report, slides/demo, and submission format.
- Confirm constraints: allowed libraries, dataset restrictions, model limits, and citation rules.
- Define success criteria (for example: baseline + improved model + analysis + error cases).

### Technical Prerequisites

- Python 3.10+ installed.
- Git + GitHub/GitLab repository created.
- Virtual environment set up.
- Core packages installed (adjust as needed):
  - numpy, pandas, scikit-learn
  - nltk or spacy
  - matplotlib or seaborn
  - jupyter, pytest
- Shared folder for report assets and figures.
- Team communication channel (WhatsApp/Discord/Teams).

### Team Prerequisites

- Daily stand-up time fixed (15 minutes).
- Branching rule decided (main + feature branches).
- Definition of done agreed:
  - Code runs end-to-end.
  - Results are reproducible.
  - Report sections include figures and interpretation.

## 2. Role Split for 2 Members

- Member 1 (Engineering Lead):
  - Repo setup, integration, code quality, experiment tracking, final submission packaging.
  - Shared responsibility: modeling and analysis.
- Member 2 (Data + Report Lead):
  - Data pipeline, EDA, documentation, report/slides ownership.
  - Shared responsibility: modeling and analysis.

## 3. Working Rules (Every Day)

- Start with a 15-minute sync: yesterday progress, today goals, blockers.
- End with a 10-minute wrap-up: completed tasks and next-day priorities.
- Both members push code daily with clear commit messages.
- Keep one shared experiment log file (for example experiments.md).
- No PR waits longer than 24 hours.

## 4. 8-Day Detailed Plan (Member-Wise)

## Day 1: Scope, Setup, and Task Split

Goal: Understand requirements and make project runnable.

- Member 1:
  - Create repo structure: src/, notebooks/, data/, reports/, figures/.
  - Add README with environment setup and run commands.
  - Create issue board with To Do, Doing, Review, Done columns.
- Member 2:
  - Download dataset(s), verify license/usage permissions.
  - Document dataset fields, labels, and size.
  - Create report template with all required section headings.

End-of-day output: repo is ready, data is available, plan is approved.

## Day 2: Data Understanding and Preprocessing

Goal: Build a reliable preprocessing pipeline and produce first insights.

- Member 1:
  - Define pipeline interfaces (input/output format for each stage).
  - Add seed handling and config file skeleton.
  - Review Member 2 preprocessing implementation and add quick checks.
- Member 2:
  - Implement cleaning, tokenization, normalization.
  - Implement train/validation/test split with fixed random seed.
  - Run EDA and produce 2-4 useful plots.

End-of-day output: reproducible preprocessing + EDA figures.

## Day 3: Baseline Modeling

Goal: Get first end-to-end baseline metrics.

- Member 1:
  - Implement baseline model training/evaluation script.
  - Ensure full pipeline runs from a single command.
  - Save metrics and confusion matrix output files.
- Member 2:
  - Validate split integrity and leakage checks.
  - Write data/preprocessing section draft in report.
  - Add baseline results table template and placeholders.

End-of-day output: baseline results logged and documented.

## Day 4: Improved Model and Early Tuning

Goal: Train stronger model candidate and compare to baseline.

- Member 1:
  - Implement improved model approach and start hyperparameter tuning.
  - Log every run (params, score, notes).
  - Prioritize fastest high-impact experiments.
- Member 2:
  - Add preprocessing variants (for example stemming/lemmatization/stopword choices).
  - Prepare clean comparative table (baseline vs improved).
  - Draft methodology explanation for model decisions.

End-of-day output: at least one improved candidate with preliminary gains.

## Day 5: Evaluation, Error Analysis, and Ablations

Goal: Produce strong analysis, not only numbers.

- Member 1:
  - Run ablation studies (change one component at a time).
  - Finalize shortlist of top model variants.
  - Generate final validation/test metrics for shortlisted runs.
- Member 2:
  - Perform error analysis and categorize failure patterns.
  - Build hard-example sheet and annotate representative cases.
  - Write analysis: why model fails and what improved performance.

End-of-day output: final model choice backed by evidence.

## Day 6: Freeze Experiments and Full Draft

Goal: Complete full first draft of code + report.

- Member 1:
  - Freeze experiment code and clean scripts.
  - Merge stable branches and tag release candidate v0.9.
  - Export model artifact and inference script.
- Member 2:
  - Complete full report draft with figures, captions, and references.
  - Create slides draft with clear story (problem, method, results, limitations).
  - Verify all reported metrics match saved outputs.

End-of-day output: complete draft package (code + report + slides v1).

## Day 7: QA and Polish

Goal: Remove defects and prepare submission-ready package.

- Member 1:
  - Re-run full pipeline from clean clone/environment.
  - Fix reproducibility issues and finalize README commands.
  - Review against rubric checklist item by item.
- Member 2:
  - Edit report language, formatting, and conclusion quality.
  - Finalize figures and tables for readability.
  - Rehearse presentation and split speaking parts.

End-of-day output: near-final deliverables with no major blockers.

## Day 8: Submission Day

Goal: Submit confidently with validation and backup.

- Member 1:
  - Final merge and tag v1.0-submission.
  - Build final submission folder/zip exactly per instructions.
  - Verify required files exist and run commands still work.
- Member 2:
  - Final proofread of report/slides.
  - Check file names, versions, timestamps, and portal requirements.
  - Store backup copy (local + cloud).

End-of-day output: submission complete and archived.

## 5. Deliverables Checklist

- Source code with clear structure.
- README with setup, run, and reproduce steps.
- Final report in required format.
- Figures/tables used in report.
- Slides/demo assets if required.
- Experiment log and final configuration record.

## 6. Risk Management

- Risk: Experiment runtime is too long.
  - Mitigation: Keep a lightweight baseline path and cap tuning rounds.
- Risk: Last-day merge issues.
  - Mitigation: Merge daily and avoid long-lived branches.
- Risk: Inconsistent results.
  - Mitigation: Fix random seeds, pin package versions, and log configs.
- Risk: Weak narrative in report.
  - Mitigation: Write findings from Day 3 onward, not only at the end.
