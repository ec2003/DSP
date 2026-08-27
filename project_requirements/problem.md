## FPT UNIVERSITY - HO CHI MINH CITY

Faculty of Information Technology

## DSP501

## Digital Signal and Image Processing

## FINAL ASSIGNMENT

A Semester-long DSP-AI Research Project

Group Project  ·  3-4 students

Project Handbook  ·  OBE  ·  Research-Based Learning (RBL)  ·  AI-Native Learning Deliverables: IEEE Report + Reproducible Package + Presentation

## The Big Picture - From Signals to Intelligence

The figure below shows where this project fits within the complete intelligent-system pipeline. DSP is the bridge  that  converts  raw physical signals into meaningful representations on which AI makes predictions and decisions.

Physical World ↓ Signals ↓ Sampling &amp; Quantization ↓ Digital Signal Processing ↓ Feature Extraction ↓ Artificial Intelligence ↓ Prediction / Decision

This  assignment spans the Sampling &amp; Quantization → Feature Extraction → AI stages: students design the DSP front-end and integrate it with an AI model to solve a real signal-processing problem.

## 1. Introduction

## Purpose

The Final  Assignment is a semester-long group project that accompanies the entire course. Students will investigate  a  real-world  signal  processing  problem  by  integrating  Digital  Signal  Processing  (DSP)  and Artificial  Intelligence  (AI).  The  project  follows  the  principles  of  Research-Based  Learning  (RBL),  where students progressively develop their solution through problem formulation, literature review, implementation, experimentation, and critical analysis.

Rather than completing the project only at the end of the course, students are expected to continuously refine  their  work  as  new  DSP  concepts  are  introduced  throughout  the  semester.  The  assignment  is problem-driven,  not  topic-driven:  for  example,  the  objective  is  not  'Speech  Emotion  Recognition'  in general,  but  a  specific,  well-defined  problem  such  as  'Robust  Speech  Emotion Recognition under Noisy Environments.'

## How to Succeed in This Project

The  following  practical  advice  reflects  what  distinguishes  strong  projects  from  weak  ones.  It  sets expectations only and introduces no additional assessment requirements.

- Choose a focused research problem - a narrow, well-defined problem is easier to investigate rigorously than a broad topic.
- Read the literature before implementation - understanding existing methods prevents reinventing weak baselines and clarifies the research gap.
- Design the DSP pipeline before selecting AI models - the signal representation often matters more than the choice of classifier.

- Conduct experiments continuously rather than at the end - iterate throughout the semester as new DSP concepts are introduced.
- Explain why the results occur - link findings back to DSP theory and the research questions, rather than only reporting performance numbers.
- Maintain reproducibility throughout - fix random seeds, record versions, and keep the run scripts working as the project evolves, not as an afterthought.

## 2. Learning Outcomes

Upon successful completion of the project, students will be able to:

- formulate a real-world signal processing problem;
- conduct a literature review and identify research gaps;
- design and justify DSP methodologies;
- implement DSP algorithms for signal preprocessing and feature extraction;
- integrate DSP with AI/ML models;
- design scientifically sound experiments;
- critically analyze experimental results;
- communicate research findings professionally.

## Alignment with Course Learning Outcomes (CLOs)

The project outcomes map to the course learning outcomes as follows (indicative alignment only):

| Project Outcome           | Related CLO   |
|---------------------------|---------------|
| Problem formulation       | CLO1          |
| DSP methodology           | CLO2          |
| Frequency-domain analysis | CLO3          |
| DSP-AI integration        | CLO4          |
| Experimental evaluation   | CLO5          |

## 3. Project Workflow

Every  project  shall  follow  the  research  workflow  below.  This  mirrors  the  process  used  when  writing  a scientific paper.

Research Problem ↓ Research Objectives ↓ Research Questions ↓ Literature Review ↓ Research Gap ↓

Research Hypothesis

↓

## 4. Project Scope

Each group ( 3-4 students ) shall complete the following five phases across the semester.

## Phase 1 - Problem Definition

- Select a real-world signal processing problem (specific and well-defined, with an associated research gap).
- Define 1-2 research objectives - what the project aims to achieve.
- Formulate 2-4 research questions that follow directly from the objectives.
- State one or more testable research hypotheses (if applicable), or a research proposition for exploratory projects.

Deliverables: Research Problem; Research Objectives; Research Questions; Research Hypothesis or Proposition.

## Phase 2 - Literature Review

Conduct a literature review of 8-12 recent publications (2021-2026). The review should:

- compare DSP techniques across papers;
- compare AI approaches (models, datasets, findings) rather than listing summaries;
- identify the research gap the project addresses;
- summarize the expected research contributions.

Composition:  at  least 3  journal  papers and 3  conference  papers ;  at  least 2  survey/review papers are recommended (reading  surveys  first  helps  orient  the  more  specialized  reading).  A Literature  Matrix  is required (see Appendix A).

Deliverables: Literature Matrix; Research Gap; Expected Contributions.

## Phase 3 - DSP Methodology

Design  a  complete  DSP  pipeline  including  signal  acquisition,  preprocessing,  digital  filtering,  frequency analysis, and feature extraction. All design decisions must be theoretically justified.

DSP Methodology ↓ Feature Extraction ↓ AI Model Development ↓ Experimental Design ↓ Experiments ↓ Analysis &amp; Discussion ↓ Conclusion ↓ Future Work Deliverables: DSP pipeline design; justified preprocessing and filtering choices; feature-extraction plan.

## Phase 4 - AI Methodology

Develop  AI/ML  models  for  classification,  recognition,  detection,  or  prediction.  Model  selection  and hyperparameters must be justified and reported (learning rate, batch size, number of epochs, optimizer, etc.).

Deliverables: Model architecture and rationale; hyperparameter configuration; trained model.

## Phase 5 - Experimental Evaluation

Experiments should answer the research questions, not merely report performance. Each project must include a baseline comparison, an ablation study, quantitative evaluation, error analysis, and discussion.

Deliverables: Baseline comparison; ablation study; quantitative results; error analysis; discussion.

## 5. Research Questions &amp; Hypotheses

Each project must formulate 2-4 research questions (RQs) that are specific, measurable, and grounded in the literature review, plus one or more testable hypotheses where applicable . For exploratory projects in which a formal hypothesis is not appropriate, a research proposition may be stated instead. Illustrative examples:

## Example research questions

- RQ1. How does digital filtering affect signal quality and downstream classification accuracy?
- RQ2. Which feature extraction technique (FFT, STFT, Wavelet, MFCC) performs best for the selected task?
- RQ3. Does AI significantly improve performance over classical DSP-only approaches?
- RQ4. What factors influence the robustness of the proposed model?

## Example hypotheses (or propositions)

- H1. MFCC combined with STFT features achieves higher recognition accuracy than MFCC alone.
- H2. Band-pass filtering significantly improves ECG classification performance.

## Reference - FINER framework

A useful guide for formulating good research questions (Feasible, Interesting, Novel, Ethical, Relevant): https://scientific-publishing.webshop.elsevier.com/research-process/finer-research-framework/

## 6. Mandatory DSP Pipeline (Two-System Comparison)

Every  project  must  implement  and compare two systems to  experimentally  demonstrate the impact of DSP preprocessing on AI performance.

## Pipeline A - Baseline (minimal DSP)

Raw Signal ↓ Minimal preprocessing ↓

AI Model

Signal ↓ DSP Preprocessing ↓ Feature Extraction ↓ AI Model

Standard pipeline reference: Signal Input → Preprocessing (DSP) → Feature Extraction → AI/ML Model → Output Classification. Each stage must be clearly explained, implemented, and justified with respect to the research questions.

## Pipeline B - Proposed (full DSP)

## 7. Minimum Technical Requirements

Signal acquisition and sampling are prerequisite stages , not DSP techniques. The overall workflow is:

Signal Acquisition ↓ Sampling &amp; Quantization ↓ DSP Processing ↓ Feature Extraction ↓

AI Model

Within the DSP Processing and Feature Extraction stages, each project must include at least three DSP techniques selected  from  the  table  below. Projects relying on only one DSP technique (e.g., MFCC only) will not satisfy the minimum requirements.

| Category                  | Techniques                                             |
|---------------------------|--------------------------------------------------------|
| Frequency analysis        | FFT, STFT, Wavelet Transform                           |
| Filtering                 | Digital filtering (FIR/IIR): bandpass, notch, low-pass |
| Spectral / power          | Power Spectral Density (PSD), band-power features      |
| Cepstral                  | MFCC                                                   |
| Statistical / information | Statistical features, Entropy features                 |

Visualization is required: plot raw  signals,  spectrograms,  and  extracted  features  (spectrogram  or frequency-domain visualization).

## 8. AI / Machine Learning Requirements

- Design and evaluate at least one AI/ML model (SVM, Random Forest, CNN, LSTM, GRU, etc.).
- Evaluate using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- Baseline comparison is required: proposed approach vs. at least one baseline (e.g., DSP-only vs. DSP + AI, or FFT vs. STFT vs. Wavelet features).
- Hyperparameter tuning must be reported (learning rate, batch size, epochs, optimizer, etc.).

## Recommended (not mandatory)

Model explainability (e.g., SHAP, LIME, Grad-CAM, or other appropriate techniques) is encouraged for interpreting model decisions and understanding which signal features drive predictions - especially valuable at master's level.

## 9. Experimental Design

Experiments must be explicitly designed to answer the research questions. Each group must specify:

- Independent variables: e.g., feature type, filter configuration, model architecture.
- Dependent variables: e.g., accuracy, F1-score, inference time.
- Evaluation metrics &amp; protocol: train/test split or cross-validation, random seeds, number of runs.
- Controlled conditions: ensure a fair comparison between baselines and the proposed method.

## Required &amp; encouraged components

- Ablation study (required): demonstrate the contribution of each pipeline component. If the method uses filtering + MFCC + CNN, also evaluate without filtering, without MFCC, and FFT-features-only.
- Statistical significance (encouraged): paired t-test, Wilcoxon signed-rank test, or 95% confidence intervals over repeated runs.
- Error analysis (required): analyze misclassified samples, failure cases, noisy signals, and difficult classes to explain where and why the model fails.

## 10. Implementation &amp; Reproducibility

- Programming environment: Python or MATLAB.
- Suggested Python libraries: NumPy, SciPy, Librosa, Scikit-learn, TensorFlow/Keras, Matplotlib.
- Use public datasets (DEAP, RAVDESS, EMO-DB, PhysioNet, etc.) or self-recorded data (with ethical consent).
- Code must be runnable, well-commented, and well-structured.

## Reproducibility (required)

Provide everything needed for another person to re-run the experiments: fixed random seeds, software versions, hardware configuration, a complete dependency list (requirements.txt), the trained model, the experiment configuration, and step-by-step reproduction instructions. Reproducibility is met only if the experiments can be re-run from the submitted package.

## 11. Suggested Project Domains

Examples include (custom topics are welcome with instructor approval):

- Speech / Audio: Speech Emotion Recognition, Speaker Identification, Environmental Sound Classification
- Biomedical: ECG Arrhythmia Detection, EEG Sleep Stage Classification
- Industrial / Structural: Machinery Fault Detection, Structural Vibration Analysis

## 12. Project Timeline

| Lecture   | Milestone                                         |
|-----------|---------------------------------------------------|
| L1        | Course & project introduction                     |
| L2        | Assignment announcement, team formation           |
| L3        | Topic registration                                |
| L4        | Research Questions + Literature Review checkpoint |
| L5        | Research proposal                                 |
| L6        | Methodology consultation                          |
| L7        | Progress review                                   |
| L8        | Experimental refinement                           |
| L9        | Final report, source code & slides submission     |
| L10       | Final presentation & demonstration                |

All milestones before Lecture 9 are formative feedback only - they guide progress and do not carry separate grades.

## 13. Deliverables

## Required

- Technical Report - 8-10 pages (excluding references and appendices), single-column IEEE format. Overleaf template: https://www.overleaf.com/read/zrmhhstdnmwd#c703cb
- Literature Matrix (Paper | DSP | AI | Dataset | Findings | Research Gap | Limitations) - may be placed in the report appendix.
- Reproducible Research Package - a self-contained package another person can download and run immediately: README.md, requirements.txt, run.py, dataset link, trained model, and experiment configuration.
- Presentation Slides - 10-12 minutes per group.

## Optional

- Demonstration Video (3-5 minutes).
- Research Poster.
- One-page Extended Abstract - recommended for developing the project into a conference submission.

## 14. Report Structure (Required)

The  report  shall  include  the  following  sections.  The  Methodology  section  must begin with a Research Framework diagram (Research Problem → Dataset → DSP → Feature Extraction → Model → Evaluation), then detail Dataset, DSP Preprocessing, Feature Extraction, and AI/ML Model Design.

Following  IEEE paper  conventions,  the  report  begins  with  an Abstract and Keywords before  the Introduction.

1. Abstract
2. Keywords
3. Introduction
4. Research Problem
5. Research Objectives
6. Research Questions
7. Literature Review
8. Research Gap
9. Expected Contributions
10. Methodology
11. Experimental Design
12. Experimental Results
13. Error Analysis (misclassified samples, failure cases, difficult classes)
14. Discussion
15. Threats to Validity
16. Conclusion
17. Future Work
18. Ethics Statement
19. AI Declaration (Appendix)

## Discussion requirements

The Discussion must go beyond accuracy figures. Analyze WHY the results occurred (link findings to DSP theory and the RQs); the STRENGTHS of the approach relative to baselines and the literature; the WEAKNESSES and limitations of method, data, and evaluation; and the THREATS TO VALIDITY (internal, external, construct) and how they were mitigated.

## 15. Evaluation Criteria

The rubric evaluates the research process as well as the final system.

| Criterion                                             | Weight   |
|-------------------------------------------------------|----------|
| Research Problem, Objectives & Literature Review      | 15%      |
| DSP Methodology                                       | 20%      |
| AI/ML Methodology                                     | 20%      |
| Experimental Design, Baseline Comparison & Evaluation | 20%      |
| Discussion, Research Findings & Threats to Validity   | 10%      |

| Criterion                               | Weight   |
|-----------------------------------------|----------|
| Report & Presentation                   | 10%      |
| Research Contribution & Reproducibility | 5%       |

Report &amp; Presentation evaluates communication quality (clarity, structure, and delivery); Research Contribution  &amp;  Reproducibility evaluates  originality,  engineering  contribution,  and the reproducibility of the submitted package.

## 16. Research Ethics

Because  DSP501  projects  involve  human  speech  and  biomedical  signals  (ECG,  EEG),  all  projects  must comply with basic research ethics. Students must:

- respect dataset licenses and cite datasets according to their terms of use;
- anonymize all human speech and biomedical data; remove personally identifiable information;
- obtain informed consent for any self-recorded data, and state the consent procedure in the report;
- acknowledge external code and datasets;
- ensure reproducibility of experiments.

An Ethics Statement must be included in the report.

## 17. AI Declaration (Required)

Generative AI tools (e.g., ChatGPT, Gemini, Claude, GitHub Copilot) may be used to support learning and research.  However,  students  remain  fully  responsible  for  the  originality,  correctness,  and integrity of all submitted work. Each group must include an AI Declaration in the appendix of the report, stating:

- AI tool(s) used;
- purpose of use;
- stages where AI was used;
- human verification and modifications;
- responsibility statement.

## Acceptable uses

- brainstorming research ideas;
- improving academic writing;
- summarizing literature;
- explaining DSP concepts;
- debugging code;
- improving figures and presentations.

## Unacceptable uses

- fabricated experimental results;
- fabricated references;
- fabricated datasets;
- submitting AI-generated reports without verification;
- copying AI outputs without understanding.

## 18. Individual AI Reflection (Required)

Each student shall submit an individual reflection (300-500 words) describing:

- How AI supported your project.
- Which AI tools were used.
- What limitations or errors you encountered.
- How AI-generated outputs were verified.
- What engineering or research decisions required your own judgment.
- How you would improve your AI usage in future projects.

## Guiding questions

The following questions are intended to guide the reflection without restricting the answers:

- Which AI tools did you use?
- At which project stages was AI most helpful?
- What errors or hallucinations did AI produce?
- How did you verify AI-generated outputs?
- What engineering decisions required your own judgment?

The reflection is intended to promote responsible, transparent, and reflective use of generative AI.

## 19. Academic Integrity

Students are expected to uphold the highest standards of academic integrity. The following are strictly prohibited:

- plagiarism;
- fabrication or falsification of data;
- manipulated experimental results;
- undisclosed AI-generated content;
- uncredited reuse of code or datasets.

## Consequences

Violations may result in a zero score for the assignment and further disciplinary action according to university regulations.

## 20. Registration

Topic registration form:

https://forms.gle/KNAdp35hSN25MAJVA

## Appendix A - Literature Matrix

Include a comparison table such as the following (add one row per paper).

| Paper               | DSP Technique          | AI Model   | Dataset   | Key Findings                  | Research Gap             | Limitations         |
|---------------------|------------------------|------------|-----------|-------------------------------|--------------------------|---------------------|
| Nguyen et al., 2024 | MFCC + bandpass filter | CNN-LSTM   | RAVDESS   | High accuracy on clean speech | Limited noise robustness | Weak on noisy input |
| …                   | …                      | …          | …         | …                             | …                        | …                   |

## Appendix B - AI Declaration Template

| Item                           | Description   |
|--------------------------------|---------------|
| AI Tool                        |               |
| Purpose                        |               |
| Stage Used                     |               |
| Human Verification             |               |
| Final Responsibility Statement |               |

Item

Description

AI Tool

Purpose

Stage Used

Human Verification

Final Responsibility Statement

## Appendix C - Project Checklist

- [ ] ☐ Team Formation

- [ ] ☐ Topic Registration

- [ ] ☐ Research Problem

- [ ] ☐ Research Objectives

- [ ] ☐ Research Questions

- [ ] ☐ Literature Review

- [ ] ☐ Research Gap

- [ ] ☐ Hypothesis

- [ ] ☐ DSP Pipeline

- [ ] ☐ Feature Extraction

- [ ] ☐ AI Model

- [ ] ☐ Experimental Design

- [ ] ☐ Baseline Comparison

- [ ] ☐ Ablation Study

- [ ] ☐ Error Analysis

- [ ] ☐ Discussion

- [ ] ☐ Ethics Statement

- [ ] ☐ AI Declaration

- [ ] ☐ AI Reflection

- [ ] ☐ Final Report

- [ ] ☐ Source Code

☐ README ☐ Presentation

## Appendix D - Frequently Asked Questions

## Q: Can we use ChatGPT / Gemini / Claude?

A: Yes.  Generative  AI  may  be  used  to  support  learning  and  research  (brainstorming, writing, summarizing literature,  explaining  concepts).  All  usage  must  be  disclosed  in  the  AI  Declaration  and  the  individual  AI Reflection.

## Q: Can we use GitHub Copilot?

A: Yes,  for code assistance and debugging. You remain fully responsible for the correctness and integrity of the submitted code, and must declare its use.

## Q: Can we use pretrained models?

A: Yes,  provided  you  cite  them,  respect  their  licenses,  and  clearly  state what was reused versus what you implemented or fine-tuned yourself.

## Q: Can we use Kaggle or other public datasets?

A: Yes.  Use  public  datasets  (DEAP,  RAVDESS,  EMO-DB,  PhysioNet,  Kaggle,  etc.),  cite  them  correctly,  and comply with their licensing terms.

## Q: Can we collect our own datasets?

A: Yes.  Self-recorded  data  requires  informed  consent,  anonymization  of  personal  information,  and  a description of the consent procedure in the Ethics Statement.

## Q: Can we reuse code from GitHub?

A: Yes, if properly credited and license-compliant. Reusing code without attribution is an academic-integrity violation.

## Q: How should AI usage be declared?

A: Complete the AI Declaration (Appendix B) in the report appendix and submit the individual AI Reflection. State the tools, purpose, stages used, and how outputs were verified.

## Q: What happens if experiments cannot be reproduced?

A: Reproducibility  is  graded.  If  the  submitted  package  cannot  re-run  the  experiments  (missing  seeds, versions, data, or instructions), the reproducibility criterion is not met and the score is reduced accordingly.