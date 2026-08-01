# Forearm Physiological Foundation Model Research Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-grounded Chinese research package that explains EEG and other physiological foundation models in depth and turns those lessons into a concrete forearm wearable physiological foundation model design.

**Architecture:** The package separates evidence collection from synthesis. A structured literature matrix and BibTeX file act as the evidence layer; eight focused Markdown documents consume that evidence and cross-link to one another. The recommended model is a muscle-centric, factorized multimodal encoder, with generic time-series models as baselines and discrete neuromuscular tokenization reserved for later work.

**Tech Stack:** Markdown, CSV, BibTeX, PowerShell validation, primary academic sources from publisher pages, proceedings, PubMed, arXiv and official repositories.

---

## File structure

- `docs/research/README.md`: package index, reading paths and evidence-status legend.
- `docs/research/00_研究地图与入门指南.md`: beginner map, terminology and recommended learning order.
- `docs/research/01_EEG基座模型现状与方法.md`: EEG model taxonomy and detailed model comparisons.
- `docs/research/02_其他生理与时间序列基座模型.md`: ECG, PPG, wearable multimodal and general time-series foundation models.
- `docs/research/03_EMG与前臂感知研究现状.md`: EMG and non-EMG forearm sensing landscape.
- `docs/research/04_公开数据集与可用资源.md`: datasets, licenses, modalities, labels and suitability.
- `docs/research/05_前臂生理基座模型方案.md`: scientific framing, architecture, objectives and training stages.
- `docs/research/06_评测基准与实验矩阵.md`: downstream tasks, split protocols, metrics, baselines and ablations.
- `docs/research/07_分阶段研究路线图.md`: minimum viable study, paper-level study and long-term extensions.
- `docs/research/literature_matrix.csv`: one row per verified work or dataset.
- `docs/research/references.bib`: reference-manager-ready citations.

Git commits in this plan must use the listed messages. The repository currently lacks author identity; do not invent one. Commit steps become executable after the user supplies a repository-local Git author name and email.

### Task 1: Establish the package contract and terminology ledger

**Files:**
- Create: `docs/research/README.md`

- [ ] **Step 1: Create the package index**

Write `docs/research/README.md` with these sections:

```markdown
# 前臂可穿戴生理基座模型研究包

## 研究范围
本研究包讨论 EEG 与其他生理信号基座模型，并据此设计面向前臂可穿戴神经肌肉信号的通用表示模型。运动解码、疲劳、力量、身份识别和信号质量检测均为下游任务，而不是模型定义本身。

## 阅读路径
1. 初学者：00 → 01 → 03 → 05 → 06 → 07
2. 模型研究者：01 → 02 → 05 → 06
3. 数据与实验负责人：03 → 04 → 06 → 07

## 证据状态
- 同行评审：已在期刊或正式会议发表。
- 预印本：仅有 arXiv、bioRxiv、medRxiv 或审稿中版本。
- 数据资源：数据集主页或数据论文。
- 推断：由多个来源综合得到的项目判断，不归因于单篇论文。

## 术语表
| 规范术语 | 首次定义 | 使用规则 |
|---|---|---|
| 生理基座模型 | physiological foundation model | 泛指生理信号上的大规模预训练与迁移模型 |
| 前臂生理基座模型 | forearm physiological foundation model | 本项目的研究对象，不提前创造模型缩写 |
| 表面肌电 | surface electromyography (sEMG) | 首次展开，后续使用 sEMG |
| 高密度表面肌电 | high-density surface electromyography (HD-sEMG) | 不与普通多通道 sEMG 混用 |
| 跨受试者 | cross-subject | 与跨会话、跨设备分别报告 |
| 冻结探测 | frozen probing | 编码器完全冻结，仅训练读取头 |
| 少样本适配 | few-shot adaptation | 必须报告每类样本数或标注比例 |

## 检索边界
检索截止日期：2026-07-31。优先引用原始论文、正式出版页面、官方数据集页面和官方代码仓库。
```

- [ ] **Step 2: Verify required sections and terms**

Run:

```powershell
rg -n "研究范围|阅读路径|证据状态|术语表|检索边界|sEMG|HD-sEMG" docs/research/README.md
```

Expected: every required section and both EMG terms appear at least once.

- [ ] **Step 3: Check Markdown whitespace**

Run:

```powershell
git diff --check -- docs/research/README.md
```

Expected: no output.

- [ ] **Step 4: Commit the package contract**

```powershell
git add -- docs/research/README.md
git commit -m "docs: define physiological foundation model research package"
```

### Task 2: Build the verified evidence layer

**Files:**
- Create: `docs/research/literature_matrix.csv`
- Create: `docs/research/references.bib`

- [ ] **Step 1: Create the literature matrix schema**

Use this exact CSV header:

```csv
record_id,category,title,year,publication_status,venue,signal_modalities,pretraining_data,model_scale,input_representation,architecture,pretraining_objectives,adaptation_modes,downstream_tasks,generalization_axes,open_code,open_weights,primary_url,doi_or_identifier,key_contribution,key_limitation,relevance_to_forearm,last_verified
```

Populate verified seed entries for the following groups:

- EEG: BENDR, BIOT, LaBraM, EEGPT, CBraMod and at least one 2026 EEG foundation-model benchmark.
- Physiological/time series: Apple wearable PPG/ECG foundation models, PaPaGei and MOMENT.
- EMG: Meta's generic non-invasive neuromotor interface, AEMG, emg2pose and emg2qwerty.
- Forearm datasets: NinaPro, Hyser/HD-sEMG, HD-FW KIN and one repeated-session dataset.

For every row, verify the publication status and primary URL. Leave a genuinely unavailable metadata field empty rather than guessing.

- [ ] **Step 2: Use reproducible search queries**

Run source searches using these query families and record all retained sources in the matrix:

```text
"EEG foundation model" AND (pretraining OR benchmark OR generalization)
(BENDR OR BIOT OR LaBraM OR EEGPT OR CBraMod) EEG
(ECG OR PPG OR wearable biosignal) AND "foundation model"
(surface EMG OR sEMG) AND (foundation model OR self-supervised pretraining)
(forearm OR wrist) AND (sEMG OR mechanomyography OR force myography OR ultrasound OR electrical impedance tomography)
(sEMG OR electromyography) AND (cross-subject OR cross-session OR electrode shift OR fatigue OR biometrics)
```

Use publisher/proceedings/PubMed/arXiv pages as evidence. Use review articles for discovery only when the original paper is available.

- [ ] **Step 3: Create verified BibTeX entries**

Create `docs/research/references.bib` with one entry for every paper cited in the Markdown package. Use DOI-based keys when practical, otherwise use stable `firstauthor_year_shorttitle` keys. Each entry must contain `title`, `author`, `year`, and either `doi`, `url`, or a stable proceedings identifier.

- [ ] **Step 4: Validate the matrix shape**

Run:

```powershell
$rows = Import-Csv docs/research/literature_matrix.csv
$required = @('record_id','category','title','year','publication_status','primary_url','key_contribution','key_limitation','last_verified')
$missing = $required | Where-Object { $_ -notin $rows[0].PSObject.Properties.Name }
if ($missing.Count -gt 0) { throw "Missing columns: $($missing -join ', ')" }
if ($rows.Count -lt 16) { throw "Expected at least 16 verified seed records; found $($rows.Count)" }
```

Expected: no output and exit code 0.

- [ ] **Step 5: Validate evidence completeness**

Run:

```powershell
$rows = Import-Csv docs/research/literature_matrix.csv
$bad = $rows | Where-Object { -not $_.record_id -or -not $_.title -or -not $_.publication_status -or -not $_.primary_url -or -not $_.last_verified }
if ($bad.Count -gt 0) { $bad | Format-Table record_id,title; throw "Incomplete required evidence fields" }
```

Expected: no incomplete rows.

- [ ] **Step 6: Validate BibTeX coverage**

Run:

```powershell
$entries = (rg -c '^@' docs/research/references.bib).Split(':')[-1]
if ([int]$entries -lt 12) { throw "Expected at least 12 BibTeX entries; found $entries" }
```

Expected: at least 12 entries.

- [ ] **Step 7: Commit the evidence layer**

```powershell
git add -- docs/research/literature_matrix.csv docs/research/references.bib
git commit -m "docs: add verified biosignal foundation model evidence"
```

### Task 3: Write the detailed EEG foundation-model review

**Files:**
- Create: `docs/research/01_EEG基座模型现状与方法.md`

- [ ] **Step 1: Draft the organizing argument**

Open with this bounded argument:

```markdown
EEG 基座模型的主要进展并非简单扩大参数规模，而是逐步解决跨数据集预训练中的格式异构：通道数量与位置不同、采样率和窗口长度不同、任务标签体系不同，以及信号噪声和缺失通道普遍存在。对前臂肌肉基座模型最有价值的经验，因此是输入统一、空间—时间建模、自监督目标与严格迁移评测，而不是照搬某个模型的规模。
```

- [ ] **Step 2: Add the method taxonomy**

Create separate sections for:

1. EEG 数据与格式异构。
2. 原始波形、频谱和离散 token 表示。
3. 卷积、Transformer、空间—时间分解和多尺度架构。
4. 对比预测、掩码重建、离散 token 预测和自回归目标。
5. 冻结探测、少样本适配、参数高效微调和完整微调。
6. 跨数据集、跨受试者、跨通道布局和临床迁移评测。

- [ ] **Step 3: Add detailed model profiles**

For BENDR, BIOT, LaBraM, EEGPT and CBraMod, report:

- publication status and year;
- input representation and channel handling;
- backbone architecture;
- pretraining corpus and objective;
- adaptation protocol;
- downstream task coverage;
- open code/weights status;
- strongest transferable idea;
- principal limitation.

Include a comparison table whose rows are the five models and whose columns cover all nine fields above.

- [ ] **Step 4: Add benchmark corrections to model claims**

Use primary 2026 benchmark sources to explain that frozen linear probes can understate or expose weak transfer, specialist models remain competitive, channel adaptation is architecture-dependent, and larger parameter counts do not guarantee better cross-subject generalization. Mark these as evidence-based cautions rather than universal laws.

- [ ] **Step 5: End each topic with a forearm transfer lesson**

Add a final section named `## 对前臂肌肉基座模型的可迁移经验` covering:

- flexible channel tokenization;
- explicit device and montage metadata;
- spatial and temporal factorization;
- mixed reconstruction, prediction and alignment objectives;
- frozen, few-shot and full-finetuning evaluation;
- comparison with compact specialist baselines.

- [ ] **Step 6: Verify coverage**

Run:

```powershell
rg -n "BENDR|BIOT|LaBraM|EEGPT|CBraMod|掩码|自回归|冻结探测|少样本|跨受试者|前臂" docs/research/01_EEG基座模型现状与方法.md
```

Expected: every term appears in substantive prose or a comparison table.

- [ ] **Step 7: Commit the EEG review**

```powershell
git add -- docs/research/01_EEG基座模型现状与方法.md
git commit -m "docs: review EEG foundation model methods"
```

### Task 4: Review other physiological and time-series foundation models

**Files:**
- Create: `docs/research/02_其他生理与时间序列基座模型.md`

- [ ] **Step 1: Structure the review by signal and modeling problem**

Use these sections:

- ECG foundation and self-supervised models.
- PPG foundation models.
- Wearable multimodal models.
- General-purpose time-series foundation models.
- Cross-modal transfer and missing-modality learning.
- Lessons that do and do not transfer to forearm signals.

- [ ] **Step 2: Compare representative systems**

Include verified profiles for the Apple wearable PPG/ECG work, PaPaGei and MOMENT. Add peer-reviewed multimodal physiological models only when their primary papers can be verified. Compare participant-level contrastive learning, morphology-aware objectives, masked autoencoding, shared latent spaces and modality-specific encoders.

- [ ] **Step 3: Separate transferable and non-transferable assumptions**

Explain explicitly:

- PPG/ECG have relatively stable physiological morphology, whereas sEMG is strongly affected by electrode location and muscle recruitment.
- Participant-level positive pairs may preserve identity-related physiology but can conflict with identity-invariant downstream transfer.
- General time-series models provide strong architecture baselines but may ignore sensor physics and anatomy.
- Multimodal alignment is most useful when synchronized signals observe different stages of the same physiological process.

- [ ] **Step 4: Verify coverage and evidence links**

Run:

```powershell
rg -n "ECG|PPG|Apple|PaPaGei|MOMENT|多模态|缺失模态|前臂" docs/research/02_其他生理与时间序列基座模型.md
```

Expected: all terms appear, and each named model has a primary-source link.

- [ ] **Step 5: Commit the cross-modal review**

```powershell
git add -- docs/research/02_其他生理与时间序列基座模型.md
git commit -m "docs: review physiological and time-series foundation models"
```

### Task 5: Map EMG and non-EMG forearm sensing

**Files:**
- Create: `docs/research/03_EMG与前臂感知研究现状.md`

- [ ] **Step 1: Define the physiological chain**

Organize modalities by what they observe:

```text
neural drive → muscle electrical activation → mechanical deformation/force → limb kinematics
                 sEMG/HD-sEMG          MMG/FMG/ultrasound/EIT       IMU/mocap
```

Clarify that EIT primarily observes impedance distribution, ultrasound observes structural deformation, and IMU/kinematics are movement outcomes rather than direct muscle signals.

- [ ] **Step 2: Review sEMG and HD-sEMG**

Cover motor-unit action potentials, amplitude and spectral features, electrode placement, cross-talk, skin impedance, normalization, high-density spatial maps, motor-unit decomposition and wearable constraints.

- [ ] **Step 3: Review alternative forearm modalities**

For MMG, FMG/pFMG, ultrasound/sonomyography/echomyography, EIT and IMU, report measurement principle, temporal and spatial resolution, wearable feasibility, susceptibility to placement and motion, depth sensitivity, power/compute implications and likely role in pretraining.

- [ ] **Step 4: Review EMG foundation-model precursors and current work**

Cover self-supervised representation learning, domain adaptation, rapid recalibration, Meta's population-scale generic neuromotor interface and the 2026 AEMG preprint. Distinguish task-specific large supervised decoders from foundation models using explicit criteria: broad pretraining data, reusable encoder, multiple adaptation settings and diverse downstream tasks.

- [ ] **Step 5: Analyze the central domain shifts**

Create one subsection each for:

- subject anatomy and recruitment;
- session and re-donning drift;
- electrode rotation, translation and channel failure;
- device and sampling-rate differences;
- posture, load and contraction-intensity confounding;
- fatigue and temporal non-stationarity;
- identity information and privacy.

- [ ] **Step 6: Verify modality and risk coverage**

Run:

```powershell
rg -n "sEMG|HD-sEMG|MMG|FMG|超声|EIT|IMU|疲劳|身份|电极|跨会话|AEMG" docs/research/03_EMG与前臂感知研究现状.md
```

Expected: every modality and domain-shift term appears.

- [ ] **Step 7: Commit the forearm sensing review**

```powershell
git add -- docs/research/03_EMG与前臂感知研究现状.md
git commit -m "docs: map EMG and forearm sensing research"
```

### Task 6: Build the public dataset and resource guide

**Files:**
- Create: `docs/research/04_公开数据集与可用资源.md`

- [ ] **Step 1: Define the dataset table schema**

Use these columns:

```markdown
| 数据集 | 受试者/会话 | 模态与通道 | 采样率 | 同步标签 | 主要任务 | 泛化价值 | 许可/访问 | 主要限制 |
```

- [ ] **Step 2: Add verified datasets**

Include NinaPro, Hyser/HD-sEMG, emg2pose, emg2qwerty, HD-FW KIN, a demographic-diversity dataset and a repeated-session dataset. Add other datasets only when participant counts, modalities and access terms are verified from a data paper or official page.

- [ ] **Step 3: Score suitability without fabricating precision**

For each dataset, assign categorical suitability—high, medium or low—for:

- unlabeled pretraining;
- motion decoding;
- force/kinematics;
- fatigue;
- identity;
- cross-session transfer;
- cross-device transfer.

Explain the reason for each non-obvious rating. Do not create numerical composite scores without a validated weighting scheme.

- [ ] **Step 4: Propose the first public-data corpus**

Define three tiers:

1. Broad sEMG representation pretraining.
2. Synchronized sEMG–kinematics/force physiological anchoring.
3. Repeated-session and demographic stress testing.

For each tier, list candidate datasets, required preprocessing metadata and license compatibility concerns.

- [ ] **Step 5: Verify dataset coverage**

Run:

```powershell
rg -n "NinaPro|Hyser|emg2pose|emg2qwerty|HD-FW KIN|疲劳|身份|跨会话|许可" docs/research/04_公开数据集与可用资源.md
```

Expected: all named datasets and use cases appear.

- [ ] **Step 6: Commit the dataset guide**

```powershell
git add -- docs/research/04_公开数据集与可用资源.md
git commit -m "docs: catalogue public forearm biosignal datasets"
```

### Task 7: Specify the forearm physiological foundation model

**Files:**
- Create: `docs/research/05_前臂生理基座模型方案.md`

- [ ] **Step 1: State the task and non-claim**

Use this framing:

```markdown
目标是学习可在多个前臂生理任务间迁移的表示，而不是建立一个将所有肌肉与所有传感模态统一起来的全身模型。第一阶段以 sEMG 为核心部署输入，IMU、运动学和力量作为优先生理锚点；MMG/FMG、超声和 EIT 按数据可得性逐步接入。
```

- [ ] **Step 2: Specify the input contract**

Define signal tensors, channel masks, timestamps and metadata conceptually. Require explicit sampling rate, channel identity or relative position, device, session and participant identifiers. Explain which identifiers may enter the encoder, which are used only for splitting/auditing, and how deployment behaves when metadata is absent.

- [ ] **Step 3: Specify model modules**

Document each module with motivation, mechanism and ablation role:

- modality-specific signal adapter;
- channel and spatial metadata encoder;
- multi-timescale shared encoder;
- dynamic physiological state branch;
- identity/anatomy branch;
- device/session nuisance branch;
- task-specific heads;
- optional cross-modal teacher encoders.

- [ ] **Step 4: Specify the training stages**

Use four stages:

1. Single-modality masked and predictive sEMG pretraining.
2. Channel dropout, sampling-rate variation and device-conditioned robustness.
3. Cross-modal alignment with kinematics, force, IMU and available muscle modalities.
4. Factorization using dynamic, identity and device/session objectives with leakage probes.

- [ ] **Step 5: Specify objective roles and failure checks**

For temporal masking, channel masking, latent prediction, cross-modal alignment, cross-session consistency and factorization, state:

- what representation it should improve;
- which shortcut it may introduce;
- which ablation tests its necessity;
- which probe detects failure.

- [ ] **Step 6: Define minimum viable and extended configurations**

Minimum viable configuration:

- sEMG-only adapters;
- compact multi-scale encoder;
- temporal and channel masking;
- motion, fatigue and identity probes;
- cross-subject and cross-session splits.

Extended configuration:

- synchronized IMU/kinematics/force alignment;
- factorized latent branches;
- MMG/FMG or ultrasound teacher modality;
- parameter-efficient personalization;
- edge deployment measurements.

- [ ] **Step 7: Verify module coverage**

Run:

```powershell
rg -n "模态适配器|多时间尺度|动态生理|身份|设备/会话|掩码|跨模态|泄漏|最小可行|扩展配置" docs/research/05_前臂生理基座模型方案.md
```

Expected: every architectural and training component appears.

- [ ] **Step 8: Commit the model proposal**

```powershell
git add -- docs/research/05_前臂生理基座模型方案.md
git commit -m "docs: design forearm physiological foundation model"
```

### Task 8: Define the evaluation benchmark and experiment matrix

**Files:**
- Create: `docs/research/06_评测基准与实验矩阵.md`

- [ ] **Step 1: Define split protocols before metrics**

Specify:

- subject-disjoint split;
- session-disjoint split;
- device-disjoint split;
- dataset-disjoint transfer;
- joint subject-and-task holdout where data permit.

Require splitting before windowing and fitting preprocessing only on training data.

- [ ] **Step 2: Define downstream tasks and metrics**

Use:

- gesture/action: macro-F1, balanced accuracy and per-class recall;
- kinematics/force: MAE, RMSE, correlation and latency-aware error;
- fatigue: subject-disjoint AUROC or macro-F1 plus calibration, with protocol-dependent regression metrics when continuous ground truth exists;
- identity verification: ROC-AUC and equal error rate;
- identity identification: top-1 and macro-F1;
- signal quality/anomaly: AUROC, AUPRC and false-alarm rate.

- [ ] **Step 3: Define adaptation regimes**

Compare frozen probe, few-shot adaptation, parameter-efficient adaptation, full fine-tuning and from-scratch specialist models. Report trainable parameters, calibration data and compute for every regime.

- [ ] **Step 4: Define baseline families**

Include:

- handcrafted EMG features with linear or tree-based models;
- compact CNN or temporal convolution specialist;
- Transformer trained from scratch;
- general time-series pretrained model;
- EEG-inspired flexible-channel model;
- the proposed factorized model.

- [ ] **Step 5: Define ablations and stress tests**

Test removal of channel metadata, multiscale modeling, cross-modal alignment, identity branch, device/session branch and each objective. Stress tests must include channel loss, electrode rotation/shift, amplitude scaling, additive noise, resampling, long-duration drift and missing teacher modalities.

- [ ] **Step 6: Define leakage and negative controls**

Include shuffled-label controls, dataset-identity probes, window-adjacency checks, participant-template leakage checks, chronological confound tests for fatigue and action-intensity controls for fatigue classification.

- [ ] **Step 7: Build the claim–evidence matrix**

Add a table mapping each proposed claim to required evidence. At minimum:

```markdown
| 主张 | 必要证据 | 失败判据 |
|---|---|---|
| 表征可跨任务迁移 | 至少三个不同任务族的冻结或少样本提升 | 仅完整微调有效 |
| 表征可跨人泛化 | 受试者完全隔离的比较 | 随机窗口划分才有提升 |
| 模型具有设备鲁棒性 | 设备隔离或受控设备扰动 | 设备识别探针主导表示 |
| 表征包含疲劳信息 | 控制动作与强度后的疲劳预测 | 仅按时间顺序即可预测 |
| 身份与动态状态可分解 | 子空间交叉探针和消融 | 动态分支仍高精度识别身份 |
```

- [ ] **Step 8: Verify evaluation completeness**

Run:

```powershell
rg -n "subject-disjoint|session-disjoint|device-disjoint|macro-F1|RMSE|equal error rate|冻结|少样本|从零训练|通道缺失|电极|泄漏|失败判据" docs/research/06_评测基准与实验矩阵.md
```

Expected: all protocols, metric families and risk controls appear.

- [ ] **Step 9: Commit the benchmark design**

```powershell
git add -- docs/research/06_评测基准与实验矩阵.md
git commit -m "docs: define forearm foundation model benchmark"
```

### Task 9: Write the roadmap and beginner research map

**Files:**
- Create: `docs/research/07_分阶段研究路线图.md`
- Create: `docs/research/00_研究地图与入门指南.md`

- [ ] **Step 1: Define Phase 0—reproduction and audit**

Phase 0 must reproduce representative baselines on two public sEMG datasets, verify subject/session splits, and establish a common preprocessing and evaluation harness. Exit only when leakage checks and from-scratch baselines are stable.

- [ ] **Step 2: Define Phase 1—minimum viable foundation model**

Phase 1 uses sEMG-only pretraining with temporal and channel masking. It evaluates motion, fatigue and identity probes under cross-subject and cross-session splits. The phase succeeds only if pretraining improves at least two task families in frozen or low-label settings without relying on random-window splits.

- [ ] **Step 3: Define Phase 2—physiological anchoring and factorization**

Phase 2 adds synchronized kinematics, force or IMU alignment and separates dynamic, identity and device/session subspaces. Success requires improved transfer or robustness plus evidence from leakage probes and branch ablations.

- [ ] **Step 4: Define Phase 3—multimodal and deployment extension**

Phase 3 evaluates MMG/FMG, ultrasound or EIT as additional input or training-only teacher modalities, parameter-efficient personalization, missing-modality behavior and edge latency/memory. This phase is optional until Phase 2 passes its criteria.

- [ ] **Step 5: Add resource-scaled schedules**

Provide three schedules without assuming exact hardware:

- lightweight: public datasets, compact models, sEMG-only;
- standard: several public datasets, synchronized anchors, factorized model;
- ambitious: new longitudinal collection, multimodal teachers and deployment study.

For each schedule list deliverables, dependencies, main risk and stop/go criteria.

- [ ] **Step 6: Write the beginner map last**

In `00_研究地图与入门指南.md`, synthesize the completed package into:

- what a physiological foundation model is and is not;
- essential EEG lessons;
- forearm sensing modalities;
- recommended paper-reading sequence;
- recommended hands-on reproduction sequence;
- ten high-value research questions;
- a glossary cross-linked to `README.md`;
- links to documents 01–07.

- [ ] **Step 7: Verify roadmap and cross-links**

Run:

```powershell
rg -n "Phase 0|Phase 1|Phase 2|Phase 3|退出|成功|停止|轻量|标准|进取" docs/research/07_分阶段研究路线图.md
rg -n "01_EEG|02_其他生理|03_EMG|04_公开数据集|05_前臂生理|06_评测|07_分阶段" docs/research/00_研究地图与入门指南.md
```

Expected: all phases, decision criteria and document links appear.

- [ ] **Step 8: Commit the roadmap and research map**

```powershell
git add -- docs/research/00_研究地图与入门指南.md docs/research/07_分阶段研究路线图.md
git commit -m "docs: add forearm foundation model research roadmap"
```

### Task 10: Integrate, audit and finalize the package

**Files:**
- Modify: `docs/research/README.md`
- Modify: `docs/research/00_研究地图与入门指南.md`
- Modify: `docs/research/01_EEG基座模型现状与方法.md`
- Modify: `docs/research/02_其他生理与时间序列基座模型.md`
- Modify: `docs/research/03_EMG与前臂感知研究现状.md`
- Modify: `docs/research/04_公开数据集与可用资源.md`
- Modify: `docs/research/05_前臂生理基座模型方案.md`
- Modify: `docs/research/06_评测基准与实验矩阵.md`
- Modify: `docs/research/07_分阶段研究路线图.md`
- Modify: `docs/research/literature_matrix.csv`
- Modify: `docs/research/references.bib`

- [ ] **Step 1: Audit spec coverage**

Confirm that the package answers:

1. What methods current EEG and physiological foundation models use.
2. Which lessons transfer to forearm muscle sensing and which do not.
3. How the proposed factorized multimodal model is structured and trained.
4. How experiments distinguish physiological transfer from dataset shortcuts.

Add missing links or explanations to the responsible document rather than duplicating long passages.

- [ ] **Step 2: Audit terminology consistency**

Run:

```powershell
rg -n "surface EMG|sEMG|HD-sEMG|生理基座模型|前臂生理基座模型|冻结探测|少样本适配" docs/research/*.md
```

Expected: abbreviations are expanded at first use in each standalone document, and canonical terms remain consistent.

- [ ] **Step 3: Audit unsupported claims and publication status**

Search for absolute novelty and universality language. Replace it with bounded wording unless primary evidence directly supports the statement. Every preprint cited in prose must be labeled `预印本`, and every model table must contain publication status.

Run:

```powershell
rg -n "首次|唯一|全面|彻底|总是|从不|state-of-the-art|SOTA" docs/research/*.md
```

Expected: each match is either attributed to the original authors as a claim, bounded by scope, or removed.

- [ ] **Step 4: Audit local Markdown links**

Run:

```powershell
$files = Get-ChildItem docs/research -Filter *.md
$broken = @()
foreach ($file in $files) {
  $content = Get-Content -Raw $file.FullName
  foreach ($m in [regex]::Matches($content, '\]\((?!https?://|#)([^)]+\.md)(?:#[^)]+)?\)')) {
    $target = Join-Path $file.DirectoryName $m.Groups[1].Value
    if (-not (Test-Path -LiteralPath $target)) { $broken += "$($file.Name) -> $($m.Groups[1].Value)" }
  }
}
if ($broken.Count -gt 0) { $broken; throw "Broken local Markdown links" }
```

Expected: no broken links.

- [ ] **Step 5: Audit primary-source URLs**

Compare every named paper and dataset in documents 01–04 against `literature_matrix.csv`. Add any missing record and BibTeX entry. Ensure each retained matrix row has a primary URL and verification date of 2026-07-31.

- [ ] **Step 6: Run global file and whitespace checks**

Run:

```powershell
$expected = @(
  'README.md',
  '00_研究地图与入门指南.md',
  '01_EEG基座模型现状与方法.md',
  '02_其他生理与时间序列基座模型.md',
  '03_EMG与前臂感知研究现状.md',
  '04_公开数据集与可用资源.md',
  '05_前臂生理基座模型方案.md',
  '06_评测基准与实验矩阵.md',
  '07_分阶段研究路线图.md',
  'literature_matrix.csv',
  'references.bib'
)
$missing = $expected | Where-Object { -not (Test-Path -LiteralPath (Join-Path 'docs/research' $_)) }
if ($missing.Count -gt 0) { throw "Missing deliverables: $($missing -join ', ')" }
git diff --check -- docs/research
```

Expected: all deliverables exist and `git diff --check` reports no errors.

- [ ] **Step 7: Update the package index**

Add a one-sentence description and relative link for every deliverable to `docs/research/README.md`. State the final source count, number of peer-reviewed papers, number of preprints and number of dataset resources based on `literature_matrix.csv`.

- [ ] **Step 8: Commit the integrated package**

```powershell
git add -- docs/research
git commit -m "docs: finalize forearm physiological foundation model research package"
```

- [ ] **Step 9: Report completion evidence**

Record in the final handoff:

- exact file links;
- literature matrix row count;
- peer-reviewed/preprint/dataset counts;
- validation commands run and their outcomes;
- unresolved evidence gaps, if any;
- recommended first experiment from document 07.
