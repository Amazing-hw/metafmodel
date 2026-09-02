# Physiological Time-Series Foundation Models PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a visually verified Chinese PPTX that deeply reviews 17 influential physiological/time-series foundation-model works and turns their lessons into a concrete forearm/wrist sEMG motor-interaction research programme.

**Architecture:** A source manifest keeps every factual claim traceable to an official proceedings, publisher, or arXiv page. A single JavaScript generator uses `@oai/artifact-tool` to turn structured paper records into a 16:9 deck with three slides and at least 500 Chinese explanation characters in notes per work. A rendering-and-lint stage checks every exported slide for readability and layout defects before delivery.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled Node.js runtime, PowerPoint/PPTX, Poppler-based slide rendering, Python slide checks.

---

### Task 1: Build the verified source manifest

**Files:**
- Create: `C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\sources.md`
- Create: `C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\source-notes.txt`

- [ ] **Step 1: Create one record for each of the 17 selected papers**

Use the fixed corpus in `C:\Users\User\Documents\F_model\docs\superpowers\specs\2026-09-02-physiological-timeseries-foundation-models-ppt-design.md`. For each record, include title, venue/year, modality, data scale, tokenizer/representation, backbone, objective, evaluation, takeaway, and an official source URL. Mark `Biosignal Fingerprinting / M2AE` as a preprint.

- [ ] **Step 2: Record all source URLs in the delivery provenance file**

Use this exact format for every source, so it can be inserted into PowerPoint notes:

```text
LaBraM | ICLR 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html
```

- [ ] **Step 3: Verify that the corpus has 17 unique works and all URLs are primary sources**

Run:

```powershell
(Select-String -Path 'C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\sources.md' -Pattern '^## ').Count
```

Expected: `17`.

### Task 2: Author structured review content and speaker-note copy

**Files:**
- Create: `C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\paper_records.js`

- [ ] **Step 1: Define the paper-record schema**

Use exactly this shape for every work:

```js
{
  id: 'labram',
  title: 'LaBraM',
  venue: 'ICLR 2024',
  category: 'EEG / neural foundation model',
  slides: {
    positioning: { title: '问题与定位', bullets: [], diagram: [] },
    method: { title: '技术路线拆解', bullets: [], diagram: [] },
    evidence: { title: '证据与迁移判断', bullets: [], diagram: [] }
  },
  notes: { positioning: '', method: '', evidence: '' },
  sources: ['https://example.org']
}
```

- [ ] **Step 2: Write exactly three slide payloads per paper**

For each record, make the payloads answer: (a) why the work exists and what representation gap it addresses; (b) input, patch/tokenizer/codebook if any, backbone, self-supervised loss, adaptation path; (c) empirical evidence, limitation, `可借鉴` and `不应照搬` implications for forearm/wrist sEMG movement interaction.

- [ ] **Step 3: Write speaker notes in Chinese and attach sources**

Concatenate the three notes for each paper. Their text excluding the source block must contain at least 500 Chinese characters. End each note with:

```text
[Sources]
- https://official-primary-source.example
[/Sources]
```

- [ ] **Step 4: Add end-section comparison records**

Add data, tokenization, objective, evaluation, design recommendation, and submission-roadmap records. The recommendation must prioritize continuous multi-scale temporal encoding, masked reconstruction, EMG-to-future-kinematics/IMU prediction, selective contrastive consistency, and a stage-two VQ-codebook ablation.

### Task 3: Generate the PowerPoint deck

**Files:**
- Create: `C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\build_deck.mjs`
- Create: `C:\Users\User\Documents\F_model\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx`

- [ ] **Step 1: Load the bundled artifact runtime and start the artifact operation**

Run the mandated operation marker once before the first creation command:

```powershell
& $skillDir\container_tools\mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format pptx
```

Set `RUNTIME_NODE`, `RUNTIME_NODE_MODULES`, and `RUNTIME_BIN_DIR` from `load_workspace_dependencies`; create a `node_modules` junction in `deliverables\physio_fm_ppt` pointing at `RUNTIME_NODE_MODULES`.

- [ ] **Step 2: Implement a reusable 16:9 slide system**

In `build_deck.mjs`, implement `addChrome(slide, section, page)`, `addPaperSlide(slide, record, kind)`, `addNotes(slide, text, urls)`, and `addPipeline(slide, nodes)`. Use a dark navy background, muted teal as the primary accent, and warm orange solely for the project-relevance callouts. Set slide titles at 36 pt or larger and body text at 16 pt or larger. Include page number and section label on every slide.

- [ ] **Step 3: Generate all required slides in this order**

Create 5 landscape slides, 51 paper slides (3 × 17), and 6–10 synthesis slides. Use clear section-divider slides between EEG/heterogeneous signals, wearable/general time series, wrist sEMG/motor interaction, and the project proposal. Preserve one logical claim per slide and use compact native PowerPoint process diagrams rather than copied paper figures.

- [ ] **Step 4: Export and save the deck**

Run:

```powershell
& $env:RUNTIME_NODE 'C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\build_deck.mjs'
```

Expected: a PPTX with 60–66 slides at `C:\Users\User\Documents\F_model\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx`.

### Task 4: Render, inspect, and correct the deck

**Files:**
- Create: `C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\rendered\`
- Create: `C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\slides_test_report.txt`

- [ ] **Step 1: Run structural slide checks**

Run:

```powershell
python 'C:\Users\User\.codex\plugins\cache\openai-primary-runtime\presentations\26.813.12317\skills\presentations\scripts\slides_test.py' 'C:\Users\User\Documents\F_model\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx'
```

Expected: no text-overflow or invalid-slide errors.

- [ ] **Step 2: Render all slides to PNGs and create a montage**

Run:

```powershell
python 'C:\Users\User\.codex\plugins\cache\openai-primary-runtime\presentations\26.813.12317\skills\presentations\scripts\render_slides.py' 'C:\Users\User\Documents\F_model\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx' --output_dir 'C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\rendered'
```

Inspect the montage and every slide image. Correct clipping, low contrast, unintended overlap, and empty-space imbalance in `build_deck.mjs`, then regenerate and repeat checks until clean.

- [ ] **Step 3: Verify the content-count criteria**

Check: 60–66 slides; 17 distinct paper sections; exactly three deep-dive slides per work; all records contain a source block; all note triplets meet the 500-Chinese-character threshold; no reused paper figures. Record the results in `slides_test_report.txt`.

### Task 5: Deliver the verified artifact

**Files:**
- Modify: `C:\Users\User\Documents\F_model\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx`
- Modify: `C:\Users\User\Documents\F_model\deliverables\physio_fm_ppt\slides_test_report.txt`

- [ ] **Step 1: Confirm final output exists and is non-empty**

Run:

```powershell
Get-Item 'C:\Users\User\Documents\F_model\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx' | Select-Object Name,Length,LastWriteTime
```

Expected: `Length` is greater than zero.

- [ ] **Step 2: Hand off the deck with scope and verification summary**

Report that the deck covers 17 works, distinguishes peer-reviewed papers from the marked preprint, includes source-linked speaker notes, and was rendered and lint-checked. Link only the final PPTX in the delivery response.

## Self-Review

1. **Spec coverage:** Tasks 1–2 implement the fixed 17-paper corpus, three-slide format, detailed methods, notes, and source provenance. Task 3 implements the 16:9 visual system and 60–66-slide deck. Task 4 covers visual and structural QA. Task 5 covers the deliverable and handoff. No design-spec requirement is unassigned.
2. **Placeholder scan:** The plan contains no `TODO`, `TBD`, “similar to”, or deferred implementation language. Paths, commands, expected outcomes, and content schema are explicit.
3. **Type consistency:** `paper_records.js` is the single source of paper fields; `build_deck.mjs` consumes `record`, `kind`, notes, and source URLs using the schema defined in Task 2. Output and QA paths match across Tasks 3–5.
