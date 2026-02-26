# 🏥 Multi-Med: High-Fidelity Interactive Medical Diagnosis via Multi-Agent Framework

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" />
  <img src="https://img.shields.io/badge/LLM-LLaMA%203-green.svg" />
  <img src="https://img.shields.io/badge/Benchmarks-Craft--MD%20%7C%20MedQA-orange.svg" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

> A multi-agent framework for structured, multi-turn clinical diagnosis that mitigates information decay and hallucination in conversational medical reasoning.

---

## 📄 Abstract

Despite strong performance on single-turn medical benchmarks, LLMs struggle in real-world multi-turn diagnostic interactions, where cumulative information decay and hallucination impair conversational reasoning. **Multi-Med** addresses this by decoupling agents to enable modular reasoning and reduce hallucination and error propagation during the diagnostic process.

On **Craft-MD**, Multi-Med achieves up to **82.85% accuracy** (LLaMA-3-70B), and on **MedQA** up to **65.0%**, consistently outperforming state-of-the-art interactive methods while approaching performance parity with non-interactive single-turn baselines.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Multi-Med Pipeline                       │
│                                                                 │
│  Step 1          Step 2                Step 3        Step 4    │
│  Case       Multi-Agent Clinical    Synthesis &     Final      │
│  Vignette ─► Dialogue Simulation ─► Faithfulness ─► Single-   │
│  Loading                             Evaluation     Turn Dx    │
│                                                                 │
│           ┌──────────────────────┐                             │
│           │  Patient Agent       │ ◄── Case Vignette           │
│           │       ▲  │           │                             │
│           │       │  ▼           │                             │
│           │  Interrogator Agent  │ ──► Summarizer ──► Diagnosis│
│           │       ▲  │           │         Agent        Agent  │
│           │       │  ▼           │                             │
│           │    Expert Agent      │                             │
│           └──────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

The framework consists of five specialized agents:

| Agent | Role |
|-------|------|
| **Patient Agent** | Simulates a clinical patient; responds strictly from the gold-standard case vignette without inference or fabrication |
| **Interrogator Agent** | Conducts the diagnostic interview, asking atomic one-at-a-time questions with explicit clinical justifications |
| **Expert Agent** | Supervises inquiry by assessing diagnostic sufficiency (4-tier confidence scale) and advising on the type of questions to ask next — never directly diagnoses |
| **Summarizer Agent** | Reconstructs the patient–interrogator dialogue into a structured case vignette, suppressing redundancy and long-context noise |
| **Diagnosis Agent** | Performs single-turn diagnostic inference on the reconstructed vignette, fully decoupled from conversational history |

---

## 📊 Results

### Diagnostic Accuracy

| Task | Model | Multi-Med | MedIQ | Non-Interactive |
|------|-------|-----------|-------|-----------------|
| **Craft-MD** | LLaMA-3-8B | 62.8% | 50.0% | 76.4% |
| | LLaMA-3-70B | **82.0%** | 72.1% | 82.1% |
| | Gemma-3-27B | 70.7% | 61.1% | 78.1% |
| | Baichuan-M2 | 76.6% | 70.0% | 80.4% |
| **MedQA** | LLaMA-3-8B | 55.7% | 45.8% | 68.1% |
| | LLaMA-3-70B | **65.0%** | 60.9% | 84.7% |
| | Gemma-3-27B | 57.8% | 51.0% | 72.0% |
| | Baichuan-M2 | 62.2% | 55.0% | 75.0% |

### Ablation Study (Craft-MD)

| Model | Full Multi-Med | W/O Options | W/O Expert Guidance | W/O Summarization |
|-------|---------------|-------------|---------------------|-------------------|
| LLaMA-3-70B | 82.25% | 74.0% | — | 72.0% (−10.25) |
| Gemma-3-27B | 70.7% | 59.0% | — | 64.0% (−6.7) |
| LLaMA-3-8B | 62.8% | 56.0% | — | 49.3% (−13.5) |

### Reconstruction Fidelity (Craft-MD)

| Model | Automatic (LLM-based) | Clinician-Validated | Cohen's κ |
|-------|-----------------------|---------------------|-----------|
| LLaMA-3-70B | 86% | 90.21% | 0.78 (Substantial) |
| Gemma-3-27B | 83% | 85.43% | 0.72 (Substantial) |
| LLaMA-3-8B | 79% | 82.20% | 0.64 (Moderate) |

---

## 🗂️ Repository Structure

```
multi-med/
├── main-framework.py           # Full Multi-Med pipeline (Craft-MD)
├── mediq-main-framework.py     # Full Multi-Med pipeline (MedQA)
├── without-summary.py          # Ablation: w/o Summarization agent
├── without-options.py          # Ablation: w/o diagnostic options in Interrogator
├── withou_guidanceandoptions.py # Ablation: w/o Expert guidance + options
└── README.md
```

---

## ⚙️ Setup

### Prerequisites

```bash
pip install openai pandas numpy tqdm
```

### API Configuration

Multi-Med uses the [NVIDIA NIM API](https://integrate.api.nvidia.com) to access LLaMA-3 models. Set your API key before running:

```python
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="YOUR_API_KEY_HERE"
)
```

### Data

- **Craft-MD**: Download from [Kaggle — icraft-md](https://www.kaggle.com/datasets/icraft-md) and place at `/kaggle/input/icraft-md/`
  - `all_craft_md.jsonl`
  - `patient facts.pickle`
- **MedQA**: Download and place at the configured path
  - `all_dev_good.jsonl`
  - `facts.pickle`

---

## 🚀 Usage

### Running the Full Framework (Craft-MD)

```bash
python main-framework.py
```

### Running on MedQA

```bash
python mediq-main-framework.py
```

### Running Ablations

```bash
# Without summarization agent
python without-summary.py

# Without diagnostic options in Interrogator
python without-options.py

# Without expert guidance and options
python withou_guidanceandoptions.py
```

### Configuring the Evaluation Window

In each script, adjust the `start` and `len_dataset` variables to control which dataset samples to evaluate:

```python
start = 105       # Starting index
len_dataset = 35  # Number of samples to evaluate
```

### Output

Results are saved to `file.pickle` and contain:

```python
patients_history    # Full dialogue + justified dialogue per patient
summary_history     # Summarizer agent outputs
diagnosis_history   # Diagnosis agent raw outputs
diagnosis_answers   # Extracted predicted answers
gold_answers        # Ground truth answers
gold_options        # Answer options per sample
gold                # Gold answer indices
generated           # Predicted answer indices
```

---

## 🔑 Key Design Decisions

**Why decouple agents?** Tightly coupled information acquisition and diagnosis leads to error accumulation and hallucination. Separating the Patient, Interrogator, Expert, Summarizer, and Diagnosis agents into modular roles enforces clean information flow and reduces conversational drift.

**Why a Summarizer?** LLMs diagnose significantly worse from raw dialogue vs. structured vignettes. The Summarizer eliminates redundancy and long-context noise, aligning the diagnostic input distribution with standard clinical evaluation formats.

**Why Expert guidance?** Without a supervising Expert, the Interrogator asks redundant or clinically irrelevant questions. The Expert's 4-tier confidence scale (`"Yes, very confident"` → `"No"`) dynamically regulates the inquiry loop and steers toward diagnostically decisive evidence.

---

## 📝 Citation

If you use Multi-Med in your research, please cite:

```bibtex
@inproceedings{multimed2025,
  title     = {Multi-Med: High-Fidelity Interactive Medical Diagnosis via Multi-Agent Framework},
  author    = {Anonymized Authors},
  year      = {2025}
}
```

---

## 🤝 Acknowledgements

Multi-Med builds on and compares against [MedIQ](https://arxiv.org/abs/2406.00922) (Kim et al., 2024) and evaluates on the [Craft-MD](https://arxiv.org/abs/2402.01374) (Johri et al., 2024) and [MedQA](https://arxiv.org/abs/2009.13081) (Jin et al., 2021) benchmarks. Reconstruction fidelity uses [BioBERT](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) embeddings.
