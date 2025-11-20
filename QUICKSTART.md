# 🚀 Quickstart Guide - ML Reproduction Agent

**Get started in 5 minutes** with this autonomous ML research reproduction system.

## Prerequisites

- **Python 3.9+**
- **pip** or **conda**
- **10 MB disk space** for testing (100+ GB for full datasets)
- **(Optional) API keys** for OpenAI, DeepSeek, or Google Gemini

---

## Quick Setup (No API Keys Required)

Perfect for testing and development without spending money on LLM APIs:

### 1. Clone & Install

```bash
# Clone repository
git clone <repo-url>
cd Autonomous-Earth-Observation-ML-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Test Data

```bash
# Generate synthetic papers and datasets (takes ~10 seconds)
python scripts/generate_synthetic_data.py \
    --output-dir ./test_data \
    --num-samples 50 \
    --num-papers 5 \
    --image-size 64 \
    --seed 42
```

**Output:**
```
✓ Generated 5 mock paper specifications
✓ Generated synthetic EO dataset (50 samples, ~2 MB)
```

### 3. Run Tests

```bash
# Run comprehensive test suite
pytest tests/test_comprehensive.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Expected:** All tests pass (30+ tests, < 10 seconds)

### 4. Run Evaluation (Mock Mode)

```bash
# Evaluate agent without API keys (uses mock LLMs)
python scripts/evaluate_agent.py \
    --data-dir ./test_data \
    --mode mock \
    --quick

# Full evaluation on all papers
python scripts/evaluate_agent.py \
    --data-dir ./test_data \
    --mode mock \
    --output evaluation_results.json
```

**Output:**
```
✓ Paper parsed
✓ Dataset found (EuroSAT)
✓ Code generated (16 files)
Success rate: 100%
```

---

## Full Setup (With Real LLMs)

For actual paper reproduction with real research PDFs:

### 1. Get API Keys

You need **at least ONE** of:
- **OpenAI API Key** ([get here](https://platform.openai.com/api-keys)) - for code generation
- **DeepSeek API Key** ([get here](https://platform.deepseek.com/api_keys)) - for paper parsing
- **Google API Key** ([get here](https://aistudio.google.com/app/apikey)) - for dataset resolution

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your keys
nano .env  # or vim, vscode, etc.
```

```bash
# .env file
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
GOOGLE_API_KEY=...
```

### 3. Run on Real Paper

```bash
# Option 1: Use arXiv paper
python main.py --paper arxiv:2103.14030 --task segmentation

# Option 2: Use local PDF
python main.py --paper /path/to/paper.pdf --task classification

# Option 3: With custom settings
python main.py \
    --paper paper.pdf \
    --task segmentation \
    --gpu-hours 4 \
    --sensors Sentinel-2 SAR \
    --output-dir ./my_reproduction
```

### 4. Check Outputs

```bash
# Artifacts are saved to:
ls artifacts/run_<paper_name>/

# Structure:
# ├── paper.json              # Parsed paper spec
# ├── code/                   # Generated PyTorch project
# │   ├── src/                # DataModule, Model, Tasks
# │   ├── scripts/train.py    # Training script
# │   ├── configs/default.yaml
# │   └── tests/              # Unit tests
# ├── data/                   # Dataset (if downloaded)
# ├── runs/                   # Training checkpoints
# └── report.md               # Reproducibility report
```

---

## Usage Examples

### Example 1: Basic Classification Task

```bash
python main.py \
    --paper arxiv:1802.02697 \
    --task classification \
    --gpu-hours 2
```

**What it does:**
1. Downloads & parses paper from arXiv
2. Identifies task: land cover classification
3. Finds dataset: EuroSAT (Sentinel-2)
4. Generates PyTorch Lightning project
5. Creates reproducibility report

### Example 2: Segmentation with Specific Sensors

```bash
python main.py \
    --paper paper.pdf \
    --task segmentation \
    --sensors Sentinel-2 \
    --gpu-hours 6 \
    --output-dir ./urban_segmentation
```

### Example 3: Programmatic Usage

```python
from src.agent.graph import create_ml_agent_graph
from src.agent.state import AgentState, init_phase_tracking

# Create agent
graph = create_ml_agent_graph()

# Initialize state
state = AgentState(
    paper_uri="arxiv:2103.14030",
    task_hint="segmentation",
    max_gpu_hours=4.0
)
state = init_phase_tracking(state)

# Run
result = graph.invoke(state, config={"configurable": {"thread_id": state.run_id}})

# Access results
print(f"Dataset: {result['dataset_info']['name']}")
print(f"Model: {result['code_artifacts']['model_architecture']}")
```

---

## Testing Different Components

### Test Dataset Loader

```python
# Test synthetic dataset loading
python src/data/synthetic_dataset.py \
    --data-dir ./test_data/synthetic_eo_dataset \
    --batch-size 8
```

### Test Individual Tools

```python
# Run examples
python example.py  # Demonstrates all tools

# Or individually:
from src.tools.dataset_resolver import DatasetResolver

resolver = DatasetResolver()
matches = resolver.find_matching_datasets(
    task_type="classification",
    modality=["optical"],
    resolution_m=10
)
```

---

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_comprehensive.py::TestStateManagement -v

# Generate synthetic data
python scripts/generate_synthetic_data.py

# Evaluate agent (mock mode, no API keys)
python scripts/evaluate_agent.py --mode mock --quick

# Evaluate agent (real LLMs)
python scripts/evaluate_agent.py --mode real --max-papers 3

# Check Python syntax
python -m py_compile src/**/*.py

# Format code
black src/ tests/ scripts/

# Type checking
mypy src/
```

---

## Troubleshooting

### Issue: "OpenAI API key not found"

**Solution:**
```bash
# Option 1: Set in .env file
echo "OPENAI_API_KEY=sk-..." >> .env

# Option 2: Export as environment variable
export OPENAI_API_KEY=sk-...

# Option 3: Use mock mode for testing
python scripts/evaluate_agent.py --mode mock
```

### Issue: "File not found: paper.pdf"

**Solution:**
```bash
# Use absolute path
python main.py --paper /full/path/to/paper.pdf

# Or use arXiv
python main.py --paper arxiv:2103.14030
```

### Issue: "No suitable dataset found"

**Solution:**
1. Check task type matches paper (classification vs segmentation)
2. Try without `--sensors` flag to allow any dataset
3. Add custom dataset to `src/tools/dataset_resolver.py`

### Issue: Tests failing with "scipy not found"

**Solution:**
```bash
pip install scipy>=1.11.0
```

### Issue: "Context length exceeded"

The agent automatically falls back to Gemini (1M tokens) if GPT-4 context is exceeded. No action needed!

---

## Architecture Overview

```
User Input (PDF/arXiv)
        ↓
┌───────────────────────┐
│   LangGraph Agent     │
│                       │
│  1. Parse Paper       │──→ Extract: task, method, data requirements
│  2. Validate Spec     │──→ Fill gaps with defaults
│  3. Resolve Dataset   │──→ Match to 10+ EO datasets
│  4. Synthesize Code   │──→ Generate PyTorch Lightning project
│  5. Prepare Data      │──→ Download dataset (stub)
│  6. Train/Evaluate    │──→ Run training (stub)
│  7. Generate Report   │──→ Markdown report (stub)
└───────────────────────┘
        ↓
Artifacts: code/ + data/ + report.md
```

**Key Features:**
- ✅ **Multi-LLM routing** (cost-optimized: ~$0.70/paper)
- ✅ **Stateful workflow** (checkpointing, retry logic)
- ✅ **Fallback mechanisms** (template code if LLM fails)
- ✅ **Validated inputs** (robust error handling)
- ⚠️ **Stub implementations** (training, data download need implementation)

---

## Performance Benchmarks

| Component | Time | Cost |
|-----------|------|------|
| Paper parsing | ~30s | $0.02 |
| Dataset resolution | ~10s | $0.05 |
| Code generation | ~60s | $0.50 |
| **Total (no training)** | **~2 min** | **~$0.67** |

**System Requirements (CPU-only mode):**
- Memory: 4 GB RAM
- Storage: 10 MB (testing), 100 GB (full datasets)
- Time: 2-5 minutes per paper

---

## Next Steps

1. **Implement Training Loop**
   - Integrate with PyTorch Lightning Trainer
   - Add W&B logging
   - Implement actual dataset download

2. **Extend Dataset Catalog**
   - Add more EO datasets
   - Support custom datasets
   - Improve matching heuristics

3. **Add More Task Types**
   - Object detection (oriented bounding boxes)
   - Change detection
   - Time series prediction

4. **Production Deployment**
   - Dockerize
   - Add web UI (Streamlit/Gradio)
   - Batch processing mode

---

## Resources

- **Documentation**: [README.md](README.md)
- **Examples**: [example.py](example.py)
- **Tests**: [tests/](tests/)
- **LangChain Docs**: https://python.langchain.com/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/

---

## Contributing

We welcome contributions! Areas of interest:

- 🔧 **Implement training loop** (high priority)
- 📊 **Add more datasets**
- 🧪 **Improve test coverage**
- 📝 **Better prompt engineering**
- 🎨 **UI development**

---

## Support

**Questions?**
- 📖 Check [README.md](README.md)
- 🐛 Report issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

---

**Built with [LangChain](https://langchain.com) • [LangGraph](https://github.com/langchain-ai/langgraph) • [PyTorch Lightning](https://lightning.ai)**

**License:** MIT
