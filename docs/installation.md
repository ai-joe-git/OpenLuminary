## 5. Installation Instructions

```markdown
# Installation Guide

## Prerequisites
- Python 3.9+
- Git
- CUDA-compatible GPU (recommended for full Qwen3 usage)

## Step 1: Clone the repository
```
git clone https://github.com/yourusername/OpenLuminary.git
cd OpenLuminary
```

## Step 2: Set up a virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 3: Install dependencies
```
pip install -e .
```

## Step 4: Run the demo application
```
streamlit run app.py
```

## For Development

Install additional development dependencies:
```
pip install -e ".[dev]"
```

Run tests:
```
pytest
```
