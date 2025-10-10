# AutoGluon Tabular Benchmark

## Installation

### Prerequisites

- Python 3.12 or higher

### Step 1: Clone the Repository

```bash
git clone git@github.com:do3-173/thesis.git
cd thesis
```

### Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install AutoML Multimodal Benchmark Package

The benchmark requires the `auto_mm_bench` package from the AutoML multimodal benchmark suite:

```bash
# Navigate to the automl_multimodal_benchmark directory (adjust path as needed)
cd ../automl_multimodal_benchmark/multimodal_text_benchmark
pip install -U -e .
cd ../../thesis  # Return to thesis directory
```

**Note**: Make sure you have the `automl_multimodal_benchmark` repository cloned in the parent directory or adjust the path accordingly.

### Step 5: Verify Installation

Test that all packages are installed correctly:

```bash
python -c "
import autogluon.tabular
import auto_mm_bench
from auto_mm_bench.datasets import dataset_registry
print(f'AutoGluon and auto_mm_bench installed successfully')
"
```
