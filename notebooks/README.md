# Notebooks

Notebooks are for exploration and for producing figures. Once logic is reused, move it into `src/trial_outcome/` and import it here. That way it is linted, tested, and shared.

## Naming

`<NN>-<initials>-<short-description>.ipynb`, for example `01-hs-eda-trial-status.ipynb`. The number gives the rough order in the pipeline:

| Prefix | Stage |
| --- | --- |
| `00` | Data acquisition (e.g. the provided full-dataset download notebook) |
| `01` | Exploratory data analysis |
| `02` | Labeling |
| `03` | Feature engineering |
| `04` | Modeling |
| `05` | Evaluation and interpretation |

## Setup

Select the `dev/bin/python` kernel. The project package is installed in editable mode, so this works from any notebook:

```python
from trial_outcome.config import RAW_DATA_DIR
from trial_outcome.data.ctgov import read_jsonl
```

Save presentation figures with `trial_outcome.visualization.save_figure`, which writes to `reports/figures/`. ruff lints and formats notebooks on commit, just like `.py` files.
