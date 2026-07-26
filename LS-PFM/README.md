# LS-PFM Reproduction

A compact reproduction package for running the saved Cluster3 LS-PFM checkpoint on the Cluster1 shielding and decomming test sets. The reported metrics are adjusted-only.

## Directory

```text
LS-PFM/
  checkpoints/              # best.pt checkpoint
  configs/                  # fixed shielding/decomming configs
  data/
    cluster1/               # Cluster1 test data and labels
    cluster3/               # Cluster3 training data
  src/                      # model, data, topology, and eval code
  test/                     # fixed-best test entry
  train/                    # Cluster3 training entry
  outputs/test_fixed_best/  # adjusted test outputs
  requirements.txt
```

## Install

Python 3.10 is recommended. This package was verified with Python 3.10.18.

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
python test/test_fixed_best.py --preset all
```

Windows shortcut:

```bat
test\run_test_windows.bat
```

Single preset:

```bash
python test/test_fixed_best.py --preset shielding
python test/test_fixed_best.py --preset decomming
```

Outputs are written to:

```text
outputs/test_fixed_best/<preset>/
```

Metrics JSON files keep only adjusted metrics. Score CSV files keep only adjusted prediction columns.

## Retrain

```bat
train\run_train_cluster3_windows.bat
```

Default training output:

```text
outputs/train_cluster3/
```

## Expected Adjusted Results

- shielding adjusted F1: `0.733333`
- decomming adjusted F1: `0.804878`