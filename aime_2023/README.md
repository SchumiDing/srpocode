---
dataset_info:
  features:
  - name: problem_idx
    dtype: int64
  - name: problem
    dtype: string
  - name: answer
    dtype: int64
  splits:
  - name: train
    num_bytes: 11349
    num_examples: 30
  download_size: 11114
  dataset_size: 11349
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
