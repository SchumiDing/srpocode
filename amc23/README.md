---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: answer
    dtype: float64
  - name: url
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: test
    num_bytes: 14871
    num_examples: 40
  download_size: 11935
  dataset_size: 14871
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
