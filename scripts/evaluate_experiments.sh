#!/bin/bash
python ./infectio/evaluate.py \
    --target_csv ./examples/vacv-epithelial/reference_metrics_for_M061_dVGFdF11_handpicked.csv \
    --root ./output/dVGFdF11 \
    --n_dataset 11 \
    --output ./output/CHANGE/0_evaluation/bulk_evaluate.csv