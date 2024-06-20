#!/bin/bash
python ./infectio/evaluate.py \
    --reference ./examples/vacv-epithelial/reference_metrics_for_M061_WR_handpicked.csv \
    --root ./output/paridentify_0/ \
    --tcol Infected_Count Mean_Radius \
    --rmeancol inf-count-mean radius-mean(um) \
    --rstdcol inf-count-std radius-std(um) \
    --output ./output/paridentify_0/0_evaluation/bulk_evaluate.csv