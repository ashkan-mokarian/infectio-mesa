#!/bin/bash
python ./infectio/evaluate.py \
    --reference ./examples/vacv-epithelial/reference_metrics_for_M061_WR_handpicked.csv \
    --root ./output/CHANGE \
    --tcol infected-count area\(um2\) radial-velocity\(um/min\) \
    --rmeancol inf-count-mean area-mean\(um2\) radial-velocity-mean\(um/min\) \
    --rstdcol inf-count-std area-std\(um2\) radial-velocity-std\(um/min\) \
    --output ./output/CHANGE/0_evaluation/bulk_evaluate.csv