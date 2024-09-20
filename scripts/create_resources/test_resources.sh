#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

RAW_DATA=resources_test/common
DATASET_DIR=resources_test/task_denoising

mkdir -p $DATASET_DIR

# process dataset
viash run src/data_processors/process_dataset/config.vsh.yaml -- \
  --input $RAW_DATA/cxg_mouse_pancreas_atlas/dataset.h5ad \
  --output_train $DATASET_DIR/cxg_mouse_pancreas_atlas/train.h5ad \
  --output_test $DATASET_DIR/cxg_mouse_pancreas_atlas/test.h5ad

# run one method
viash run src/methods/magic/config.vsh.yaml -- \
    --input_train $DATASET_DIR/cxg_mouse_pancreas_atlas/train.h5ad \
    --output $DATASET_DIR/cxg_mouse_pancreas_atlas/denoised.h5ad

# run one metric
viash run src/metrics/poisson/config.vsh.yaml -- \
    --input_prediction $DATASET_DIR/cxg_mouse_pancreas_atlas/denoised.h5ad \
    --input_test $DATASET_DIR/cxg_mouse_pancreas_atlas/test.h5ad \
    --output $DATASET_DIR/cxg_mouse_pancreas_atlas/score.h5ad

# write manual state.yaml. this is not actually necessary but you never know it might be useful
cat > $DATASET_DIR/cxg_mouse_pancreas_atlas/state.yaml << HERE
id: cxg_mouse_pancreas_atlas
train: !file train.h5ad
test: !file test.h5ad
prediction: !file denoised.h5ad
score: !file score.h5ad
HERE

# only run this if you have access to the openproblems-data bucket
aws s3 sync --profile OP \
  "$DATASET_DIR" s3://openproblems-data/resources_test/task_denoising \
  --delete --dryrun
