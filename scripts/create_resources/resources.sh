#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

cat > /tmp/params.yaml << 'HERE'
input_states: s3://openproblems-data/resources/datasets/**/log_cp10k/state.yaml
rename_keys: 'input:output_dataset'
settings: '{"output_train": "$id/train.h5ad", "output_test": "$id/test.h5ad"}'
output_state: "$id/state.yaml"
publish_dir: s3://openproblems-data/resources/denoising/datasets
HERE

tw launch https://github.com/openproblems-bio/task_denoising.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/process_datasets/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --entry-name auto \
  --config common/nextflow_helpers/labels_tw.config \
  --labels denoising,process_datasets
