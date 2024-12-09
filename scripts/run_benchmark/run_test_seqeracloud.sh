#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

resources_test_s3=s3://openproblems-data/resources_test/task_denoising
publish_dir_s3="s3://openproblems-nextflow/temp/results/task_denoising/$(date +%Y-%m-%d_%H-%M-%S)"

# write the parameters to file
cat > /tmp/params.yaml << HERE
id: cxg_immune_cell_atlas
input_train: $resources_test_s3/cxg_immune_cell_atlas/train.h5ad
input_test: $resources_test_s3/cxg_immune_cell_atlas/test.h5ad
output_state: "state.yaml"
publish_dir: $publish_dir_s3
HERE

tw launch https://github.com/openproblems-bio/task_denoising.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/run_benchmark/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --config common/nextflow_helpers/labels_tw.config \
  --labels task_denoising,test
