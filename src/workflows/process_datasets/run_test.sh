#!/bin/bash

# Run this prior to executing this script:
# bin/viash_build -q 'batch_integration'

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

nextflow run . \
  -main-script target/nextflow/workflows/process_datasets/main.nf \
  -profile docker \
  -entry auto \
  -c common/nextflow_helpers/labels_ci.config \
  --id run_test \
  --input_states "resources_test/common/**/state.yaml" \
  --rename_keys 'input:output_dataset' \
  --settings '{"output_train": "train.h5ad", "output_test": "test.h5ad"}' \
  --publish_dir "resources_test/task_denoising"