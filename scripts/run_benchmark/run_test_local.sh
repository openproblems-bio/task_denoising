#!/bin/bash

# get the root of the directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# ensure that the command below is run from the root of the repository
cd "$REPO_ROOT"

set -e

echo "Running benchmark on test data"
echo "  Make sure to run 'scripts/project/build_all_docker_containers.sh'!"

# generate a unique id
RUN_ID="testrun_$(date +%Y-%m-%d_%H-%M-%S)"
publish_dir="temp/results/${RUN_ID}"

nextflow run . \
  -main-script target/nextflow/workflows/run_benchmark/main.nf \
  -profile docker \
  -resume \
  -c common/nextflow_helpers/labels_ci.config \
  --id cxg_mouse_pancreas_atlas \
  --input_train resources_test/task_denoising/cxg_mouse_pancreas_atlas/train.h5ad \
  --input_test resources_test/task_denoising/cxg_mouse_pancreas_atlas/test.h5ad \
  --output_state state.yaml \
  --publish_dir "$publish_dir"
