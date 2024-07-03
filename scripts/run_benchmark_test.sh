#!/bin/bash

cat > /tmp/params.yaml << 'HERE'
input_states: s3://openproblems-data/resources_test/denoising/**/state.yaml
rename_keys: 'input_train:output_train;input_test:output_test'
output_state: "state.yaml"
publish_dir: s3://openproblems-nextflow/temp/denoising/
HERE

tw launch https://github.com/openproblems-bio/task_denoising.git \
  --revision build/main \
  --pull-latest \
  --main-script target/nextflow/workflows/run_benchmark/main.nf \
  --workspace 53907369739130 \
  --compute-env 6TeIFgV5OY4pJCk8I0bfOh \
  --params-file /tmp/params.yaml \
  --entry-name auto \
  --config common/nextflow_helpers/labels_tw.config \
  --labels denoising,test