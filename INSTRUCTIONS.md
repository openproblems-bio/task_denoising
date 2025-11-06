# Instructions

This is a guide on what to do after you have created a new task repository from the template. More in depth information about how to create a new task can be found in the [OpenProblems Documentation](https://openproblems.bio/documentation/create_task/).

## First things first

* Update the `_viash.yaml` file with the correct task information.
* Update the `src/api/task_info.yaml` file with the information you have provied in the task issue.

## Resources

THe OpenProblems team has provided some test resources that can be used to test the task. These resources are stored in the `resources` folder. The `scripts/download_resources.sh` script can be used to download these resources.

If these resources are not sufficient, you can add more resources to the `resources` folder. The `scripts/download_resources.sh` script can be updated to download these resources.





<!-- Add to readme 
* update _viash.yaml
* update src/api/task_info.yaml
* update scripts/download_resources
-->

#!/bin/bash

echo "This script is not supposed to be run directly."
echo "Please run the script step-by-step."
exit 1

# sync resources
scripts/download_resources.sh

# create a new component
method_id="my_metric"
method_lang="python" # change this to "r" if need be

common/create_component/create_component -- \
  --language "$method_lang" \
  --name "$method_id"

# TODO: fill in required fields in src/task/methods/foo/config.vsh.yaml
# TODO: edit src/task/methods/foo/script.py/R

# test the component
viash test src/task/methods/$method_id/config.vsh.yaml

# rebuild the container (only if you change something to the docker platform)
# You can reduce the memory and cpu allotted to jobs in _viash.yaml by modifying .platforms[.type == "nextflow"].config.labels
viash run src/task/methods/$method_id/config.vsh.yaml -- \
  ---setup cachedbuild ---verbose

# run the method (using parquet as input)
viash run src/task/methods/$method_id/config.vsh.yaml -- \
  --de_train "resources/neurips-2023-kaggle/de_train.parquet" \
  --id_map "resources/neurips-2023-kaggle/id_map.csv" \
  --output "output/prediction.parquet"

# run the method (using h5ad as input)
viash run src/task/methods/$method_id/config.vsh.yaml -- \
  --de_train_h5ad "resources/neurips-2023-kaggle/2023-09-12_de_by_cell_type_train.h5ad" \
  --id_map "resources/neurips-2023-kaggle/id_map.csv" \
  --output "output/prediction.parquet"

# run evaluation metric
viash run src/task/metrics/mean_rowwise_error/config.vsh.yaml -- \
  --de_test "resources/neurips-2023-kaggle/de_test.parquet" \
  --prediction "output/prediction.parquet" \
  --output "output/score.h5ad"

# print score on kaggle test dataset
python -c 'import anndata; print(anndata.read_h5ad("output/score.h5ad").uns)'