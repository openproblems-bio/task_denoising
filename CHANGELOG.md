# denoising 0.1.0

## BREAKING CHANGES

* Update to viash 0.9.0 RC6

* Directory structure has been updated.

## NEW FUNCTIONALITY

* Add `CHANGELOG.md` (PR #7).

## MINOR CHANGES

* Remove dtype parameter in `.Anndata()` (PR #6).

* Fix target_sum deprecation warning in `mse` mmetric (PR #8).

* Update `task_name` variable to denoising in component scripts (PR #9).

## transfer from openproblems-v2 repository

### NEW FUNCTIONALITY

* `api/file_*`: Created a file format specifications for the h5ad files throughout the pipeline.

* `api/comp_*`: Created an api definition for the split, method and metric components.

* `process_dataset`: Added a component for processing common datasets into task-ready dataset objects.

* `resources_test/denoising/pancreas` with `src/tasks/denoising/resources_test_scripts/pancreas.sh`.
  
* `workflows/run`: Added nf-tower test script. (PR #205)

### V1 MIGRATION

* `control_methods/no_denoising`: Migrated from v1. Extracted from baseline method

* `control_methods/perfect_denoising`: Migrated from v1.Extracted from baseline method

* `methods/alra`: Migrated from v1. Changed from python to R and uses lg_cpm normalised data instead of L1 sqrt

* `methods/dca`: Migrated and adapted from v1.

* `methods/knn_smoothing`: Migrated and adapted from v1.

* `methods/magic`: Migrated from v1.

* `metrics/mse`: Migrated from v1.

* `metrics/poisson`: Migrated from v1.

### Changes from V1

* Anndata layers are used to store data instead of obsm
  
* extended the use of sparse data in methods unless it was not possible

* process_dataset also removes unnecessary data from train and test datasets not needed by the methods and metrics.