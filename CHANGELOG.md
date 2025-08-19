# denoising devel

## NEW FUNCTIONALITY

* Add new method: CellMapper (PR #31)

## MINOR CHANGES

* Update the Viash version to 0.9.4 (PR #32)

* Dependencies were moved to main openproblems repository (PR #32)

## BUG FIXES

* Update scPRINT, including a new default model (PR #26)


# denoising v1.0.0

## BREAKING CHANGES

* Update to viash 0.9.0 RC6

* Directory structure has been updated.

* Update to viash 0.9.0 (PR #13).

## NEW FUNCTIONALITY

* Add `CHANGELOG.md` (PR #7).

* Update `process_dataset` component to subsample large datasets (PR #14).

* Add the scPRINT method (PR #25)

## MAJOR CHANGES

* Revamp `scripts` directory (PR #13).

* Relocated `process_datasets` to `data_processors/process_datasets` (PR #13).

## MINOR CHANGES

* Remove dtype parameter in `.Anndata()` (PR #6).

* Fix target_sum deprecation warning in `mse` mmetric (PR #8).

* Update `task_name` variable to denoising in component scripts (PR #9).

* Update docker containers used in components (PR #12).

* Set `numpy<2` for some failing methods (PR #13).

* Small changes to api file names (PR #13).

* Update test_resources path in components (PR #18).

* Update workflows to use core repository dependency (PR #20).

* Update the `common` submodule (PR #24)

* Use the common `checkItemAllowed()` for the method check in the benchmark workflow (PR #24)

* Use the `cxg_immune_cell_atlas` dataset instead of the `cxg_mouse_pancreas_atlas` for testing (PR #24)

* Update `README` (PR #24)

* Add a base method API schema (PR #24)

* Add `dataset_organism` to training input files (PR #24)

* Remove n_obs_limit default setting.

## BUG FIXES

* Update the nextflow workflow dependencies (PR #17).

* Fix paths in scripts (PR #18).

* Subsample datasets by batch if batch is defined (PR #22).

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
