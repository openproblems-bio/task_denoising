import sys
import random
import anndata as ad
import numpy as np

## VIASH START
par = {
    'input': "resources_test/common/cxg_mouse_pancreas_atlas/dataset.h5ad",
    'output_train': "train.h5ad",
    'output_test': "test.h5ad",
    'train_frac': 0.9,
    'seed': 0
}
meta = {
    "name": "process_dataset",
    "resources_dir": "src/tasks/denoising/process_dataset"
}
## VIASH END

# add helper scripts to path
sys.path.append(meta["resources_dir"])
from helper import split_molecules

# set random state
random_state = np.random.RandomState(par['seed'])

print(">> Load Data", flush=True)
adata = ad.read_h5ad(par["input"])

# limit to max number of observations
adata_output = adata.copy()
if adata.n_obs > par["n_obs_limit"]:
    print(">> Subsampling the observations", flush=True)
    print(f">> Setting seed to {par['seed']}")
    random.seed(par["seed"])
    if "batch" not in adata.obs:
        obs_filt = np.ones(dtype=np.bool_, shape=adata.n_obs)
        obs_index = np.random.choice(np.where(obs_filt)[0], par["n_obs_limit"], replace=False)
        adata_output = adata[obs_index].copy()
    else:
        batch_counts = adata.obs.groupby('batch').size()
        filtered_batches = batch_counts[batch_counts <= par["n_obs_limit"]]
        sorted_filtered_batches = filtered_batches.sort_values(ascending=False)
        selected_batch = sorted_filtered_batches.index[0]
        adata_output = adata[adata.obs["batch"]==selected_batch,:].copy()
        
# remove all layers except for counts
for key in list(adata_output.layers.keys()):
    if key != "counts":
        del adata_output.layers[key]

# round counts and convert to int
counts = np.array(adata_output.layers["counts"]).round().astype(int)

print(">> process and split data", flush=True)
train_data, test_data = split_molecules(
    counts.data, par["train_frac"], 0.0, random_state
)

X_train = counts.copy()
X_test = counts.copy()
X_train.data = train_data
X_test.data = test_data
X_train.eliminate_zeros()
X_test.eliminate_zeros()

# copy adata to train_set, test_set
output_train = ad.AnnData(
    layers={"counts": X_train},
    obs=adata_output.obs[[]],
    var=adata_output.var[[]],
    uns={"dataset_id": adata_output.uns["dataset_id"]}
)
test_uns_keys = ["dataset_id", "dataset_name", "dataset_url", "dataset_reference", "dataset_summary", "dataset_description", "dataset_organism"]
output_test = ad.AnnData(
    layers={"counts": X_test},
    obs=adata_output.obs[[]],
    var=adata_output.var[[]],
    uns={key: adata_output.uns[key] for key in test_uns_keys}
)

# add additional information for the train set
output_test.uns["train_sum"] = X_train.sum()

# Remove no cells that do not have enough reads
is_missing = np.array(X_train.sum(axis=0) == 0)

output_train = output_train[:, ~is_missing.flatten()]
output_test = output_test[:, ~is_missing.flatten()]

print(">> Write to file", flush=True)
output_train.write_h5ad(par["output_train"])
output_test.write_h5ad(par["output_test"])
