import os

import anndata as ad
import scprint
import torch
from huggingface_hub import hf_hub_download
from scdataloader import Preprocessor
from scprint import scPrint
from scprint.tasks import Denoiser
import numpy as np

## VIASH START
par = {
    "input_train": "resources_test/task_batch_integration/cxg_immune_cell_atlas/train.h5ad",
    "output": "output.h5ad",
    "model_name": "large",
    "model": None,
}
meta = {"name": "scprint"}
## VIASH END

print(f"====== scPRINT version {scprint.__version__} ======", flush=True)

print("\n>>> Reading input data...", flush=True)
input = ad.read_h5ad(par["input_train"])
print(input)

print("\n>>> Preprocessing data...", flush=True)
adata = ad.AnnData(
    X=input.layers["counts"]
)
adata.obs_names = input.obs_names
adata.var_names = input.var_names
if input.uns["dataset_organism"] == "homo_sapiens":
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
elif input.uns["dataset_organism"] == "mus_musculus":
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:10090"
else:
    raise ValueError(
        f"scPRINT requires human or mouse data, not '{input.uns['dataset_organism']}'"
    )

preprocessor = Preprocessor(
    # Lower this threshold for test datasets
    min_valid_genes_id=1000 if input.n_vars < 2000 else 10000,
    # Turn off cell filtering to return results for all cells
    filter_cell_by_counts=False,
    min_nnz_genes=False,
    do_postp=False,
    # Skip ontology checks
    skip_validate=True,
)
adata = preprocessor(adata)
print(adata)

model_checkpoint_file = par["model"]
if model_checkpoint_file is None:
    print(f"\n>>> Downloading '{par['model_name']}' model...", flush=True)
    model_checkpoint_file = hf_hub_download(
        repo_id="jkobject/scPRINT", filename=f"{par['model_name']}.ckpt"
    )
print(f"Model checkpoint file: '{model_checkpoint_file}'", flush=True)
model = scPrint.load_from_checkpoint(
    model_checkpoint_file,
    transformer="normal",  # Don't use this for GPUs with flashattention
    precpt_gene_emb=None,
)

print("\n>>> Denoising data...", flush=True)
if torch.cuda.is_available():
    print("CUDA is available, using GPU", flush=True)
    precision = "16-mixed"
    dtype = torch.float16
else:
    print("CUDA is not available, using CPU", flush=True)
    precision = "32"
    dtype = torch.float32
n_cores_available = len(os.sched_getaffinity(0))
print(f"Using {n_cores_available} worker cores")
denoiser = Denoiser(
    num_workers=n_cores_available,
    precision=precision,
    max_cells=adata.n_obs,
    doplot=False,
    dtype=dtype,
)
_, idxs, genes, expr_pred = denoiser(model, adata)
print(f"Predicted expression dimensions: {expr_pred.shape}")

print("\n>>> Applying denoising...", flush=True)
adata.X = adata.X.tolil()
idxs = idxs if idxs is not None else range(adata.shape[0])
for i, idx in enumerate(idxs):
    adata.X[idx, adata.var.index.get_indexer(genes)] = expr_pred[i]
adata.X = adata.X.tocsr()
print(adata)

print("\n>>> Storing output...", flush=True)
output = ad.AnnData(
    layers={
        "denoised": adata.X[:, adata.var.index.get_indexer(input.var_names)],
    },
    obs=input.obs[[]],
    var=input.var[[]],
    uns={
        "dataset_id": input.uns["dataset_id"],
        "method_id": meta["name"],
    },
)
print(output)

print("\n>>> Writing output AnnData to file...", flush=True)
output.write_h5ad(par["output"], compression="gzip")

print("\n>>> Done!", flush=True)
