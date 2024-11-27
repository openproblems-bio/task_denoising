import anndata as ad
from scdataloader import Preprocessor
import sys
from huggingface_hub import hf_hub_download
from scprint.tasks import Embedder
from scprint import scPrint
import scprint
import torch
import os

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "model_name": "large",
    "model": None,
}
meta = {"name": "scprint"}
## VIASH END

sys.path.append(meta["resources_dir"])
from read_anndata_partial import read_anndata

print(f"====== scPRINT version {scprint.__version__} ======", flush=True)

print("\n>>> Reading input data...", flush=True)
input = read_anndata(par["input"], X="layers/counts", obs="obs", var="var", uns="uns")
if input.uns["dataset_organism"] == "homo_sapiens":
    input.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
elif input.uns["dataset_organism"] == "mus_musculus":
    input.obs["organism_ontology_term_id"] = "NCBITaxon:10090"
else:
    raise ValueError(
        f"scPRINT requires human or mouse data, not '{input.uns['dataset_organism']}'"
    )
adata = input.copy()

print("\n>>> Preprocessing data...", flush=True)
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

print("\n>>> Embedding data...", flush=True)
if torch.cuda.is_available():
    print("CUDA is available, using GPU", flush=True)
    precision = "16"
    dtype = torch.float16
else:
    print("CUDA is not available, using CPU", flush=True)
    precision = "32"
    dtype = torch.float32
n_cores_available = len(os.sched_getaffinity(0))
print(f"Using {n_cores_available} worker cores")
embedder = Embedder(
    how="random expr",
    max_len=4000,
    add_zero_genes=0,
    num_workers=n_cores_available,
    doclass=False,
    doplot=False,
    precision=precision,
    dtype=dtype,
)
embedded, _ = embedder(model, adata, cache=False)

print("\n>>> Storing output...", flush=True)
output = ad.AnnData(
    obs=input.obs[[]],
    var=input.var[[]],
    obsm={
        "X_emb": embedded.obsm["scprint"],
    },
    uns={
        "dataset_id": input.uns["dataset_id"],
        "normalization_id": input.uns["normalization_id"],
        "method_id": meta["name"],
    },
)
print(output)

print("\n>>> Writing output AnnData to file...", flush=True)
output.write_h5ad(par["output"], compression="gzip")

print("\n>>> Done!", flush=True)
