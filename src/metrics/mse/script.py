import anndata as ad
import scanpy as sc
import scprep
import sklearn.metrics

## VIASH START
par = {
    "input_test": "resources_test/task_denoising/cxg_immune_cell_atlas/test.h5ad",
    "input_prediction": "resources_test/task_denoising/cxg_immune_cell_atlas/denoised.h5ad",
    "output": "output_mse.h5ad",
}
meta = {"name": "mse"}
## VIASH END

print("Load data", flush=True)
input_denoised = ad.read_h5ad(par["input_prediction"])
input_test = ad.read_h5ad(par["input_test"])

test_data = ad.AnnData(X=input_test.layers["counts"])
denoised_data = ad.AnnData(X=input_denoised.layers["denoised"])

print("Normalize data", flush=True)

# scaling and transformation
target_sum = 10000

sc.pp.normalize_total(test_data, target_sum=target_sum)
sc.pp.log1p(test_data)

sc.pp.normalize_total(denoised_data, target_sum=target_sum)
sc.pp.log1p(denoised_data)

print("Compute mse value", flush=True)
error = sklearn.metrics.mean_squared_error(
    scprep.utils.toarray(test_data.X), scprep.utils.toarray(denoised_data.X)
)

print("Store mse value", flush=True)
output = ad.AnnData(
    uns={key: val for key, val in input_test.uns.items()},
)

output.uns["method_id"] = input_denoised.uns["method_id"]
output.uns["metric_ids"] = meta["name"]
output.uns["metric_values"] = error

print("Write adata to file", flush=True)
print(output.uns)
output.write_h5ad(par["output"], compression="gzip")
