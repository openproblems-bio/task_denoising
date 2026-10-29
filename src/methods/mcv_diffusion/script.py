import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

## VIASH START
par = {
    "input_train": "resources_test/task_denoising/cxg_immune_cell_atlas/train.h5ad",
    "output": "output_mcv_diffusion.h5ad",
    "n_pcs": 50,
    "n_neighbors": 15,
    "t_grid": [1, 2, 3, 4, 6, 8, 12, 16],
    "b_grid": [0.3, 0.6, 0.8, 0.9, 1.0],
    "mcv_frac": 0.888888889,
    "mcv_gene_frac": 0.25,
    "seed": 0,
}
meta = {"name": "mcv_diffusion"}
## VIASH END

# Denoising by kNN-graph diffusion in log-CP10k space, with the diffusion time t and the
# restart weight b calibrated by molecular cross-validation (MCV, Batson et al. 2019) on the
# training counts themselves:
#   Y = log1p(1e4 * X / libsize)        (per cell)
#   Z = (1 - b) * Y + b * P^t Y          (P: row-normalised symmetric kNN graph from PCA of Y)
#   denoised = expm1(Z), rescaled per cell to the observed library size
# (t, b) minimise the MSE between log1p(CP10k) of the denoised train' split and of a held-out
# test' split obtained by binomially splitting the training molecules (no test data is used).

rng = np.random.RandomState(par["seed"])
GENE_CHUNK = 2048


def log_cp10k(X):
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64)
    lib[lib == 0] = 1
    Y = sp.csr_matrix(sp.diags(1e4 / lib) @ X, dtype=np.float32)
    Y.data = np.log1p(Y.data)
    return Y


def transition_matrix(X):
    """Symmetrised, row-normalised kNN graph (incl. self) on the PCA of log-CP10k (scanpy HVG + PCA)."""
    a = ad.AnnData(X=log_cp10k(X))
    sc.pp.highly_variable_genes(a)
    n_pcs = min(par["n_pcs"], a.n_obs - 1, int(a.var["highly_variable"].sum()) - 1)
    sc.pp.pca(a, n_comps=n_pcs, mask_var="highly_variable", random_state=par["seed"])
    E = a.obsm["X_pca"]
    k = min(par["n_neighbors"], a.n_obs)
    idx = NearestNeighbors(n_neighbors=k).fit(E).kneighbors(E, return_distance=False)
    n = E.shape[0]
    W = sp.csr_matrix((np.ones(idx.size, dtype=np.float32), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = (W + W.T) * 0.5
    return sp.csr_matrix(sp.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W, dtype=np.float32)


def diffusion_states(P, Yc, ts):
    """Yield (t, P^t Yc) for t in ts (sorted), Yc dense."""
    Z = Yc
    for t in range(1, max(ts) + 1):
        Z = P @ Z
        if t in ts:
            yield t, Z


def calibrate(X):
    """MCV: split molecules of X into train'/test', return (t, b) minimising the log-CP10k MSE on test'."""
    X = sp.csr_matrix(X)
    tr = X.copy()
    tr.data = rng.binomial(np.rint(X.data).astype(np.int32), par["mcv_frac"]).astype(np.float32)
    te = X.copy()
    te.data = (X.data - tr.data).astype(np.float32)
    tr.eliminate_zeros(); te.eliminate_zeros()
    keep = np.asarray(tr.sum(0)).ravel() > 0
    tr, te = tr[:, keep].tocsr(), te[:, keep].tocsc()
    n, g = tr.shape
    P = transition_matrix(tr)
    Y = sp.csc_matrix(log_cp10k(tr))
    LT = sp.csc_matrix(log_cp10k(te))
    ts, bs = sorted(par["t_grid"]), list(par["b_grid"])
    # pass 1: per-cell totals of expm1(Z) for every (t, b) over all genes
    rows = {}
    for j in range(0, g, GENE_CHUNK):
        B = Y[:, j:j + GENE_CHUNK].toarray()
        for t, Zt in diffusion_states(P, B, ts):
            for b in bs:
                rows[(t, b)] = rows.get((t, b), 0) + np.expm1((1 - b) * B + b * Zt).sum(1, dtype=np.float64)
    # pass 2: squared error on a random subset of genes (unbiased estimate of the MSE)
    genes = np.sort(rng.choice(g, max(1, int(round(par["mcv_gene_frac"] * g))), replace=False))
    err = {key: 0.0 for key in rows}
    for j in range(0, len(genes), GENE_CHUNK):
        gj = genes[j:j + GENE_CHUNK]
        B = Y[:, gj].toarray()
        T = LT[:, gj].toarray()
        for t, Zt in diffusion_states(P, B, ts):
            for b in bs:
                s = np.maximum(rows[(t, b)], 1e-30)[:, None]
                V = np.log1p(np.expm1((1 - b) * B + b * Zt) / s * 1e4)
                err[(t, b)] += float(((T - V) ** 2).sum())
    (t_best, b_best), e_best = min(err.items(), key=lambda kv: kv[1])
    print(f"MCV grid: best t={t_best} b={b_best} mse~{e_best / (n * len(genes)):.5f}", flush=True)
    return t_best, b_best


def denoise(X, t, b):
    X = sp.csr_matrix(X)
    n, g = X.shape
    P = transition_matrix(X)
    Y = sp.csc_matrix(log_cp10k(X))
    out = np.empty((n, g), dtype=np.float32)
    for j in range(0, g, GENE_CHUNK):
        B = Y[:, j:j + GENE_CHUNK].toarray()
        Zt = dict(diffusion_states(P, B, [t]))[t]
        out[:, j:j + GENE_CHUNK] = np.expm1((1 - b) * B + b * Zt)
    lib = np.asarray(X.sum(1)).ravel()
    s = out.sum(1, dtype=np.float64)
    s[s == 0] = 1
    out *= (lib / s).astype(np.float32)[:, None]
    return out


print("Load data", flush=True)
input_train = ad.read_h5ad(par["input_train"])
X = sp.csr_matrix(input_train.layers["counts"]).astype(np.float32)

print("Calibrate diffusion time and restart weight by molecular cross-validation", flush=True)
t_best, b_best = calibrate(X)

print("Denoise", flush=True)
denoised = denoise(X, t_best, b_best)

output = ad.AnnData(
    obs=input_train.obs[[]],
    var=input_train.var[[]],
    layers={"denoised": denoised},
    uns={"dataset_id": input_train.uns["dataset_id"], "method_id": meta["name"],
         "mcv_t": t_best, "mcv_b": b_best},
)
print("Write data", flush=True)
output.write_h5ad(par["output"], compression="gzip")
