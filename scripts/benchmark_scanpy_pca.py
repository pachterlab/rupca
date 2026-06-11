import json
import sys
import time

import scanpy as sc
from anndata import AnnData
from scipy.io import mmread
from scipy.sparse import csr_matrix


def main():
    matrix_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/lpachter/Dropbox/claude/projects/rutest/pbmc10k/normalized_log1pPF.mtx"
    )
    n_components = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    start = time.perf_counter()
    x = csr_matrix(mmread(matrix_path))
    load_seconds = time.perf_counter() - start

    best = float("inf")
    total = 0.0
    for _ in range(repeats):
        adata = AnnData(x.copy())
        start = time.perf_counter()
        sc.pp.pca(
            adata,
            n_comps=n_components,
            zero_center=True,
            svd_solver="arpack",
            random_state=0,
        )
        secs = time.perf_counter() - start
        best = min(best, secs)
        total += secs
        adata.obsm["X_pca"].sum()

    print(
        json.dumps(
            {
                "rows": int(x.shape[0]),
                "cols": int(x.shape[1]),
                "nnz": int(x.nnz),
                "load_seconds": load_seconds,
                "pca_best_seconds": best,
                "pca_mean_seconds": total / repeats,
                "repeats": repeats,
                "scanpy_version": sc.__version__,
            }
        )
    )


if __name__ == "__main__":
    main()
