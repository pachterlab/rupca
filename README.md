# rupca (rust PCA)

`rupca` is a small Rust crate that mirrors the sparse centered PCA path used by Scanpy for sparse input with `zero_center=True`.
It is compatible with both ordinary sparse inputs, such as logPF-normalized matrices, and PFlogPF / shifted-CLR inputs represented without densifying the matrix; see [*Depth normalization for single-cell genomics count data*](https://www.biorxiv.org/content/10.1101/2022.05.06.490859v3) by A. Sina Booeshaghi, Ingileif B. Hallgrímsdóttir, Ángel Gálvez-Merchán, and Lior Pachter (doi: `10.1101/2022.05.06.490859v3`).

The implemented path is:

1. Compute per-feature means and variances.
2. Represent centering implicitly as `X - 1 * mean^T`.
3. Build the normal operator on the smaller side:
   - `(X - mean)^T (X - mean)` when `n_samples >= n_features`
   - `(X - mean) (X - mean)^T` otherwise
4. Run a Rust-native symmetric Lanczos/Ritz eigensolver on that implicit operator.
5. Form `Av` exactly as SciPy does.
6. Run dense SVD on `Av`.
7. Recover scores, components, singular values, explained variance, and noise variance in the same style as sklearn PCA.

The current public entrypoints are:

- `pca_scanpy_sparse_csr(&CsrMatrix, ScanpyPcaParams) -> Result<ScanpyPcaResult>`
- `pca_shifted_clr_sparse_csr(&ShiftedClrCsrMatrix, ScanpyPcaParams) -> Result<ScanpyPcaResult>`

The plain sparse matrix format is a simple CSR container owned by `rupca`.
`ShiftedClrCsrMatrix` represents the dense matrix `sparse[i, j] - row_center[i]`.
This keeps PFlogPF / shifted-CLR data as a sparse shifted-log matrix plus a row
centering vector while preserving implicit column centering inside PCA.

## Status

The crate currently:

- compiles cleanly
- passes unit tests on both tall and wide sparse matrices against the corresponding centered dense SVD reference
- passes unit tests on both tall and wide shifted-CLR-style matrices against a centered dense SVD reference
- passes representation-level tests showing sparse PFlogPF / shifted-CLR operations match dense materialization to floating-point precision
- vendors the exact ARPACK symmetric reference sources used for the Scanpy/sklearn sparse path in [vendor/arpack-ng/SRC](/Users/lpachter/Dropbox/claude/projects/rupca/vendor/arpack-ng/SRC)

## Notes

- This is intended to match the Scanpy sparse PCA algorithmic path, not the full Scanpy Python object model.
- The eigensolver is now Rust-native rather than calling external ARPACK.
- In PBMC 10k benchmarks with 50 PCs, Rust sparse logPF PCA was about 1.35x faster than Scanpy sparse logPF PCA; Rust PFlogPF / shifted-CLR PCA with `ncv=250` was about 1.41x faster than Scanpy sparse logPF PCA.
- The imported ARPACK reference sources and porting map are documented in [docs/arpack/PORTING.md](/Users/lpachter/Dropbox/claude/projects/rupca/docs/arpack/PORTING.md).
