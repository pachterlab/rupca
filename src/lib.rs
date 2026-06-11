mod arpack_symm;
#[cfg(feature = "ruanndata")]
mod ruanndata_adapter;

#[cfg(feature = "ruanndata")]
pub use ruanndata_adapter::{
    dense_from_ruanndata, pca_scanpy_ruanndata, shifted_clr_csr_from_ruanndata,
    sparse_csr_from_ruanndata,
};

use arpack_symm::aup2::dsaup2_mode1;
use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuPcaError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("solver error: {0}")]
    Solver(String),
    #[error("SVD failed to produce requested factors")]
    MissingSvdFactors,
}

pub type Result<T> = std::result::Result<T, RuPcaError>;

#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: Vec<f64>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
}

impl CsrMatrix {
    pub fn validate(&self) -> Result<()> {
        if self.indices.len() != self.data.len() {
            return Err(RuPcaError::InvalidInput(
                "CSR data and indices length mismatch".to_string(),
            ));
        }
        if self.indptr.len() != self.n_rows + 1 {
            return Err(RuPcaError::InvalidInput(
                "CSR indptr length must be n_rows + 1".to_string(),
            ));
        }
        if *self.indptr.first().unwrap_or(&0) != 0 {
            return Err(RuPcaError::InvalidInput(
                "CSR indptr must start at 0".to_string(),
            ));
        }
        if *self.indptr.last().unwrap_or(&0) != self.data.len() {
            return Err(RuPcaError::InvalidInput(
                "CSR indptr must end at nnz".to_string(),
            ));
        }
        for window in self.indptr.windows(2) {
            if window[0] > window[1] {
                return Err(RuPcaError::InvalidInput(
                    "CSR indptr must be nondecreasing".to_string(),
                ));
            }
        }
        if self.indices.iter().any(|&j| j >= self.n_cols) {
            return Err(RuPcaError::InvalidInput(
                "CSR column index out of bounds".to_string(),
            ));
        }
        Ok(())
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n_cols);
        debug_assert_eq!(y.len(), self.n_rows);
        y.fill(0.0);
        for i in 0..self.n_rows {
            let mut acc = 0.0;
            for p in self.indptr[i]..self.indptr[i + 1] {
                acc += self.data[p] * x[self.indices[p]];
            }
            y[i] = acc;
        }
    }

    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n_rows);
        debug_assert_eq!(y.len(), self.n_cols);
        y.fill(0.0);
        for i in 0..self.n_rows {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            for p in self.indptr[i]..self.indptr[i + 1] {
                y[self.indices[p]] += self.data[p] * xi;
            }
        }
    }
}

/// Dense row-major matrix input for direct centered dense SVD.
///
/// This is intended for ordinary dense matrices. PFlogPF / shifted-CLR input
/// should use `ShiftedClrCsrMatrix` so the shifted dense values are not
/// materialized.
#[derive(Debug, Clone)]
pub struct DenseMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: Vec<f64>,
}

impl DenseMatrix {
    pub fn validate(&self) -> Result<()> {
        if self.data.len() != self.n_rows * self.n_cols {
            return Err(RuPcaError::InvalidInput(
                "dense data length must equal n_rows * n_cols".to_string(),
            ));
        }
        Ok(())
    }
}

trait PcaInputMatrix {
    fn validate(&self) -> Result<()>;
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn mean_variance_axis0(&self) -> (Vec<f64>, Vec<f64>);
    fn matvec(&self, x: &[f64], y: &mut [f64]);
    fn rmatvec(&self, x: &[f64], y: &mut [f64]);

    fn centered_normal_right<'a>(
        &'a self,
        mean: &'a [f64],
    ) -> Box<dyn FnMut(&[f64], &mut [f64]) + 'a>
    where
        Self: Sized,
    {
        let centered = ImplicitColumnOffset { x: self, mean };
        let mut tmp = vec![0.0; self.n_rows()];
        Box::new(move |vin, vout| {
            centered.matvec(vin, &mut tmp);
            centered.rmatvec(&tmp, vout);
        })
    }

    fn centered_normal_left<'a>(
        &'a self,
        mean: &'a [f64],
    ) -> Box<dyn FnMut(&[f64], &mut [f64]) + 'a>
    where
        Self: Sized,
    {
        let centered = ImplicitColumnOffset { x: self, mean };
        let mut tmp = vec![0.0; self.n_cols()];
        Box::new(move |vin, vout| {
            centered.rmatvec(vin, &mut tmp);
            centered.matvec(&tmp, vout);
        })
    }
}

impl PcaInputMatrix for CsrMatrix {
    fn validate(&self) -> Result<()> {
        CsrMatrix::validate(self)
    }

    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }

    fn mean_variance_axis0(&self) -> (Vec<f64>, Vec<f64>) {
        mean_variance_axis0(self)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        CsrMatrix::matvec(self, x, y)
    }

    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        CsrMatrix::rmatvec(self, x, y)
    }

    fn centered_normal_right<'a>(
        &'a self,
        mean: &'a [f64],
    ) -> Box<dyn FnMut(&[f64], &mut [f64]) + 'a> {
        let mut tmp = vec![0.0; self.n_rows];
        let n_rows_f = self.n_rows as f64;
        Box::new(move |vin, vout| {
            self.matvec(vin, &mut tmp);
            self.rmatvec(&tmp, vout);
            let alpha = n_rows_f * dot(mean, vin);
            for (y, m) in vout.iter_mut().zip(mean.iter()) {
                *y -= alpha * m;
            }
        })
    }
}

/// Dense shifted-CLR-style data represented as a sparse matrix plus a row centering vector.
///
/// The represented dense value is `sparse[i, j] - row_center[i]`. For a common
/// shifted CLR workflow, `sparse` would contain the nonzero shifted-log values
/// and `row_center[i]` would be the row mean subtracted from every feature.
#[derive(Debug, Clone)]
pub struct ShiftedClrCsrMatrix {
    pub sparse: CsrMatrix,
    pub row_center: Vec<f64>,
}

impl ShiftedClrCsrMatrix {
    pub fn validate(&self) -> Result<()> {
        self.sparse.validate()?;
        if self.row_center.len() != self.sparse.n_rows {
            return Err(RuPcaError::InvalidInput(
                "row_center length must equal sparse.n_rows".to_string(),
            ));
        }
        Ok(())
    }
}

impl PcaInputMatrix for ShiftedClrCsrMatrix {
    fn validate(&self) -> Result<()> {
        ShiftedClrCsrMatrix::validate(self)
    }

    fn n_rows(&self) -> usize {
        self.sparse.n_rows
    }

    fn n_cols(&self) -> usize {
        self.sparse.n_cols
    }

    fn mean_variance_axis0(&self) -> (Vec<f64>, Vec<f64>) {
        let mut sums = vec![0.0; self.sparse.n_cols];
        let row_center_sum = self.row_center.iter().sum::<f64>();
        let row_center_sq_sum = self.row_center.iter().map(|v| v * v).sum::<f64>();
        let mut sq_sums = vec![row_center_sq_sum; self.sparse.n_cols];
        for i in 0..self.sparse.n_rows {
            let row_center = self.row_center[i];
            for p in self.sparse.indptr[i]..self.sparse.indptr[i + 1] {
                let j = self.sparse.indices[p];
                let v = self.sparse.data[p];
                sums[j] += v;
                sq_sums[j] += v * v - 2.0 * v * row_center;
            }
        }
        let n = self.sparse.n_rows as f64;
        let mean = sums
            .iter()
            .map(|s| (*s - row_center_sum) / n)
            .collect::<Vec<_>>();
        let var = sq_sums
            .iter()
            .zip(mean.iter())
            .map(|(ss, m)| (ss / n) - m * m)
            .collect::<Vec<_>>();
        (mean, var)
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        self.sparse.matvec(x, y);
        let s = x.iter().sum::<f64>();
        for (out, center) in y.iter_mut().zip(self.row_center.iter()) {
            *out -= center * s;
        }
    }

    fn rmatvec(&self, x: &[f64], y: &mut [f64]) {
        self.sparse.rmatvec(x, y);
        let off = dot(&self.row_center, x);
        for out in y.iter_mut() {
            *out -= off;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScanpyPcaParams {
    pub n_components: usize,
    pub tol: f64,
    pub ncv: Option<usize>,
    pub maxiter: Option<usize>,
    pub seed: u64,
}

impl Default for ScanpyPcaParams {
    fn default() -> Self {
        Self {
            n_components: 50,
            tol: 0.0,
            ncv: None,
            maxiter: None,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScanpyPcaResult {
    pub scores: Vec<f64>,
    pub components: Vec<f64>,
    pub mean: Vec<f64>,
    pub explained_variance: Vec<f64>,
    pub explained_variance_ratio: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub noise_variance: f64,
    pub n_samples: usize,
    pub n_features: usize,
    pub n_components: usize,
    pub warnings: Vec<String>,
}

pub fn pca_scanpy_sparse_csr(x: &CsrMatrix, params: ScanpyPcaParams) -> Result<ScanpyPcaResult> {
    pca_column_centered(x, params)
}

pub fn pca_scanpy_dense(x: &DenseMatrix, params: ScanpyPcaParams) -> Result<ScanpyPcaResult> {
    pca_dense_centered_svd(x, params)
}

pub fn pca_shifted_clr_sparse_csr(
    x: &ShiftedClrCsrMatrix,
    params: ScanpyPcaParams,
) -> Result<ScanpyPcaResult> {
    pca_column_centered(x, params)
}

fn pca_column_centered<M: PcaInputMatrix>(
    x: &M,
    params: ScanpyPcaParams,
) -> Result<ScanpyPcaResult> {
    x.validate()?;
    let n_samples = x.n_rows();
    let n_features = x.n_cols();
    if n_samples == 0 || n_features == 0 {
        return Err(RuPcaError::InvalidInput("empty matrix".to_string()));
    }
    let k = params.n_components;
    let min_dim = n_samples.min(n_features);
    if !(1..min_dim).contains(&k) {
        return Err(RuPcaError::InvalidInput(format!(
            "n_components must be in [1, {}), got {}",
            min_dim, k
        )));
    }

    let (mean, var) = x.mean_variance_axis0();
    let total_var = if n_samples > 1 {
        var.iter().sum::<f64>() * n_samples as f64 / (n_samples as f64 - 1.0)
    } else {
        0.0
    };

    let transpose = n_samples < n_features;
    let normal_dim = min_dim;
    let profile = std::env::var_os("RUPCA_PROFILE").is_some();
    let ncv = params.ncv.unwrap_or_else(|| choose_ncv(k)).min(normal_dim);
    let maxiter = params.maxiter.unwrap_or(normal_dim * 10);
    let v0 = init_arpack_v0(normal_dim, params.seed);

    let eigsh_start = Instant::now();
    let mut op_calls = 0usize;
    let eigvec = {
        let mut op = if !transpose {
            x.centered_normal_right(&mean)
        } else {
            x.centered_normal_left(&mean)
        };
        eigsh(
            normal_dim,
            k,
            ncv,
            maxiter,
            params.tol * params.tol,
            &v0,
            |vin, vout| {
                op_calls += 1;
                op(vin, vout);
            },
        )?
    };
    let eigsh_secs = eigsh_start.elapsed().as_secs_f64();

    let av_start = Instant::now();
    let av = {
        let centered = ImplicitColumnOffset { x, mean: &mean };
        if !transpose {
            centered.matmat(&eigvec)
        } else {
            centered.rmatmat(&eigvec)
        }
    };
    let av_secs = av_start.elapsed().as_secs_f64();

    let svd_start = Instant::now();
    let svd = av.svd(true, true);
    let mut u = svd.u.ok_or(RuPcaError::MissingSvdFactors)?;
    let s = svd.singular_values.as_slice().to_vec();
    let mut vt = svd.v_t.ok_or(RuPcaError::MissingSvdFactors)?;
    let svd_secs = svd_start.elapsed().as_secs_f64();

    svd_flip_u_based_false(&mut u, &mut vt);

    let (u_final, vt_final) = if transpose {
        let u_tmp = &eigvec * vt.transpose();
        (u_tmp, u.transpose())
    } else {
        (u, &vt * eigvec.transpose())
    };

    let n_keep = k.min(s.len());
    let mut scores = vec![0.0; n_samples * n_keep];
    for i in 0..n_samples {
        for j in 0..n_keep {
            scores[i * n_keep + j] = u_final[(i, j)] * s[j];
        }
    }

    let components_mat = vt_final.rows(0, n_keep).into_owned();
    let mut components = vec![0.0; n_keep * n_features];
    for i in 0..n_keep {
        for j in 0..n_features {
            components[i * n_features + j] = components_mat[(i, j)];
        }
    }

    let explained_variance = s
        .iter()
        .take(n_keep)
        .map(|v| (v * v) / (n_samples as f64 - 1.0))
        .collect::<Vec<_>>();
    let explained_variance_ratio = if total_var > 0.0 {
        explained_variance
            .iter()
            .map(|v| *v / total_var)
            .collect::<Vec<_>>()
    } else {
        vec![0.0; n_keep]
    };
    let noise_variance = if n_keep < min_dim {
        let residual = total_var - explained_variance.iter().sum::<f64>();
        residual / (min_dim - n_keep) as f64
    } else {
        0.0
    };

    if profile {
        eprintln!(
            "rupca_profile transpose={} ncv={} maxiter={} op_calls={} eigsh_seconds={:.6} av_seconds={:.6} svd_seconds={:.6}",
            transpose, ncv, maxiter, op_calls, eigsh_secs, av_secs, svd_secs
        );
    }

    Ok(ScanpyPcaResult {
        scores,
        components,
        mean,
        explained_variance,
        explained_variance_ratio,
        singular_values: s.into_iter().take(n_keep).collect(),
        noise_variance,
        n_samples,
        n_features,
        n_components: n_keep,
        warnings: Vec::new(),
    })
}

fn pca_dense_centered_svd(x: &DenseMatrix, params: ScanpyPcaParams) -> Result<ScanpyPcaResult> {
    x.validate()?;
    let n_samples = x.n_rows;
    let n_features = x.n_cols;
    if n_samples == 0 || n_features == 0 {
        return Err(RuPcaError::InvalidInput("empty matrix".to_string()));
    }
    let k = params.n_components;
    let min_dim = n_samples.min(n_features);
    if !(1..=min_dim).contains(&k) {
        return Err(RuPcaError::InvalidInput(format!(
            "n_components must be in [1, {}], got {}",
            min_dim, k
        )));
    }

    let mut centered = DMatrix::from_row_slice(n_samples, n_features, &x.data);
    let mut mean = vec![0.0; n_features];
    let mut var = vec![0.0; n_features];
    for j in 0..n_features {
        for i in 0..n_samples {
            mean[j] += centered[(i, j)];
            var[j] += centered[(i, j)] * centered[(i, j)];
        }
        mean[j] /= n_samples as f64;
        var[j] = var[j] / n_samples as f64 - mean[j] * mean[j];
        for i in 0..n_samples {
            centered[(i, j)] -= mean[j];
        }
    }
    let total_var = if n_samples > 1 {
        var.iter().sum::<f64>() * n_samples as f64 / (n_samples as f64 - 1.0)
    } else {
        0.0
    };

    let svd = centered.svd(true, true);
    let mut u = svd.u.ok_or(RuPcaError::MissingSvdFactors)?;
    let s = svd.singular_values.as_slice().to_vec();
    let mut vt = svd.v_t.ok_or(RuPcaError::MissingSvdFactors)?;
    svd_flip_u_based_false(&mut u, &mut vt);

    let n_keep = k.min(s.len());
    let mut scores = vec![0.0; n_samples * n_keep];
    for i in 0..n_samples {
        for j in 0..n_keep {
            scores[i * n_keep + j] = u[(i, j)] * s[j];
        }
    }

    let mut components = vec![0.0; n_keep * n_features];
    for i in 0..n_keep {
        for j in 0..n_features {
            components[i * n_features + j] = vt[(i, j)];
        }
    }

    let explained_variance = s
        .iter()
        .take(n_keep)
        .map(|v| (v * v) / (n_samples as f64 - 1.0))
        .collect::<Vec<_>>();
    let explained_variance_ratio = if total_var > 0.0 {
        explained_variance
            .iter()
            .map(|v| *v / total_var)
            .collect::<Vec<_>>()
    } else {
        vec![0.0; n_keep]
    };
    let noise_variance = if n_keep < min_dim {
        let residual = total_var - explained_variance.iter().sum::<f64>();
        residual / (min_dim - n_keep) as f64
    } else {
        0.0
    };

    Ok(ScanpyPcaResult {
        scores,
        components,
        mean,
        explained_variance,
        explained_variance_ratio,
        singular_values: s.into_iter().take(n_keep).collect(),
        noise_variance,
        n_samples,
        n_features,
        n_components: n_keep,
        warnings: vec![
            "input matrix is dense; using direct centered dense SVD. PFlogPF / shifted-CLR input should be supplied as ShiftedClrCsrMatrix rather than a dense matrix".to_string(),
        ],
    })
}

fn mean_variance_axis0(x: &CsrMatrix) -> (Vec<f64>, Vec<f64>) {
    let mut sums = vec![0.0; x.n_cols];
    let mut sq_sums = vec![0.0; x.n_cols];
    for i in 0..x.n_rows {
        for p in x.indptr[i]..x.indptr[i + 1] {
            let j = x.indices[p];
            let v = x.data[p];
            sums[j] += v;
            sq_sums[j] += v * v;
        }
    }
    let n = x.n_rows as f64;
    let mean = sums.iter().map(|s| *s / n).collect::<Vec<_>>();
    let var = sq_sums
        .iter()
        .zip(mean.iter())
        .map(|(ss, m)| (ss / n) - m * m)
        .collect::<Vec<_>>();
    (mean, var)
}

struct ImplicitColumnOffset<'a, M: PcaInputMatrix> {
    x: &'a M,
    mean: &'a [f64],
}

impl<M: PcaInputMatrix> ImplicitColumnOffset<'_, M> {
    fn matvec(&self, v: &[f64], out: &mut [f64]) {
        self.x.matvec(v, out);
        let off = dot(self.mean, v);
        for y in out.iter_mut() {
            *y -= off;
        }
    }

    fn rmatvec(&self, v: &[f64], out: &mut [f64]) {
        self.x.rmatvec(v, out);
        let s = v.iter().sum::<f64>();
        for (y, m) in out.iter_mut().zip(self.mean.iter()) {
            *y -= m * s;
        }
    }

    fn matmat(&self, m: &DMatrix<f64>) -> DMatrix<f64> {
        let mut out = DMatrix::zeros(self.x.n_rows(), m.ncols());
        let mut tmp = vec![0.0; self.x.n_rows()];
        for col in 0..m.ncols() {
            self.matvec(m.column(col).as_slice(), &mut tmp);
            for row in 0..self.x.n_rows() {
                out[(row, col)] = tmp[row];
            }
        }
        out
    }

    fn rmatmat(&self, m: &DMatrix<f64>) -> DMatrix<f64> {
        let mut out = DMatrix::zeros(self.x.n_cols(), m.ncols());
        let mut tmp = vec![0.0; self.x.n_cols()];
        for col in 0..m.ncols() {
            self.rmatvec(m.column(col).as_slice(), &mut tmp);
            for row in 0..self.x.n_cols() {
                out[(row, col)] = tmp[row];
            }
        }
        out
    }
}

fn choose_ncv(k: usize) -> usize {
    (3 * k + 1).max(20)
}

fn init_arpack_v0(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

fn svd_flip_u_based_false(u: &mut DMatrix<f64>, v: &mut DMatrix<f64>) {
    for i in 0..v.nrows() {
        let mut max_j = 0usize;
        let mut max_abs = 0.0f64;
        for j in 0..v.ncols() {
            let a = v[(i, j)].abs();
            if a > max_abs {
                max_abs = a;
                max_j = j;
            }
        }
        let sign = if v[(i, max_j)] < 0.0 { -1.0 } else { 1.0 };
        for r in 0..u.nrows() {
            u[(r, i)] *= sign;
        }
        for c in 0..v.ncols() {
            v[(i, c)] *= sign;
        }
    }
}

fn eigsh<F>(
    n: usize,
    k: usize,
    ncv: usize,
    maxiter: usize,
    tol: f64,
    v0: &[f64],
    op: F,
) -> Result<DMatrix<f64>>
where
    F: FnMut(&[f64], &mut [f64]),
{
    if k >= n {
        return Err(RuPcaError::InvalidInput(
            "ARPACK requires k < n".to_string(),
        ));
    }
    if !(k < ncv && ncv <= n) {
        return Err(RuPcaError::InvalidInput(
            "solver requires k < ncv <= n".to_string(),
        ));
    }
    let mut resid = v0.to_vec();
    let mut rnorm = norm(&resid);
    let mut v = DMatrix::zeros(n, ncv);
    let mut h = DMatrix::zeros(ncv, 2);
    let result = dsaup2_mode1(
        "LM",
        k,
        ncv - k,
        tol.max(1e-10),
        maxiter,
        &mut resid,
        &mut rnorm,
        &mut v,
        &mut h,
        op,
    )?;
    if std::env::var_os("RUPCA_PROFILE").is_some() {
        eprintln!(
            "rupca_eigsh info={} basis_dim={} nconv={} nev={} np={} rnorm={:.6} first_ritz={:?} first_bounds={:?}",
            result.info,
            result.basis_dim,
            result.nconv,
            result.nev,
            result.np,
            result.rnorm,
            &result.ritz[..result.ritz.len().min(5)],
            &result.bounds[..result.bounds.len().min(5)],
        );
    }
    if result.info == -9999 && result.basis_dim < k {
        return Err(RuPcaError::Solver(format!(
            "insufficient invariant subspace: built {}, need {}",
            result.basis_dim, k
        )));
    }
    // Robustness: if the eigensolver converged no Ritz values, the extracted basis would be empty
    // and the downstream dense SVD would panic. Return a clean error instead.
    if result.nconv == 0 {
        return Err(RuPcaError::Solver(format!(
            "eigensolver converged 0 of {k} requested eigenpairs (did not converge); \
             try a larger ncv, fewer components, or check the input"
        )));
    }
    let basis_dim = result.basis_dim.min(ncv);
    extract_selected_ritz_vectors(&v, &h, basis_dim, &result.ritz[..result.nconv], k)
}

fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn orthonormalize_columns(m: &mut DMatrix<f64>) -> Result<()> {
    for col in 0..m.ncols() {
        let mut col_norm = project_and_norm(m, col);
        if col_norm <= 1e-14 {
            let mut repaired = false;
            for basis_row in 0..m.nrows() {
                for row in 0..m.nrows() {
                    m[(row, col)] = if row == basis_row { 1.0 } else { 0.0 };
                }
                col_norm = project_and_norm(m, col);
                if col_norm > 1e-14 {
                    repaired = true;
                    break;
                }
            }
            if !repaired {
                return Err(RuPcaError::Solver(
                    "orthonormalization produced a numerically zero column".to_string(),
                ));
            }
        }
        for row in 0..m.nrows() {
            m[(row, col)] /= col_norm;
        }
    }
    Ok(())
}

fn project_and_norm(m: &mut DMatrix<f64>, col: usize) -> f64 {
    for _ in 0..2 {
        for prev in 0..col {
            let mut coeff = 0.0;
            for row in 0..m.nrows() {
                coeff += m[(row, prev)] * m[(row, col)];
            }
            for row in 0..m.nrows() {
                m[(row, col)] -= coeff * m[(row, prev)];
            }
        }
    }
    let mut col_norm = 0.0;
    for row in 0..m.nrows() {
        col_norm += m[(row, col)] * m[(row, col)];
    }
    col_norm.sqrt()
}

fn apply_operator_matrix<F>(n: usize, q: &DMatrix<f64>, op: &mut F) -> DMatrix<f64>
where
    F: FnMut(&[f64], &mut [f64]),
{
    let mut out = DMatrix::zeros(n, q.ncols());
    let mut tmp = vec![0.0; n];
    for col in 0..q.ncols() {
        op(q.column(col).as_slice(), &mut tmp);
        for row in 0..n {
            out[(row, col)] = tmp[row];
        }
    }
    out
}

fn extract_selected_ritz_vectors(
    basis: &DMatrix<f64>,
    h: &DMatrix<f64>,
    m: usize,
    selected_ritz: &[f64],
    k: usize,
) -> Result<DMatrix<f64>> {
    if m == 0 || basis.ncols() < m || h.nrows() < m || h.ncols() != 2 {
        return Err(RuPcaError::InvalidInput(
            "invalid Lanczos factorization for Ritz extraction".to_string(),
        ));
    }
    let mut t = DMatrix::zeros(m, m);
    for i in 0..m {
        t[(i, i)] = h[(i, 1)];
        if i > 0 {
            t[(i, i - 1)] = h[(i, 0)];
            t[(i - 1, i)] = h[(i, 0)];
        }
    }
    let eig = SymmetricEigen::new(t);
    let keep = k.min(selected_ritz.len());
    let mut y = DMatrix::zeros(m, keep);
    let mut used = vec![false; m];
    for (out_col, &target) in selected_ritz.iter().take(keep).enumerate() {
        let mut best_idx = None;
        let mut best_err = f64::INFINITY;
        for idx in 0..m {
            if used[idx] {
                continue;
            }
            let err = (eig.eigenvalues[idx] - target).abs();
            if err < best_err {
                best_err = err;
                best_idx = Some(idx);
            }
        }
        let idx = best_idx.ok_or_else(|| {
            RuPcaError::Solver(
                "failed to match selected Ritz values to tridiagonal eigenpairs".to_string(),
            )
        })?;
        used[idx] = true;
        for row in 0..m {
            y[(row, out_col)] = eig.eigenvectors[(row, idx)];
        }
    }
    let mut z = basis.columns(0, m).into_owned() * y;
    orthonormalize_columns(&mut z)?;
    Ok(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arpack_symm::aitr::dsaitr_mode1;
    use crate::arpack_symm::apps::dsapps;
    use crate::arpack_symm::aup2::dsaup2_mode1;
    use crate::arpack_symm::conv::dsconv;
    use crate::arpack_symm::sort::{dsesrt, dsgets, dsortr};
    use crate::arpack_symm::tridiag::dseigt;
    use serde_json::Value;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::{Mutex, OnceLock};

    fn arpack_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn lock_solver() -> std::sync::MutexGuard<'static, ()> {
        arpack_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn test_csr(n_rows: usize, n_cols: usize, entries: &[(usize, usize, f64)]) -> CsrMatrix {
        let mut rows = vec![Vec::<(usize, f64)>::new(); n_rows];
        for &(i, j, v) in entries {
            rows[i].push((j, v));
        }
        for row in &mut rows {
            row.sort_by_key(|(j, _)| *j);
        }
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = Vec::with_capacity(n_rows + 1);
        indptr.push(0);
        for row in rows {
            for (j, v) in row {
                indices.push(j);
                data.push(v);
            }
            indptr.push(data.len());
        }
        CsrMatrix {
            n_rows,
            n_cols,
            data,
            indices,
            indptr,
        }
    }

    fn centered_dense(x: &CsrMatrix) -> DMatrix<f64> {
        let mut dense = DMatrix::zeros(x.n_rows, x.n_cols);
        for i in 0..x.n_rows {
            for p in x.indptr[i]..x.indptr[i + 1] {
                dense[(i, x.indices[p])] = x.data[p];
            }
        }
        for j in 0..x.n_cols {
            let mean = dense.column(j).iter().sum::<f64>() / x.n_rows as f64;
            for i in 0..x.n_rows {
                dense[(i, j)] -= mean;
            }
        }
        dense
    }

    fn dense_reference(
        x: &CsrMatrix,
        n_components: usize,
    ) -> (Vec<f64>, DMatrix<f64>, DMatrix<f64>) {
        let centered = centered_dense(x);
        let svd = centered.svd(true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let s = svd.singular_values.as_slice()[..n_components].to_vec();
        let mut scores = DMatrix::zeros(x.n_rows, n_components);
        for i in 0..x.n_rows {
            for j in 0..n_components {
                scores[(i, j)] = u[(i, j)] * s[j];
            }
        }
        (s, scores, vt.rows(0, n_components).into_owned())
    }

    fn dense_matrix_from_csr(x: &CsrMatrix) -> DenseMatrix {
        let mut data = vec![0.0; x.n_rows * x.n_cols];
        for i in 0..x.n_rows {
            for p in x.indptr[i]..x.indptr[i + 1] {
                data[i * x.n_cols + x.indices[p]] = x.data[p];
            }
        }
        DenseMatrix {
            n_rows: x.n_rows,
            n_cols: x.n_cols,
            data,
        }
    }

    fn shifted_clr_from_sparse(sparse: CsrMatrix) -> ShiftedClrCsrMatrix {
        let mut row_center = vec![0.0; sparse.n_rows];
        for i in 0..sparse.n_rows {
            let mut row_sum = 0.0;
            for p in sparse.indptr[i]..sparse.indptr[i + 1] {
                row_sum += sparse.data[p];
            }
            row_center[i] = row_sum / sparse.n_cols as f64;
        }
        ShiftedClrCsrMatrix { sparse, row_center }
    }

    fn shifted_clr_dense(x: &ShiftedClrCsrMatrix) -> DMatrix<f64> {
        let mut dense = DMatrix::zeros(x.sparse.n_rows, x.sparse.n_cols);
        for i in 0..x.sparse.n_rows {
            for j in 0..x.sparse.n_cols {
                dense[(i, j)] = -x.row_center[i];
            }
            for p in x.sparse.indptr[i]..x.sparse.indptr[i + 1] {
                dense[(i, x.sparse.indices[p])] += x.sparse.data[p];
            }
        }
        dense
    }

    fn shifted_clr_centered_dense(x: &ShiftedClrCsrMatrix) -> DMatrix<f64> {
        let mut dense = shifted_clr_dense(x);
        for j in 0..x.sparse.n_cols {
            let mean = dense.column(j).iter().sum::<f64>() / x.sparse.n_rows as f64;
            for i in 0..x.sparse.n_rows {
                dense[(i, j)] -= mean;
            }
        }
        dense
    }

    fn shifted_clr_dense_reference(
        x: &ShiftedClrCsrMatrix,
        n_components: usize,
    ) -> (Vec<f64>, DMatrix<f64>, DMatrix<f64>) {
        let centered = shifted_clr_centered_dense(x);
        let svd = centered.svd(true, true);
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let s = svd.singular_values.as_slice()[..n_components].to_vec();
        let mut scores = DMatrix::zeros(x.sparse.n_rows, n_components);
        for i in 0..x.sparse.n_rows {
            for j in 0..n_components {
                scores[(i, j)] = u[(i, j)] * s[j];
            }
        }
        (s, scores, vt.rows(0, n_components).into_owned())
    }

    fn assert_all_close(a: &[f64], b: &[f64], tol: f64) {
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(
            max_abs <= tol,
            "max abs difference {max_abs} exceeded tolerance {tol}"
        );
    }

    fn abs_dot(a: &[f64], b: &[f64]) -> f64 {
        dot(a, b).abs()
    }

    fn scanpy_python() -> PathBuf {
        std::env::var_os("SCANPY_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/Users/lpachter/Dropbox/claude/projects/scanpy-env/bin/python")
            })
    }

    fn compare_with_scanpy(x: &CsrMatrix, k: usize, seed: u64) {
        let rust = pca_scanpy_sparse_csr(
            x,
            ScanpyPcaParams {
                n_components: k,
                tol: 0.0,
                ncv: None,
                maxiter: None,
                seed,
            },
        )
        .unwrap();

        let dir = std::env::temp_dir().join(format!(
            "rupca_scanpy_parity_{}_{}_{}",
            std::process::id(),
            x.n_rows,
            x.n_cols
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let matrix_path = dir.join("matrix.tsv");
        let script_path = dir.join("check.py");
        let mut matrix = String::new();
        matrix.push_str(&format!("{} {}\n", x.n_rows, x.n_cols));
        for i in 0..x.n_rows {
            for p in x.indptr[i]..x.indptr[i + 1] {
                matrix.push_str(&format!("{}\t{}\t{:.17}\n", i, x.indices[p], x.data[p]));
            }
        }
        fs::write(&matrix_path, matrix).unwrap();
        fs::write(
            &script_path,
            r#"
import json, sys
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

matrix_path = sys.argv[1]
k = int(sys.argv[2])
seed = int(sys.argv[3])
with open(matrix_path) as fh:
    header = fh.readline().strip().split()
    n_rows, n_cols = map(int, header)
    rows = []
    cols = []
    data = []
    for line in fh:
        i, j, v = line.rstrip().split('\t')
        rows.append(int(i))
        cols.append(int(j))
        data.append(float(v))
X = csr_matrix((np.array(data, dtype=np.float64), (np.array(rows), np.array(cols))), shape=(n_rows, n_cols))
pca = PCA(n_components=k, svd_solver='arpack', random_state=seed)
scores = pca.fit_transform(X)
out = {
    'scores': scores.tolist(),
    'components': pca.components_.tolist(),
    'mean': pca.mean_.tolist(),
    'explained_variance': pca.explained_variance_.tolist(),
    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
    'singular_values': pca.singular_values_.tolist(),
    'noise_variance': float(pca.noise_variance_),
}
print(json.dumps(out))
"#,
        )
        .unwrap();
        let output = Command::new(scanpy_python())
            .arg(&script_path)
            .arg(&matrix_path)
            .arg(k.to_string())
            .arg(seed.to_string())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "python failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let py: Value = serde_json::from_slice(&output.stdout).unwrap();

        let py_s = py["singular_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let py_ev = py["explained_variance"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let py_evr = py["explained_variance_ratio"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let py_mean = py["mean"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let py_scores = py["scores"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let py_components = py["components"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let py_noise = py["noise_variance"].as_f64().unwrap();

        for (a, b) in rust.mean.iter().zip(py_mean.iter()) {
            assert!((a - b).abs() < 1e-12, "mean mismatch: {a} vs {b}");
        }
        for (a, b) in rust.singular_values.iter().zip(py_s.iter()) {
            assert!((a - b).abs() < 1e-8, "singular value mismatch: {a} vs {b}");
        }
        for (a, b) in rust.explained_variance.iter().zip(py_ev.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "explained variance mismatch: {a} vs {b}"
            );
        }
        for (a, b) in rust.explained_variance_ratio.iter().zip(py_evr.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "explained variance ratio mismatch: {a} vs {b}"
            );
        }
        assert!((rust.noise_variance - py_noise).abs() < 1e-10);

        for comp in 0..k {
            let rust_score_col = (0..x.n_rows)
                .map(|i| rust.scores[i * k + comp])
                .collect::<Vec<_>>();
            let py_score_col = (0..x.n_rows)
                .map(|i| py_scores[i * k + comp])
                .collect::<Vec<_>>();
            let rust_comp_row = (0..x.n_cols)
                .map(|j| rust.components[comp * x.n_cols + j])
                .collect::<Vec<_>>();
            let py_comp_row = (0..x.n_cols)
                .map(|j| py_components[comp * x.n_cols + j])
                .collect::<Vec<_>>();
            let sign = if dot(&rust_comp_row, &py_comp_row) < 0.0 {
                -1.0
            } else {
                1.0
            };
            for (a, b) in rust_score_col.iter().zip(py_score_col.iter()) {
                assert!(
                    (a - sign * b).abs() < 1e-8,
                    "score mismatch: {a} vs {}",
                    sign * b
                );
            }
            for (a, b) in rust_comp_row.iter().zip(py_comp_row.iter()) {
                assert!(
                    (a - sign * b).abs() < 1e-8,
                    "component mismatch: {a} vs {}",
                    sign * b
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_csr_inputs() {
        let data_indices_mismatch = CsrMatrix {
            n_rows: 1,
            n_cols: 2,
            data: vec![1.0],
            indices: vec![],
            indptr: vec![0, 1],
        };
        assert!(matches!(
            data_indices_mismatch.validate(),
            Err(RuPcaError::InvalidInput(msg))
                if msg.contains("data and indices length mismatch")
        ));

        let wrong_indptr_len = CsrMatrix {
            n_rows: 2,
            n_cols: 2,
            data: vec![1.0],
            indices: vec![0],
            indptr: vec![0, 1],
        };
        assert!(matches!(
            wrong_indptr_len.validate(),
            Err(RuPcaError::InvalidInput(msg))
                if msg.contains("indptr length must be n_rows + 1")
        ));

        let indptr_not_starting_at_zero = CsrMatrix {
            n_rows: 1,
            n_cols: 2,
            data: vec![1.0],
            indices: vec![0],
            indptr: vec![1, 1],
        };
        assert!(matches!(
            indptr_not_starting_at_zero.validate(),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("indptr must start at 0")
        ));

        let indptr_not_ending_at_nnz = CsrMatrix {
            n_rows: 1,
            n_cols: 2,
            data: vec![1.0],
            indices: vec![0],
            indptr: vec![0, 0],
        };
        assert!(matches!(
            indptr_not_ending_at_nnz.validate(),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("indptr must end at nnz")
        ));

        let nondecreasing_indptr = CsrMatrix {
            n_rows: 2,
            n_cols: 2,
            data: vec![1.0],
            indices: vec![0],
            indptr: vec![0, 2, 1],
        };
        assert!(matches!(
            nondecreasing_indptr.validate(),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("indptr must be nondecreasing")
        ));

        let out_of_bounds_column = CsrMatrix {
            n_rows: 1,
            n_cols: 2,
            data: vec![1.0],
            indices: vec![2],
            indptr: vec![0, 1],
        };
        assert!(matches!(
            out_of_bounds_column.validate(),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("column index out of bounds")
        ));
    }

    #[test]
    fn rejects_invalid_shifted_clr_row_center_length() {
        let x = ShiftedClrCsrMatrix {
            sparse: test_csr(2, 3, &[(0, 0, 1.0), (1, 2, 2.0)]),
            row_center: vec![0.5],
        };
        assert!(matches!(
            x.validate(),
            Err(RuPcaError::InvalidInput(msg))
                if msg.contains("row_center length must equal sparse.n_rows")
        ));
    }

    #[test]
    fn rejects_invalid_dense_inputs() {
        let bad_shape = DenseMatrix {
            n_rows: 2,
            n_cols: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        assert!(matches!(
            bad_shape.validate(),
            Err(RuPcaError::InvalidInput(msg))
                if msg.contains("dense data length must equal n_rows * n_cols")
        ));

        let empty = DenseMatrix {
            n_rows: 0,
            n_cols: 3,
            data: vec![],
        };
        assert!(matches!(
            pca_scanpy_dense(&empty, ScanpyPcaParams::default()),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("empty matrix")
        ));

        let x = DenseMatrix {
            n_rows: 3,
            n_cols: 2,
            data: vec![1.0, 0.0, 0.0, 2.0, 3.0, 1.0],
        };
        assert!(matches!(
            pca_scanpy_dense(
                &x,
                ScanpyPcaParams {
                    n_components: 0,
                    ..ScanpyPcaParams::default()
                }
            ),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("n_components")
        ));
        assert!(matches!(
            pca_scanpy_dense(
                &x,
                ScanpyPcaParams {
                    n_components: 3,
                    ..ScanpyPcaParams::default()
                }
            ),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("n_components")
        ));
    }

    #[test]
    fn rejects_invalid_public_pca_inputs() {
        let empty_rows = CsrMatrix {
            n_rows: 0,
            n_cols: 3,
            data: vec![],
            indices: vec![],
            indptr: vec![0],
        };
        assert!(matches!(
            pca_scanpy_sparse_csr(&empty_rows, ScanpyPcaParams::default()),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("empty matrix")
        ));

        let x = test_csr(
            4,
            3,
            &[
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 3.0),
                (3, 0, 4.0),
                (3, 2, 1.0),
            ],
        );
        assert!(matches!(
            pca_scanpy_sparse_csr(
                &x,
                ScanpyPcaParams {
                    n_components: 0,
                    ..ScanpyPcaParams::default()
                }
            ),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("n_components")
        ));
        assert!(matches!(
            pca_scanpy_sparse_csr(
                &x,
                ScanpyPcaParams {
                    n_components: 3,
                    ..ScanpyPcaParams::default()
                }
            ),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("n_components")
        ));
        assert!(matches!(
            pca_scanpy_sparse_csr(
                &x,
                ScanpyPcaParams {
                    n_components: 2,
                    ncv: Some(2),
                    ..ScanpyPcaParams::default()
                }
            ),
            Err(RuPcaError::InvalidInput(msg)) if msg.contains("k < ncv <= n")
        ));
    }

    #[test]
    fn matches_dense_svd_on_tall_sparse_matrix() {
        let x = test_csr(
            6,
            4,
            &[
                (0, 0, 2.0),
                (0, 2, 1.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 3, 4.0),
                (3, 1, 2.0),
                (3, 3, 1.0),
                (4, 0, 3.0),
                (4, 2, 2.0),
                (5, 1, 1.0),
                (5, 3, 2.0),
            ],
        );
        let k = 2;
        let _guard = lock_solver();
        let got = pca_scanpy_sparse_csr(
            &x,
            ScanpyPcaParams {
                n_components: k,
                tol: 0.0,
                ncv: None,
                maxiter: None,
                seed: 1,
            },
        )
        .unwrap();
        let (s_ref, scores_ref, comps_ref) = dense_reference(&x, k);

        for (a, b) in got.singular_values.iter().zip(s_ref.iter()) {
            assert!((a - b).abs() < 1e-8);
        }
        for j in 0..k {
            let got_col = (0..x.n_rows)
                .map(|i| got.scores[i * k + j])
                .collect::<Vec<_>>();
            let ref_col = scores_ref.column(j).iter().copied().collect::<Vec<_>>();
            assert!(
                abs_dot(&got_col, &ref_col)
                    / (dot(&got_col, &got_col).sqrt() * dot(&ref_col, &ref_col).sqrt())
                    > 0.999999
            );
        }
        for j in 0..k {
            let got_row = (0..x.n_cols)
                .map(|i| got.components[j * x.n_cols + i])
                .collect::<Vec<_>>();
            let ref_row = comps_ref.row(j).iter().copied().collect::<Vec<_>>();
            assert!(abs_dot(&got_row, &ref_row) > 0.999999);
        }
    }

    #[test]
    fn matches_dense_svd_on_wide_sparse_matrix() {
        let x = test_csr(
            4,
            6,
            &[
                (0, 0, 3.0),
                (0, 3, 1.0),
                (0, 5, 2.0),
                (1, 1, 4.0),
                (1, 4, 1.0),
                (2, 0, 1.0),
                (2, 2, 2.0),
                (2, 5, 3.0),
                (3, 1, 2.0),
                (3, 3, 2.0),
                (3, 4, 4.0),
            ],
        );
        let k = 2;
        let _guard = lock_solver();
        let got = pca_scanpy_sparse_csr(
            &x,
            ScanpyPcaParams {
                n_components: k,
                tol: 0.0,
                ncv: None,
                maxiter: None,
                seed: 1,
            },
        )
        .unwrap();
        let (s_ref, scores_ref, comps_ref) = dense_reference(&x, k);

        for (a, b) in got.singular_values.iter().zip(s_ref.iter()) {
            assert!((a - b).abs() < 1e-8);
        }
        for j in 0..k {
            let got_col = (0..x.n_rows)
                .map(|i| got.scores[i * k + j])
                .collect::<Vec<_>>();
            let ref_col = scores_ref.column(j).iter().copied().collect::<Vec<_>>();
            assert!(
                abs_dot(&got_col, &ref_col)
                    / (dot(&got_col, &got_col).sqrt() * dot(&ref_col, &ref_col).sqrt())
                    > 0.999999
            );
        }
        for j in 0..k {
            let got_row = (0..x.n_cols)
                .map(|i| got.components[j * x.n_cols + i])
                .collect::<Vec<_>>();
            let ref_row = comps_ref.row(j).iter().copied().collect::<Vec<_>>();
            assert!(abs_dot(&got_row, &ref_row) > 0.999999);
        }
    }

    #[test]
    fn dense_path_matches_dense_svd_on_tall_matrix_and_warns() {
        let sparse = test_csr(
            6,
            4,
            &[
                (0, 0, 2.0),
                (0, 2, 1.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 3, 4.0),
                (3, 1, 2.0),
                (3, 3, 1.0),
                (4, 0, 3.0),
                (4, 2, 2.0),
                (5, 1, 1.0),
                (5, 3, 2.0),
            ],
        );
        let x = dense_matrix_from_csr(&sparse);
        let k = 2;
        let got = pca_scanpy_dense(
            &x,
            ScanpyPcaParams {
                n_components: k,
                tol: 0.0,
                ncv: Some(100),
                maxiter: Some(10),
                seed: 99,
            },
        )
        .unwrap();
        let (s_ref, scores_ref, comps_ref) = dense_reference(&sparse, k);

        assert_eq!(got.warnings.len(), 1);
        assert!(got.warnings[0].contains("input matrix is dense"));
        assert!(got.warnings[0].contains("ShiftedClrCsrMatrix"));
        for (a, b) in got.singular_values.iter().zip(s_ref.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        for j in 0..k {
            let got_col = (0..x.n_rows)
                .map(|i| got.scores[i * k + j])
                .collect::<Vec<_>>();
            let ref_col = scores_ref.column(j).iter().copied().collect::<Vec<_>>();
            assert!(
                abs_dot(&got_col, &ref_col)
                    / (dot(&got_col, &got_col).sqrt() * dot(&ref_col, &ref_col).sqrt())
                    > 0.999999
            );
        }
        for j in 0..k {
            let got_row = (0..x.n_cols)
                .map(|i| got.components[j * x.n_cols + i])
                .collect::<Vec<_>>();
            let ref_row = comps_ref.row(j).iter().copied().collect::<Vec<_>>();
            assert!(abs_dot(&got_row, &ref_row) > 0.999999);
        }
    }

    #[test]
    fn dense_path_matches_dense_svd_on_wide_matrix() {
        let sparse = test_csr(
            4,
            6,
            &[
                (0, 0, 2.0),
                (0, 3, 1.0),
                (0, 5, 3.0),
                (1, 1, 4.0),
                (1, 4, 1.0),
                (2, 0, 1.0),
                (2, 2, 2.0),
                (2, 5, 1.0),
                (3, 1, 3.0),
                (3, 3, 2.0),
                (3, 4, 2.0),
            ],
        );
        let x = dense_matrix_from_csr(&sparse);
        let k = 2;
        let got = pca_scanpy_dense(
            &x,
            ScanpyPcaParams {
                n_components: k,
                ..ScanpyPcaParams::default()
            },
        )
        .unwrap();
        let (s_ref, scores_ref, comps_ref) = dense_reference(&sparse, k);

        for (a, b) in got.singular_values.iter().zip(s_ref.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        for j in 0..k {
            let got_col = (0..x.n_rows)
                .map(|i| got.scores[i * k + j])
                .collect::<Vec<_>>();
            let ref_col = scores_ref.column(j).iter().copied().collect::<Vec<_>>();
            assert!(
                abs_dot(&got_col, &ref_col)
                    / (dot(&got_col, &got_col).sqrt() * dot(&ref_col, &ref_col).sqrt())
                    > 0.999999
            );
        }
        for j in 0..k {
            let got_row = (0..x.n_cols)
                .map(|i| got.components[j * x.n_cols + i])
                .collect::<Vec<_>>();
            let ref_row = comps_ref.row(j).iter().copied().collect::<Vec<_>>();
            assert!(abs_dot(&got_row, &ref_row) > 0.999999);
        }
    }

    #[test]
    fn shifted_clr_path_matches_dense_svd_on_tall_matrix() {
        let sparse = test_csr(
            7,
            4,
            &[
                (0, 0, 1.4),
                (0, 2, 0.6),
                (1, 1, 1.8),
                (1, 3, 0.9),
                (2, 0, 0.7),
                (2, 2, 1.5),
                (2, 3, 0.3),
                (3, 1, 1.2),
                (3, 2, 0.4),
                (4, 0, 1.9),
                (4, 3, 1.1),
                (5, 1, 0.8),
                (5, 2, 1.7),
                (6, 0, 0.5),
                (6, 1, 1.3),
            ],
        );
        let x = shifted_clr_from_sparse(sparse);
        let k = 2;
        let _guard = lock_solver();
        let got = pca_shifted_clr_sparse_csr(
            &x,
            ScanpyPcaParams {
                n_components: k,
                tol: 0.0,
                ncv: None,
                maxiter: None,
                seed: 1,
            },
        )
        .unwrap();
        let (s_ref, scores_ref, comps_ref) = shifted_clr_dense_reference(&x, k);

        for (a, b) in got.singular_values.iter().zip(s_ref.iter()) {
            assert!((a - b).abs() < 1e-8);
        }
        for j in 0..k {
            let got_col = (0..x.sparse.n_rows)
                .map(|i| got.scores[i * k + j])
                .collect::<Vec<_>>();
            let ref_col = scores_ref.column(j).iter().copied().collect::<Vec<_>>();
            assert!(
                abs_dot(&got_col, &ref_col)
                    / (dot(&got_col, &got_col).sqrt() * dot(&ref_col, &ref_col).sqrt())
                    > 0.999999
            );
        }
        for j in 0..k {
            let got_row = (0..x.sparse.n_cols)
                .map(|i| got.components[j * x.sparse.n_cols + i])
                .collect::<Vec<_>>();
            let ref_row = comps_ref.row(j).iter().copied().collect::<Vec<_>>();
            assert!(abs_dot(&got_row, &ref_row) > 0.999999);
        }
    }

    #[test]
    fn shifted_clr_sparse_representation_matches_dense_materialization() {
        let sparse = test_csr(
            6,
            5,
            &[
                (0, 0, 1.4),
                (0, 2, 0.6),
                (1, 1, 1.8),
                (1, 3, 0.9),
                (2, 0, 0.7),
                (2, 2, 1.5),
                (2, 4, 0.3),
                (3, 1, 1.2),
                (3, 2, 0.4),
                (4, 0, 1.9),
                (4, 3, 1.1),
                (5, 1, 0.8),
                (5, 2, 1.7),
                (5, 4, 0.5),
            ],
        );
        let x = shifted_clr_from_sparse(sparse);
        let dense = shifted_clr_dense(&x);
        let centered_dense = shifted_clr_centered_dense(&x);
        let (mean, var) = x.mean_variance_axis0();

        let mut dense_mean = vec![0.0; x.sparse.n_cols];
        let mut dense_var = vec![0.0; x.sparse.n_cols];
        for j in 0..x.sparse.n_cols {
            for i in 0..x.sparse.n_rows {
                dense_mean[j] += dense[(i, j)];
                dense_var[j] += dense[(i, j)] * dense[(i, j)];
            }
            dense_mean[j] /= x.sparse.n_rows as f64;
            dense_var[j] = dense_var[j] / x.sparse.n_rows as f64 - dense_mean[j] * dense_mean[j];
        }
        assert_all_close(&mean, &dense_mean, 1e-14);
        assert_all_close(&var, &dense_var, 1e-14);

        let right = [0.7, -1.1, 0.3, 2.0, -0.4];
        let mut got_rows = vec![0.0; x.sparse.n_rows];
        x.matvec(&right, &mut got_rows);
        let mut dense_rows = vec![0.0; x.sparse.n_rows];
        for i in 0..x.sparse.n_rows {
            for j in 0..x.sparse.n_cols {
                dense_rows[i] += dense[(i, j)] * right[j];
            }
        }
        assert_all_close(&got_rows, &dense_rows, 1e-14);

        let left = [1.3, -0.2, 0.8, -1.7, 0.4, 0.6];
        let mut got_cols = vec![0.0; x.sparse.n_cols];
        x.rmatvec(&left, &mut got_cols);
        let mut dense_cols = vec![0.0; x.sparse.n_cols];
        for i in 0..x.sparse.n_rows {
            for j in 0..x.sparse.n_cols {
                dense_cols[j] += dense[(i, j)] * left[i];
            }
        }
        assert_all_close(&got_cols, &dense_cols, 1e-14);

        let centered = ImplicitColumnOffset { x: &x, mean: &mean };
        centered.matvec(&right, &mut got_rows);
        dense_rows.fill(0.0);
        for i in 0..x.sparse.n_rows {
            for j in 0..x.sparse.n_cols {
                dense_rows[i] += centered_dense[(i, j)] * right[j];
            }
        }
        assert_all_close(&got_rows, &dense_rows, 1e-14);

        centered.rmatvec(&left, &mut got_cols);
        dense_cols.fill(0.0);
        for i in 0..x.sparse.n_rows {
            for j in 0..x.sparse.n_cols {
                dense_cols[j] += centered_dense[(i, j)] * left[i];
            }
        }
        assert_all_close(&got_cols, &dense_cols, 1e-14);

        let mut tmp = vec![0.0; x.sparse.n_rows];
        let mut got_normal = vec![0.0; x.sparse.n_cols];
        centered.matvec(&right, &mut tmp);
        centered.rmatvec(&tmp, &mut got_normal);
        let mut dense_tmp = vec![0.0; x.sparse.n_rows];
        let mut dense_normal = vec![0.0; x.sparse.n_cols];
        for i in 0..x.sparse.n_rows {
            for j in 0..x.sparse.n_cols {
                dense_tmp[i] += centered_dense[(i, j)] * right[j];
            }
        }
        for i in 0..x.sparse.n_rows {
            for j in 0..x.sparse.n_cols {
                dense_normal[j] += centered_dense[(i, j)] * dense_tmp[i];
            }
        }
        assert_all_close(&got_normal, &dense_normal, 1e-13);
    }

    #[test]
    fn shifted_clr_path_matches_dense_svd_on_wide_matrix() {
        let sparse = test_csr(
            4,
            7,
            &[
                (0, 0, 1.6),
                (0, 3, 0.8),
                (0, 6, 1.1),
                (1, 1, 1.5),
                (1, 4, 0.7),
                (1, 5, 1.4),
                (2, 0, 0.9),
                (2, 2, 1.8),
                (2, 6, 0.6),
                (3, 1, 0.5),
                (3, 3, 1.2),
                (3, 4, 1.9),
            ],
        );
        let x = shifted_clr_from_sparse(sparse);
        let k = 2;
        let _guard = lock_solver();
        let got = pca_shifted_clr_sparse_csr(
            &x,
            ScanpyPcaParams {
                n_components: k,
                tol: 0.0,
                ncv: None,
                maxiter: None,
                seed: 1,
            },
        )
        .unwrap();
        let (s_ref, scores_ref, comps_ref) = shifted_clr_dense_reference(&x, k);

        for (a, b) in got.singular_values.iter().zip(s_ref.iter()) {
            assert!((a - b).abs() < 1e-8);
        }
        for j in 0..k {
            let got_col = (0..x.sparse.n_rows)
                .map(|i| got.scores[i * k + j])
                .collect::<Vec<_>>();
            let ref_col = scores_ref.column(j).iter().copied().collect::<Vec<_>>();
            assert!(
                abs_dot(&got_col, &ref_col)
                    / (dot(&got_col, &got_col).sqrt() * dot(&ref_col, &ref_col).sqrt())
                    > 0.999999
            );
        }
        for j in 0..k {
            let got_row = (0..x.sparse.n_cols)
                .map(|i| got.components[j * x.sparse.n_cols + i])
                .collect::<Vec<_>>();
            let ref_row = comps_ref.row(j).iter().copied().collect::<Vec<_>>();
            assert!(abs_dot(&got_row, &ref_row) > 0.999999);
        }
    }

    #[test]
    fn matches_scanpy_on_tall_sparse_matrix() {
        let _guard = lock_solver();
        let x = test_csr(
            6,
            4,
            &[
                (0, 0, 2.0),
                (0, 2, 1.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 3, 4.0),
                (3, 1, 2.0),
                (3, 3, 1.0),
                (4, 0, 3.0),
                (4, 2, 2.0),
                (5, 1, 1.0),
                (5, 3, 2.0),
            ],
        );
        compare_with_scanpy(&x, 2, 0);
    }

    #[test]
    fn matches_scanpy_on_wide_sparse_matrix() {
        let _guard = lock_solver();
        let x = test_csr(
            4,
            6,
            &[
                (0, 0, 3.0),
                (0, 3, 1.0),
                (0, 5, 2.0),
                (1, 1, 4.0),
                (1, 4, 1.0),
                (2, 0, 1.0),
                (2, 2, 2.0),
                (2, 5, 3.0),
                (3, 1, 2.0),
                (3, 3, 2.0),
                (3, 4, 4.0),
            ],
        );
        compare_with_scanpy(&x, 2, 0);
    }

    #[test]
    fn matches_scanpy_on_medium_sparse_matrix() {
        let _guard = lock_solver();
        let mut entries = Vec::new();
        for i in 0..9usize {
            for j in 0..7usize {
                if (i * 3 + j * 5 + 1) % 4 != 0 {
                    let v = (((i * 11 + j * 7 + 3) % 9) + 1) as f64;
                    entries.push((i, j, v));
                }
            }
        }
        let x = test_csr(9, 7, &entries);
        compare_with_scanpy(&x, 3, 0);
    }

    #[test]
    fn dsortr_matches_arpack_ordering() {
        let mut x1 = vec![3.0, -1.0, 2.0, -4.0];
        let mut x2 = vec![30.0, 10.0, 20.0, 40.0];
        dsortr("LM", true, &mut x1, &mut x2);
        assert_eq!(x1, vec![-1.0, 2.0, 3.0, -4.0]);
        assert_eq!(x2, vec![10.0, 20.0, 30.0, 40.0]);

        dsortr("SM", true, &mut x1, &mut x2);
        assert_eq!(x1, vec![-4.0, 3.0, 2.0, -1.0]);
        assert_eq!(x2, vec![40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn dsesrt_matches_arpack_column_permutation() {
        let mut x = vec![3.0, -1.0, 2.0];
        let mut a = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        dsesrt("LA", true, &mut x, &mut a);
        assert_eq!(x, vec![-1.0, 2.0, 3.0]);
        assert_eq!(
            a.column(0).iter().copied().collect::<Vec<_>>(),
            vec![2.0, 20.0]
        );
        assert_eq!(
            a.column(1).iter().copied().collect::<Vec<_>>(),
            vec![3.0, 30.0]
        );
        assert_eq!(
            a.column(2).iter().copied().collect::<Vec<_>>(),
            vec![1.0, 10.0]
        );
    }

    #[test]
    fn dsgets_matches_arpack_shift_layout() {
        let mut ritz = vec![4.0, 1.0, 3.0, 2.0];
        let mut bounds = vec![0.4, 0.1, 0.3, 0.2];
        let mut shifts = vec![0.0; 2];
        dsgets(1, "LM", 2, 2, &mut ritz, &mut bounds, &mut shifts);
        assert_eq!(ritz, vec![2.0, 1.0, 3.0, 4.0]);
        assert_eq!(bounds, vec![0.2, 0.1, 0.3, 0.4]);
        assert_eq!(shifts, vec![2.0, 1.0]);
    }

    #[test]
    fn dsconv_matches_arpack_convergence_count() {
        let ritz = vec![10.0, 1e-20, -3.0];
        let bounds = vec![1e-6, 1e-18, 1e-2];
        assert_eq!(dsconv(&ritz, &bounds, 1e-5), 2);
    }

    #[test]
    fn dseigt_matches_tridiagonal_eigendecomposition() {
        let h = DMatrix::from_row_slice(3, 2, &[0.0, 2.0, 0.5, 3.0, -0.25, 5.0]);
        let (eig, bounds) = dseigt(0.7, &h);
        assert_eq!(eig.len(), 3);
        assert_eq!(bounds.len(), 3);
        let mut eig_sorted = eig.clone();
        eig_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(eig_sorted[0] <= eig_sorted[1] && eig_sorted[1] <= eig_sorted[2]);
        assert!(bounds.iter().all(|b| *b >= 0.0));
    }

    #[test]
    fn dsapps_keeps_tridiagonal_shape() {
        let n = 5;
        let kev = 2;
        let np = 2;
        let mut v = DMatrix::from_fn(n, kev + np, |r, c| ((r + 1 + 3 * c) as f64) / 10.0);
        let mut h = DMatrix::from_row_slice(kev + np, 2, &[0.0, 4.0, 0.5, 3.0, 0.2, 2.0, 0.1, 1.0]);
        let mut resid = vec![0.2, -0.1, 0.05, 0.3, -0.2];
        let shift = vec![0.3, 0.7];
        dsapps(kev, np, &shift, &mut v, &mut h, &mut resid);
        assert!(h[(1, 0)] >= 0.0 && h[(2, 0)] >= 0.0 && h[(3, 0)] >= 0.0);
        assert!(resid.iter().all(|x| x.is_finite()));
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn dsaitr_builds_valid_lanczos_factorization() {
        let n = 5usize;
        let k = 0usize;
        let np = 3usize;
        let mut v = DMatrix::zeros(n, k + np);
        let mut h = DMatrix::zeros(k + np, 2);
        let mut resid = vec![1.0, 2.0, -1.0, 0.5, 3.0];
        let mut rnorm = dot(&resid, &resid).sqrt();
        let a = DMatrix::from_row_slice(
            n,
            n,
            &[
                4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 0.0, 1.0, 6.0,
            ],
        );
        let out = dsaitr_mode1(n, k, np, &mut resid, &mut rnorm, &mut v, &mut h, |x, y| {
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += a[(i, j)] * x[j];
                }
                y[i] = acc;
            }
        })
        .unwrap();
        assert_eq!(out.info, 0);
        for i in 0..np {
            let col_i = v.column(i);
            let ni = col_i.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((ni - 1.0).abs() < 1e-10);
            for j in 0..i {
                let col_j = v.column(j);
                let dot_ij = col_i
                    .iter()
                    .zip(col_j.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                assert!(dot_ij.abs() < 1e-8);
            }
        }
        let mut t = DMatrix::zeros(np, np);
        for i in 0..np {
            t[(i, i)] = h[(i, 1)];
            if i > 0 {
                t[(i, i - 1)] = h[(i, 0)];
                t[(i - 1, i)] = h[(i, 0)];
            }
        }
        let av = &a * &v;
        let vt = &v * &t;
        for row in 0..n {
            assert!((av[(row, np - 1)] - (vt[(row, np - 1)] + resid[row])).abs() < 1e-7);
        }
    }

    #[test]
    fn dsaup2_runs_major_iteration_loop() {
        let n = 6usize;
        let nev0 = 2usize;
        let np0 = 2usize;
        let kplusp = nev0 + np0;
        let mut v = DMatrix::zeros(n, kplusp);
        let mut h = DMatrix::zeros(kplusp, 2);
        let mut resid = vec![1.0, -1.0, 0.5, 2.0, -0.25, 1.5];
        let mut rnorm = dot(&resid, &resid).sqrt();
        let a = DMatrix::from_row_slice(
            n,
            n,
            &[
                6.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 1.0,
            ],
        );
        let out = dsaup2_mode1(
            "LM",
            nev0,
            np0,
            1e-8,
            10,
            &mut resid,
            &mut rnorm,
            &mut v,
            &mut h,
            |x, y| {
                for i in 0..n {
                    let mut acc = 0.0;
                    for j in 0..n {
                        acc += a[(i, j)] * x[j];
                    }
                    y[i] = acc;
                }
            },
        )
        .unwrap();
        assert!(out.info == 0 || out.info == 1 || out.info == 2 || out.info == -9999);
        assert!(out.nconv <= nev0);
        if out.info != -9999 {
            assert!(out.ritz.len() >= nev0);
            assert!(out.bounds.len() >= nev0);
            assert!(out.ritz.iter().all(|x| x.is_finite()));
            assert!(out.bounds.iter().all(|x| x.is_finite()));
        } else {
            assert!(out.np > 0);
        }
    }

    /// Regression test for the `dsapps` implicit-restart Q-accumulation bug.
    ///
    /// On a matrix large enough to require several implicit restarts (`ncv < min_dim`), the sparse
    /// ARPACK path previously corrupted the tridiagonal `H`, returning Ritz values *outside* the
    /// operator spectrum (so a dominant principal component was missed) or converging zero
    /// eigenpairs and panicking. The small fixtures in the other tests converge in a single sweep,
    /// so they did not exercise the restart path. The exact dense path is the oracle here.
    #[test]
    fn sparse_pca_matches_dense_on_structured_medium_matrix() {
        let (n_rows, n_cols, n_groups) = (120usize, 30usize, 3usize);
        let n_per = n_rows / n_groups;
        let block = n_cols / (n_groups + 1);
        // Deterministic LCG: structured clusters give well-separated top eigenvalues.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 40) as f64 / (1u64 << 24) as f64
        };
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0usize];
        for i in 0..n_rows {
            let g = i / n_per;
            for j in 0..n_cols {
                let mut v = if next() < 0.5 { 0.0 } else { (next() * 5.0).floor() };
                if j >= g * block && j < (g + 1) * block {
                    v += 25.0;
                }
                if j == 0 {
                    v += 1.0;
                }
                if v != 0.0 {
                    data.push(v);
                    indices.push(j);
                }
            }
            indptr.push(data.len());
        }
        let sparse = CsrMatrix { n_rows, n_cols, data, indices, indptr };
        let dense = dense_matrix_from_csr(&sparse);
        let params = ScanpyPcaParams { n_components: 5, ..Default::default() };

        let r_sparse = pca_scanpy_sparse_csr(&sparse, params).expect("sparse pca");
        let r_dense = pca_scanpy_dense(&dense, params).expect("dense pca");

        for (s, d) in r_sparse.explained_variance.iter().zip(r_dense.explained_variance.iter()) {
            let rel = (s - d).abs() / d.abs().max(1e-12);
            assert!(rel < 1e-6, "sparse EV {s} vs dense EV {d} (rel err {rel})");
        }
        // A correct Lanczos cannot report a Ritz value above the operator's largest eigenvalue.
        let max_dense = r_dense.explained_variance[0];
        assert!(
            r_sparse.explained_variance.iter().all(|&v| v <= max_dense * (1.0 + 1e-6)),
            "sparse PCA produced an explained variance above the dense maximum"
        );
    }
}
