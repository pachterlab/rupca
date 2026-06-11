use rupca::{
    pca_scanpy_sparse_csr, pca_shifted_clr_sparse_csr, CsrMatrix, ScanpyPcaParams,
    ShiftedClrCsrMatrix,
};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn read_matrix_market(path: &str) -> CsrMatrix {
    let file = File::open(path).expect("failed to open matrix");
    let reader = BufReader::new(file);

    let mut n_rows = 0usize;
    let mut n_cols = 0usize;
    let mut nnz = 0usize;
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::<usize>::new();
    let mut header_seen = false;
    let mut current_row = 0usize;

    for line in reader.lines() {
        let line = line.expect("failed to read line");
        if line.starts_with('%') || line.trim().is_empty() {
            continue;
        }
        if !header_seen {
            let parts = line.split_whitespace().collect::<Vec<_>>();
            n_rows = parts[0].parse().unwrap();
            n_cols = parts[1].parse().unwrap();
            nnz = parts[2].parse().unwrap();
            data.reserve(nnz);
            indices.reserve(nnz);
            indptr = Vec::with_capacity(n_rows + 1);
            indptr.push(0);
            header_seen = true;
            continue;
        }

        let parts = line.split_whitespace().collect::<Vec<_>>();
        let i = parts[0].parse::<usize>().unwrap() - 1;
        let j = parts[1].parse::<usize>().unwrap() - 1;
        let v = parts[2].parse::<f64>().unwrap();
        assert!(
            i >= current_row,
            "matrix entries must be sorted by row for streaming benchmark reader"
        );
        while current_row < i {
            current_row += 1;
            indptr.push(data.len());
        }
        indices.push(j);
        data.push(v);
    }

    while current_row < n_rows {
        current_row += 1;
        indptr.push(data.len());
    }
    assert_eq!(data.len(), nnz);

    CsrMatrix {
        n_rows,
        n_cols,
        data,
        indices,
        indptr,
    }
}

fn shifted_clr_from_counts(mut counts: CsrMatrix) -> ShiftedClrCsrMatrix {
    let mut row_center = vec![0.0; counts.n_rows];
    for (i, center) in row_center.iter_mut().enumerate() {
        let mut row_sum = 0.0;
        for p in counts.indptr[i]..counts.indptr[i + 1] {
            let v = counts.data[p].ln_1p();
            counts.data[p] = v;
            row_sum += v;
        }
        *center = row_sum / counts.n_cols as f64;
    }
    ShiftedClrCsrMatrix {
        sparse: counts,
        row_center,
    }
}

fn bench_shifted_clr(
    x: &ShiftedClrCsrMatrix,
    params: ScanpyPcaParams,
    repeats: usize,
) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut sum = 0.0;
    for _ in 0..repeats {
        let start = Instant::now();
        let result = pca_shifted_clr_sparse_csr(x, params).expect("shifted CLR PCA failed");
        let secs = start.elapsed().as_secs_f64();
        best = best.min(secs);
        sum += secs;
        std::hint::black_box(result);
    }
    (best, sum / repeats as f64)
}

fn bench_sparse(x: &CsrMatrix, params: ScanpyPcaParams, repeats: usize) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut sum = 0.0;
    for _ in 0..repeats {
        let start = Instant::now();
        let result = pca_scanpy_sparse_csr(x, params).expect("sparse PCA failed");
        let secs = start.elapsed().as_secs_f64();
        best = best.min(secs);
        sum += secs;
        std::hint::black_box(result);
    }
    (best, sum / repeats as f64)
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let raw_counts_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/Users/lpachter/Dropbox/claude/projects/rutest/pbmc10k/matrix.mtx");
    let logpf_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("/Users/lpachter/Dropbox/claude/projects/rutest/pbmc10k/normalized_log1pPF.mtx");
    let n_components = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);
    let repeats = args
        .get(4)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let ncv = args.get(5).and_then(|s| s.parse::<usize>().ok());
    assert!(repeats > 0, "repeats must be positive");

    let params = ScanpyPcaParams {
        n_components,
        tol: 0.0,
        ncv,
        maxiter: None,
        seed: 0,
    };

    let load_raw_start = Instant::now();
    let counts = read_matrix_market(raw_counts_path);
    let load_raw_seconds = load_raw_start.elapsed().as_secs_f64();
    let rows = counts.n_rows;
    let cols = counts.n_cols;
    let nnz = counts.data.len();

    let build_clr_start = Instant::now();
    let shifted_clr = shifted_clr_from_counts(counts);
    let build_clr_seconds = build_clr_start.elapsed().as_secs_f64();
    let (shifted_best, shifted_mean) = bench_shifted_clr(&shifted_clr, params, repeats);
    drop(shifted_clr);

    let load_logpf_start = Instant::now();
    let logpf = read_matrix_market(logpf_path);
    let load_logpf_seconds = load_logpf_start.elapsed().as_secs_f64();
    assert_eq!(logpf.n_rows, rows);
    assert_eq!(logpf.n_cols, cols);
    assert_eq!(logpf.data.len(), nnz);
    let (logpf_best, logpf_mean) = bench_sparse(&logpf, params, repeats);

    println!(
        concat!(
            "{{",
            "\"rows\":{},\"cols\":{},\"nnz\":{},",
            "\"n_components\":{},\"repeats\":{},\"ncv\":{},",
            "\"shifted_clr_load_raw_seconds\":{:.6},",
            "\"shifted_clr_build_seconds\":{:.6},",
            "\"shifted_clr_pca_best_seconds\":{:.6},",
            "\"shifted_clr_pca_mean_seconds\":{:.6},",
            "\"logpf_load_seconds\":{:.6},",
            "\"logpf_pca_best_seconds\":{:.6},",
            "\"logpf_pca_mean_seconds\":{:.6},",
            "\"pca_mean_ratio_shifted_clr_over_logpf\":{:.6}",
            "}}"
        ),
        rows,
        cols,
        nnz,
        n_components,
        repeats,
        ncv.map(|x| x.to_string())
            .unwrap_or_else(|| "null".to_string()),
        load_raw_seconds,
        build_clr_seconds,
        shifted_best,
        shifted_mean,
        load_logpf_seconds,
        logpf_best,
        logpf_mean,
        shifted_mean / logpf_mean
    );
}
