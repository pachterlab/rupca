use nalgebra::DMatrix;

fn axpy_into(dst: &mut [f64], alpha: f64, x: &[f64]) {
    for i in 0..dst.len() {
        dst[i] += alpha * x[i];
    }
}

fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    if g == 0.0 {
        let c = if f >= 0.0 { 1.0 } else { -1.0 };
        return (c, 0.0, f.abs());
    }
    if f == 0.0 {
        return (0.0, 1.0, g);
    }
    let scale = f.abs().max(g.abs());
    let fs = f / scale;
    let gs = g / scale;
    let mut r = scale * (fs * fs + gs * gs).sqrt();
    let mut c = f / r;
    let mut s = g / r;
    if f.abs() > g.abs() && c < 0.0 {
        c = -c;
        s = -s;
        r = -r;
    }
    (c, s, r)
}

pub fn dsapps(
    kev: usize,
    np: usize,
    shift: &[f64],
    v: &mut DMatrix<f64>,
    h: &mut DMatrix<f64>,
    resid: &mut [f64],
) {
    let n = v.nrows();
    let kplusp = kev + np;
    assert_eq!(shift.len(), np);
    assert_eq!(v.ncols(), kplusp);
    assert_eq!(h.nrows(), kplusp);
    assert_eq!(h.ncols(), 2);
    assert_eq!(resid.len(), n);

    let epsmch = f64::EPSILON;
    let mut q = DMatrix::identity(kplusp, kplusp);
    if np == 0 {
        return;
    }
    let mut itop = 0usize;

    for &sigma in shift {
        let mut istart = itop;
        loop {
            let mut iend = kplusp - 1;
            for i in istart..kplusp - 1 {
                let big = h[(i, 1)].abs() + h[(i + 1, 1)].abs();
                if h[(i + 1, 0)] <= epsmch * big {
                    h[(i + 1, 0)] = 0.0;
                    iend = i;
                    break;
                }
            }

            if istart < iend {
                let f = h[(istart, 1)] - sigma;
                let g = h[(istart + 1, 0)];
                let (mut c, mut s, _r) = dlartg(f, g);

                let a1 = c * h[(istart, 1)] + s * h[(istart + 1, 0)];
                let a2 = c * h[(istart + 1, 0)] + s * h[(istart + 1, 1)];
                let a4 = c * h[(istart + 1, 1)] - s * h[(istart + 1, 0)];
                let a3 = c * h[(istart + 1, 0)] - s * h[(istart, 1)];
                h[(istart, 1)] = c * a1 + s * a2;
                h[(istart + 1, 1)] = c * a4 - s * a3;
                h[(istart + 1, 0)] = c * a3 + s * a4;

                // Accumulate Q <- Q * G(istart, istart+1). The Fortran dsapps.f:308 bounds this by
                // `min(istart+jj, kplusp)` rows (jj = the implicit-shift index), an optimization
                // exploiting that lower rows are still zero at shift jj. The original port dropped
                // the `jj` term, leaving nonzero Q entries un-rotated and corrupting the restarted
                // Lanczos basis (producing Ritz values outside the operator's spectrum and false
                // convergence). A Givens right-multiply touches all rows of the two columns, so
                // applying it to every row is exact (rotating structural zeros is a no-op).
                for j in 0..kplusp {
                    let a1: f64 = c * q[(j, istart)] + s * q[(j, istart + 1)];
                    q[(j, istart + 1)] = -s * q[(j, istart)] + c * q[(j, istart + 1)];
                    q[(j, istart)] = a1;
                }

                for i in istart + 1..iend {
                    let f = h[(i, 0)];
                    let g = s * h[(i + 1, 0)];
                    h[(i + 1, 0)] = c * h[(i + 1, 0)];
                    let (c_new, s_new, mut r) = dlartg(f, g);
                    c = c_new;
                    s = s_new;
                    if r < 0.0 {
                        r = -r;
                        c = -c;
                        s = -s;
                    }
                    h[(i, 0)] = r;

                    let a1 = c * h[(i, 1)] + s * h[(i + 1, 0)];
                    let a2 = c * h[(i + 1, 0)] + s * h[(i + 1, 1)];
                    let a3 = c * h[(i + 1, 0)] - s * h[(i, 1)];
                    let a4 = c * h[(i + 1, 1)] - s * h[(i + 1, 0)];
                    h[(i, 1)] = c * a1 + s * a2;
                    h[(i + 1, 1)] = c * a4 - s * a3;
                    h[(i + 1, 0)] = c * a3 + s * a4;

                    // Same fix as the first rotation: accumulate over all rows (Fortran dsapps.f:375
                    // bounds this by `min(i+jj, kplusp)`; the dropped `jj` corrupted Q).
                    for j in 0..kplusp {
                        let a1: f64 = c * q[(j, i)] + s * q[(j, i + 1)];
                        q[(j, i + 1)] = -s * q[(j, i)] + c * q[(j, i + 1)];
                        q[(j, i)] = a1;
                    }
                }
            }

            istart = iend + 1;
            if iend < kplusp && h[(iend, 0)] < 0.0 {
                h[(iend, 0)] = -h[(iend, 0)];
                for row in 0..kplusp {
                    q[(row, iend)] = -q[(row, iend)];
                }
            }
            if iend + 1 >= kplusp {
                break;
            }
        }

        while itop + 1 < kplusp && h[(itop + 1, 0)] <= 0.0 {
            itop += 1;
        }
    }

    for i in itop..kplusp - 1 {
        let big = h[(i, 1)].abs() + h[(i + 1, 1)].abs();
        if h[(i + 1, 0)] <= epsmch * big {
            h[(i + 1, 0)] = 0.0;
        }
    }

    let mut extra = vec![0.0; n];
    let mut work_col = vec![0.0; n];
    if h[(kev, 0)] > 0.0 {
        for col in 0..kplusp {
            let coeff = q[(col, kev)];
            if coeff != 0.0 {
                axpy_into(&mut extra, coeff, v.column(col).as_slice());
            }
        }
    }

    for i in 0..kev {
        let q_col = kev - 1 - i;
        let end_col = kplusp - i;
        work_col.fill(0.0);
        for col in 0..end_col {
            let coeff = q[(col, q_col)];
            if coeff != 0.0 {
                axpy_into(&mut work_col, coeff, v.column(col).as_slice());
            }
        }
        let target_col = kplusp - 1 - i;
        v.column_mut(target_col).copy_from_slice(&work_col);
    }

    for i in 0..kev {
        let src = np + i;
        for row in 0..n {
            v[(row, i)] = v[(row, src)];
        }
    }
    if h[(kev, 0)] > 0.0 {
        for row in 0..n {
            v[(row, kev)] = extra[row];
        }
    }

    let sigmak = q[(kplusp - 1, kev - 1)];
    for r in resid.iter_mut() {
        *r *= sigmak;
    }
    if h[(kev, 0)] > 0.0 {
        let betak = h[(kev, 0)];
        for row in 0..n {
            resid[row] += betak * v[(row, kev)];
        }
    }
}
