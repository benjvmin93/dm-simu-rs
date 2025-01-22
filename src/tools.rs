use core::fmt;
use num_complex::Complex;
use std::collections::HashSet;

pub struct DisplayComplex(pub Complex<f64>);

impl fmt::Display for DisplayComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = {
            if self.0.im < 0. {
                "+".to_string()
            } else {
                "-".to_string()
            }
        };
        write!(f, "{}{}{}i", self.0.re, sign, self.0.im)
    }
}

pub fn complex_approx_eq(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
    let re_diff = (a.re - b.re).abs();
    let im_diff = (a.im - b.im).abs();
    (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol
}

pub fn are_elements_unique<T: Eq + std::hash::Hash>(slice: &[T]) -> bool {
    let mut seen = HashSet::new();
    for element in slice {
        if !seen.insert(element) {
            return false;
        }
    }
    true
}
