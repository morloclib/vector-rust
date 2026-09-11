// Rust implementations of the morloc vector stdlib. Vector maps to Vec<T>
// (List's runtime form); the Nat dimension is phantom at runtime.
//
// These mirror root-rust's List operations in core.rs but are kept
// SELF-CONTAINED (not delegating to core.rs): morloc includes a sourced .rs
// only when its symbols are USED, so core.rs is not guaranteed to be in a
// vector-only pool. Distinct morloc_vec_* names also avoid colliding with
// core.rs's morloc_* family when both files are present. Argument-passing
// follows the morloc Rust FFI convention (Copy scalars by value,
// non-Copy/generic by shared reference).

// zeros/ones are determined by the RESULT element type, not by any argument.
// This trait supplies the 0 and 1 of each numeric type; Rust resolves the type
// parameter from the call-site context. tensor-rust declares its own
// equivalent: a sourced .rs reaches a pool only when its symbols are used,
// so neither file may depend on the other being present.
trait MorlocVectorNumConst {
    fn m_vector_zero() -> Self;
    fn m_vector_one() -> Self;
}
macro_rules! morloc_vector_num_const {
    ($($t:ty),*) => {
        $(impl MorlocVectorNumConst for $t {
            fn m_vector_zero() -> Self { 0 as $t }
            fn m_vector_one() -> Self { 1 as $t }
        })*
    };
}
morloc_vector_num_const!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

// A negative length would wrap to a near-2^64 usize under `as`, turning a bad
// input into an OOM/capacity-overflow abort; throw a catchable morloc error.
fn morloc_vector_usz(d: i64) -> usize {
    if d < 0 {
        rustmorloc::morloc_throw("vector length must be non-negative");
    }
    d as usize
}

pub fn morloc_zeros1<A: MorlocVectorNumConst + Clone>(d1: i64) -> Vec<A> {
    vec![A::m_vector_zero(); morloc_vector_usz(d1)]
}

pub fn morloc_ones1<A: MorlocVectorNumConst + Clone>(d1: i64) -> Vec<A> {
    vec![A::m_vector_one(); morloc_vector_usz(d1)]
}

pub fn morloc_fill1<A: Clone>(v: &A, d1: i64) -> Vec<A> {
    vec![v.clone(); morloc_vector_usz(d1)]
}

pub fn morloc_vec_concat<A: Clone>(xs: &[A], ys: &[A]) -> Vec<A> {
    let mut v = Vec::with_capacity(xs.len() + ys.len());
    v.extend_from_slice(xs);
    v.extend_from_slice(ys);
    v
}

pub fn morloc_vec_size<A>(xs: &[A]) -> u64 {
    xs.len() as u64
}

pub fn morloc_vec_map<A, B, F: Fn(&A) -> B>(f: F, xs: &[A]) -> Vec<B> {
    xs.iter().map(|x| f(x)).collect()
}

pub fn morloc_vec_fold<B: Clone, A, F: Fn(&B, &A) -> B>(f: F, init: &B, xs: &[A]) -> B {
    xs.iter().fold(init.clone(), |acc, x| f(&acc, x))
}

pub fn morloc_vec_fold1<A: Clone, F: Fn(&A, &A) -> A>(f: F, xs: &[A]) -> A {
    let mut it = xs.iter();
    let mut acc = it
        .next()
        .unwrap_or_else(|| rustmorloc::morloc_throw("fold1: empty vector"))
        .clone();
    for x in it {
        acc = f(&acc, x);
    }
    acc
}

pub fn morloc_vec_safeFold1<A: Clone, F: Fn(&A, &A) -> A>(f: F, xs: &[A]) -> Option<A> {
    let mut it = xs.iter();
    let mut acc = it.next()?.clone();
    for x in it {
        acc = f(&acc, x);
    }
    Some(acc)
}

pub fn morloc_vec_eq<A: PartialEq>(xs: &[A], ys: &[A]) -> bool {
    xs == ys
}

pub fn morloc_vec_le<A: PartialOrd>(xs: &[A], ys: &[A]) -> bool {
    xs <= ys
}

// Negative indices count from the end (Python semantics). The index arrives as
// ?I64 to match __to_index__'s return shape.
pub fn morloc_vec_at<A: Clone>(oi: Option<i64>, xs: &[A]) -> A {
    let i = oi.unwrap_or_else(|| rustmorloc::morloc_throw("morloc_vec_at: index is null"));
    let n = xs.len() as i64;
    let j = if i < 0 { i + n } else { i };
    if j < 0 || j >= n {
        rustmorloc::morloc_throw("index out of bounds");
    }
    xs[j as usize].clone()
}

// Python-style slice with an optional start/stop/step; negative bounds and a
// negative step are supported. Bounds arrive as ?I64.
pub fn morloc_vec_slice<A: Clone>(
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
    xs: &[A],
) -> Vec<A> {
    let n = xs.len() as i64;
    let k = step.unwrap_or(1);
    if k == 0 {
        rustmorloc::morloc_throw("slice step cannot be zero");
    }
    let (mut i, mut j) = if k > 0 {
        (start.unwrap_or(0), stop.unwrap_or(n))
    } else {
        (start.unwrap_or(n - 1), stop.unwrap_or(-1))
    };
    if i < 0 && start.is_some() {
        i += n;
    }
    if j < 0 && stop.is_some() {
        j += n;
    }
    if k > 0 {
        i = i.clamp(0, n);
        j = j.clamp(0, n);
    } else {
        i = i.clamp(-1, n - 1);
        j = j.clamp(-1, n - 1);
    }
    let mut out = Vec::new();
    let mut p = i;
    while (k > 0 && p < j) || (k < 0 && p > j) {
        out.push(xs[p as usize].clone());
        p += k;
    }
    out
}
