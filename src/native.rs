use crate::{Executor, DotExec};
use std::fmt;
use ndarray::{ArrayBase, Data};
use ndarray::linalg::Dot;

#[derive(Debug)]
pub struct Exec;

impl fmt::Display for Exec {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Native")
  }
}

impl Executor for Exec {}

impl<S1, D1, S2, D2> DotExec<ArrayBase<S1, D1>, ArrayBase<S2, D2>> for Exec 
  where S1: Data,
        S2: Data,
        ArrayBase<S1, D1>: Dot<ArrayBase<S2, D2>> {
  type Output = <ArrayBase<S1, D1> as Dot<ArrayBase<S2, D2>>>::Output;
  #[inline]
  fn dot(&self, a: ArrayBase<S1, D1>, b: ArrayBase<S2, D2>) -> Self::Output {
    a.dot(&b)
  } 
}
