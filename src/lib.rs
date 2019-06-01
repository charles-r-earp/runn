//pub mod cuda;
//pub mod opencl;
#[cfg(feature="native")]
pub mod native;
use std::fmt;
use ndarray::{ArrayBase, Array, Data, Dimension};

pub trait Executor: fmt::Display + fmt::Debug {}

pub trait DotExec<A, B>: Executor {
  type Output;
  fn dot(&self, a: A, b: B) -> Self::Output;
}
