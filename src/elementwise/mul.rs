em::emu!{
  mul(global_y [f32], global_x1 [f32], global_x2 [f32]) {
    let gid: i32 = get_global_id(0);
    global_y[gid] = global_x1[gid] * global_x2[gid];
  }
}

fn emu() -> &'static str { EMU }

pub mod opencl {
  use super::emu;
  use super::super::opencl::{binary};
  
  pub trait Mul<R> {
    type Output;
    fn mul<'b>(&self, rhs: &'b R) -> Self::Output;
  }
  
  impl<S1, S2, D> Mul<ndarray::ArrayBase<S2, D>> for ndarray::ArrayBase<S1, D> 
    where S1: ndarray::Data<Elem=f32>,
          S2: ndarray::Data<Elem=f32>,
          D: ndarray::Dimension,
          D::Pattern: std::fmt::Debug + PartialEq {
    type Output = ndarray::Array<f32, D>;
    fn mul<'b>(&self, rhs: &'b ndarray::ArrayBase<S2, D>) -> Self::Output {
      debug_assert_eq!(self.dim(), rhs.dim());
      let mut y = unsafe { Self::Output::uninitialized(self.dim()) };
      binary(
        emu(),
        "mul",
        y.as_slice_mut()
          .unwrap(),
        self.as_slice()
          .unwrap(),
        rhs.as_slice()
          .unwrap());
      y
    }
  }
}

#[cfg(test)]
mod tests {
  use super::emu;
  
  #[test]
  fn test_mul() {
    let x = ndarray::Array1::<f32>::from_elem(10, 2.);
    let y = ndarray::Array1::<f32>::from_elem(10, 4.);
    use super::opencl::Mul;
    assert_eq!(x.mul(&x), y, "{}", emu());
  }
}
