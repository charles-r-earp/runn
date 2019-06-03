em::emu!{
  relu(global_y [f32], global_x [f32]) {
    if global_x[get_global_id(0)] > 0. { 
      global_y[get_global_id(0)] = global_x[get_global_id(0)]; 
    }
    else { 
      global_y[get_global_id(0)] = 0.;
    } 
  }
  relu_inplace(global_y [f32]) {
    if global_y[get_global_id(0)] < 0. {
      global_y[get_global_id(0)] = 0.;
    }
  }
}

pub fn emu() -> &'static str { EMU }

pub mod opencl {
  use super::emu;
  use super::super::opencl::{scalar_func, scalar_func_inplace};
  
  pub trait Relu {
    type Output;
    fn relu(&self) -> Self::Output;
  }
  
  pub trait ReluInplace {
    fn relu_inplace(&mut self);
  }
  
  impl<S, D> Relu for ndarray::ArrayBase<S, D>
    where S: ndarray::Data<Elem=f32>,
          D: ndarray::Dimension {
    type Output = ndarray::Array<f32, D>;
    fn relu(&self) -> Self::Output {
      let mut y = unsafe { Self::Output::uninitialized(self.dim()) };
      scalar_func(
        emu(), 
        "relu", 
        y.as_slice_mut()
          .unwrap(), 
        self.as_slice()
          .unwrap());
      y
    }
  }
  
  impl<S, D> ReluInplace for ndarray::ArrayBase<S, D>
    where S: ndarray::DataMut<Elem=f32>,
          D: ndarray::Dimension {
    fn relu_inplace(&mut self) {
      scalar_func_inplace(
        emu(), 
        "relu_inplace", 
        self.as_slice_mut()
          .unwrap());
    }
  }
  
  #[cfg(test)] 
  mod tests {
    use super::emu;
    #[test]
    fn relu() {
      use super::Relu;
      let x: ndarray::Array2<f32> = ndarray::arr2(&[[0., -1., 1.]]);
      let y = ndarray::arr2(&[[0., 0., 1.]]);
      assert_eq!(x.relu(), y, "{}", emu());
    }
    #[test]
    fn relu_inplace() {
      use super::ReluInplace;
      let mut x: ndarray::Array2<f32> = ndarray::arr2(&[[0., -1., 1.]]);
      x.relu_inplace();
      assert_eq!(x, ndarray::arr2(&[[0., 0., 1.]]), "{}", emu());
    }
  }
}



