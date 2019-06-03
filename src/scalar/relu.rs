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
  relu_grad(global_y [f32], global_x [f32]) {
    if global_x[get_global_id(0)] > 0. { 
      global_y[get_global_id(0)] = 1.; 
    }
    else { 
      global_y[get_global_id(0)] = 0.;
    }
  }
  relu_inplace_grad(global_y [f32]) {
    if global_y[get_global_id(0)] > 0. {
      global_y[get_global_id(0)] = 1.;
    }
    else {
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
    fn relu_grad(&self) -> Self::Output;
  }
  
  pub trait ReluInplace {
    fn relu_inplace(&mut self);
    fn relu_grad_inplace(&mut self);
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
    fn relu_grad(&self) -> Self::Output {
      let mut dy = unsafe { Self::Output::uninitialized(self.dim()) };
      scalar_func(
        emu(), 
        "relu_grad", 
        dy.as_slice_mut()
          .unwrap(), 
        self.as_slice()
          .unwrap());
      dy
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
    fn relu_grad_inplace(&mut self) {
      scalar_func_inplace(
        emu(), 
        "relu_inplace_grad", 
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
    fn relu_grad() {
      use super::Relu;
      let x: ndarray::Array2<f32> = ndarray::arr2(&[[0., -0.5, 1.5]]);
      let dy = ndarray::arr2(&[[0., 0., 1.]]);
      assert_eq!(x.relu_grad(), dy, "{}", emu());
    }
    #[test]
    fn relu_inplace() {
      use super::ReluInplace;
      let mut y: ndarray::Array2<f32> = ndarray::arr2(&[[0., -1., 1.]]);
      y.relu_inplace();
      assert_eq!(y, ndarray::arr2(&[[0., 0., 1.]]), "{}", emu());
    }
    #[test]
    fn relu_grad_inplace() {
      use super::ReluInplace;
      let mut y: ndarray::Array2<f32> = ndarray::arr2(&[[0., -0.5, 1.5]]);
      y.relu_grad_inplace();
      assert_eq!(y, ndarray::arr2(&[[0., 0., 1.]]), "{}", emu());
    }
  }
}



