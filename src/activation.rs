use crate::{Backend, Tensor};

em::emu!{
  sigmoid(global_y [f32], global_x [f32]) {
    let gid: i32 = get_global_id(0);
    global_y[gid] = 1. / (1. + exp(-global_x[gid]));
  }
} 

pub fn emu() -> String { EMU.to_string() }

impl<D: ndarray::Dimension> Tensor<f32, D> {
  pub fn sigmoid(&self) -> Tensor<f32, D> {
    let y = Tensor::new::<ndarray::Shape<D>>(self.shape.clone());
    let kernel = Backend::pro_que()
      .kernel_builder("sigmoid")
      .arg(y.buffer())
      .arg(self.buffer())
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap() }
    y
  }
}

pub fn array_sigmoid<S: ndarray::Data<Elem=f32>, D: ndarray::Dimension>(array: ndarray::ArrayBase<S, D>) -> ndarray::Array<f32, D> {
  use std::ops::Neg;
  use ndarray::ShapeBuilder;
  unsafe { ndarray::Array::<f32, D>::from_shape_vec_unchecked::<D>(
    array.raw_dim(), 
    array.iter()
      .map(|&x| 1. / (1. + x.neg().exp()))
      .collect::<Vec<f32>>()
  ) }
} 


#[cfg(test)]
mod tests {
  use crate::{Backend, Tensor};
  
  #[test]
  fn test_sigmoid() {
    Backend::create();
    let x = ndarray::Array1::from_vec(vec![0., 1., -1., 10., 100.]);
    let y = Tensor::from(x.clone())
      .sigmoid()
      .into_array();
    assert_eq!(y, super::array_sigmoid(x));
  } 
}
