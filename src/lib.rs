pub mod activation;
use std::ops;

pub fn emu() -> String { activation::emu() }

pub struct Tensor<T: ocl::OclPrm, D: ndarray::Dimension> {
  shape: ndarray::Shape<D>,
  data: Vec<T>,
  buffer: ocl::Buffer<T>
}

impl<T: ocl::OclPrm + Copy, S: ndarray::Data<Elem=T>, D: ndarray::Dimension> From<ndarray::ArrayBase<S, D>> for Tensor<T, D> {
  #[inline]
  fn from(array: ndarray::ArrayBase<S, D>) -> Self {
    use ndarray::ShapeBuilder;
    let shape = array.raw_dim().into_shape();
    let mut data = Vec::with_capacity(array.len());
    unsafe { data.set_len(array.len()); }
    data.copy_from_slice(array.as_slice().unwrap());
    let ref mut pro_que = Backend::pro_que();
    pro_que.set_dims(data.len());
    let mut buffer = pro_que
      .buffer_builder()
      .copy_host_slice(&data) 
      .build()
      .unwrap();
    Self{shape, data, buffer}
  }
}

impl<T: ocl::OclPrm, D: ndarray::Dimension> Tensor<T, D> {
  #[inline]
  fn new<S: ndarray::ShapeBuilder<Dim=D>>(shape: ndarray::Shape<D>) -> Self {
    use ndarray::ShapeBuilder;
    let shape = shape.into_shape();
    let mut buffer = Backend::pro_que()
      .buffer_builder()
      .len(shape.size()) 
      .build()
      .unwrap();
    Self{shape, data: Vec::new(), buffer}
  }
  #[inline]
  pub fn into_array(self) -> ndarray::Array<T, D> {
    let mut array = unsafe { ndarray::Array::uninitialized(self.shape) };
    self.buffer
      .read(array.as_slice_mut().unwrap())
      .enq()
      .unwrap();
    array
  }
  #[inline]
  pub fn buffer(&self) -> &ocl::Buffer<T> {
    &self.buffer
  }
}

pub struct Backend {
  pro_que: ocl::ProQue,
}

static mut _backend: Option<Backend> = None;

impl Backend {
  #[inline]
  pub fn create() {
    if unsafe { _backend.is_none() } {
      let pro_que = ocl::builders::ProQueBuilder::new()
        .src(emu())
        .build()
        .unwrap();
      unsafe { _backend = Some(Self{pro_que}); }
    }
  }
  #[inline]
  pub fn pro_que() -> &'static mut ocl::ProQue {
    unsafe {
      &mut _backend.as_mut()
        .unwrap()
        .pro_que
    }
  }
}

#[cfg(test)]
mod tests {
  use super::{Backend, Tensor};
  
  #[test]
  fn test_backend() {
    Backend::create();
  }
  
  #[test]
  fn test_tensor() {
    Backend::create();
    let x = ndarray::Array1::<f32>::ones(1000);
    let y = Tensor::from(x.clone())
      .into_array();
    assert_eq!(y, x);
  }
  
  #[test]
  fn test_sequential() {
    use super::activation::array_sigmoid;
    Backend::create();
    let x = ndarray::Array1::<f32>::ones(1000);
    let y = Tensor::from(x.clone())
      .sigmoid()
      .sigmoid()
      .into_array();
    assert_eq!(y, array_sigmoid(array_sigmoid(x)));
  }
}
  
      


