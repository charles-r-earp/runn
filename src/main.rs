use runn::{Backend, Tensor};
use timeit::*;

fn main() {
  Backend::create();
  let x = ndarray::Array1::<f32>::from_vec(vec![10.; 64*64]);
  println!("opencl... no op");
  timeit!({
    Tensor::from(x.view())
     // .sigmoid()
      .into_array();
  });
  println!("opencl...");
  timeit!({
    Tensor::from(x.view())
      .sigmoid()
      .into_array();
  });
  println!("native...");
  timeit!({
    runn::activation::array_sigmoid(x.view());
  });
}
  
