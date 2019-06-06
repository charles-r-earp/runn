use runn::{Backend, Shape, Tensor, HostBuffer, Net, activation::Sigmoid};
use timeit::*;

fn main() {

  let mut net = Net::new(Sigmoid::new());

  let backend = Backend::new();
  
  let x = vec![1.5; 1024];
  
  println!("Device...");
  timeit!({
    let xs = vec![Tensor::new(HostBuffer::F32(x.clone()), Shape::new([1024]))]; 
    net.eval(&backend, xs); 
  });
  
  println!("Host...");
  timeit!({
    x.iter()
      .map(|&x| 1. / (1. + (-x).exp()))
      .collect::<Vec<f32>>();
  });
}

  
