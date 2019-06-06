use crate::{Backend, Shape, Tensor, DeviceBuffer, Layer};

em::emu!{
  sigmoid_f32(global_y [f32], global_x [f32]) {
    let gid: i32 = get_global_id(0);
    global_y[gid] = 1. / (1. + exp(-global_x[gid]));
  }
} 

pub fn emu() -> String { EMU.to_string() }

pub struct Sigmoid {
  y: Option<Tensor<DeviceBuffer>>
}

impl Sigmoid {
  #[inline]
  pub fn new() -> Box<Self> { Box::new(Self{y: None}) }
}

impl Layer for Sigmoid {
  #[inline]
  fn eval<'b, 'x>(&mut self, backend: &'b Backend, kernels: &mut Vec<ocl::Kernel>, xs: Vec<&'x Tensor<DeviceBuffer>>) -> Vec<&Tensor<DeviceBuffer>> {
    debug_assert_eq!(xs.len(), 1);
    let ref x = xs[0];
    let shape = x.shape().clone();
    self.y = match x.buffer() {
      DeviceBuffer::F32(x_buffer) => {
        let queue = x_buffer.default_queue().unwrap().clone();
        let y_buffer = ocl::builders::BufferBuilder::new()
          .queue(queue.clone())
          .len(x_buffer.len())
          .build()
          .unwrap();
        let kernel = ocl::builders::KernelBuilder::new()
          .program(backend.program())
          .name("sigmoid_f32")
          .queue(queue)
          .global_work_size(x_buffer.len())
          .arg(&y_buffer)
          .arg(x_buffer)
          .build()
          .unwrap();
        kernels.push(kernel);
        Some(Tensor::new(DeviceBuffer::F32(y_buffer), shape))
      },
      DeviceBuffer::I32(_) => { unimplemented!(); }
    };
    vec![self.y.as_ref().unwrap()]
    
  }
}

#[cfg(test)]
mod tests {
  use crate::{Backend, Shape, Tensor, HostBuffer, Net};
  use super::Sigmoid;
  
  #[test]
  fn test_sigmoid_f32() {
    let mut net = Net::new(Sigmoid::new());
    let backend = Backend::new();
    let x = vec![1.5; 1024];
    let xs = vec![Tensor::new(HostBuffer::F32(x.clone()), Shape::new([x.len()]))]; 
    let a = net.eval(&backend, xs); 
    let v = x.iter()
        .map(|&x| 1. / (1. + (-x).exp()))
        .collect::<Vec<f32>>();
    let b = vec![Tensor::new(HostBuffer::F32(v), Shape::new([x.len()]))];
    assert_eq!(a, b);
  } 
}

