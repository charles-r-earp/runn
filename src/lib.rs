pub mod activation;
use std::iter;

pub fn emu() -> String { activation::emu() }

pub struct Backend {
  context: ocl::Context,
  program: ocl::Program,
}

impl Backend {
  #[inline]
  pub fn new() -> Self {
    let context = ocl::builders::ContextBuilder::new()
      .build()
      .unwrap();
    let program = ocl::builders::ProgramBuilder::new()
        .src(emu())
        .build(&context)
        .unwrap();
    Self{context, program}
  }
  #[inline]
  pub fn program(&self) -> &ocl::Program { &self.program } 
  #[inline]
  pub fn context(&self) -> &ocl::Context { &self.context }
  #[inline]
  pub fn queue(&self) -> ocl::Queue {
    ocl::Queue::new(
      &self.context,
      self.context.devices()[0],
      Some(ocl::flags::CommandQueueProperties::new())
    ).unwrap()
  }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Shape {
  dims: Vec<usize>
}

impl Shape {
  pub fn new(s: impl AsRef<[usize]>) -> Self {
    Self{dims: s.as_ref().to_vec()}
  }
  pub fn size(&self) -> usize {
    self.dims.iter().product::<usize>()
  }
}

pub trait Buffer {
  fn len(&self) -> usize;
}

#[derive(Debug, Clone, PartialEq)]
pub enum HostBuffer {
  F32(Vec<f32>),
  I32(Vec<i32>)
}

impl Buffer for HostBuffer {
  #[inline]
  fn len(&self) -> usize {
    use HostBuffer::*; 
    match self {
      F32(v) => v.len(),
      I32(v) => v.len()
    }
  }
}

pub enum DeviceBuffer {
  F32(ocl::Buffer<f32>),
  I32(ocl::Buffer<i32>)
}

impl Buffer for DeviceBuffer {
  #[inline]
  fn len(&self) -> usize {
    use DeviceBuffer::*; 
    match self {
      F32(v) => v.len(),
      I32(v) => v.len()
    }
  }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<B> {
  buffer: B,
  shape: Shape
}

impl<B> Tensor<B> {
  #[inline]
  pub fn new(buffer: B, shape: Shape) -> Self
    where B: Buffer {
    debug_assert_eq!(buffer.len(), shape.size());
    Self{buffer, shape}
  }
  #[inline]
  pub fn buffer(&self) -> &B {
    &self.buffer
  }
  #[inline]
  pub fn shape(&self) -> &Shape {
    &self.shape
  }
}

impl Tensor<HostBuffer> {
  #[inline]
  fn to_device<'b>(&self, backend: &'b Backend) -> Tensor<DeviceBuffer> {
    match &self.buffer {
      HostBuffer::F32(v) => {
        let buffer = ocl::builders::BufferBuilder::new()
          .queue(backend.queue())
          .len(v.len())
          .build()
          .unwrap();
        buffer.write(&*v)
          .enq()
          .unwrap();
        Tensor::new(DeviceBuffer::F32(buffer), self.shape.clone())
      },
      HostBuffer::I32(v) => {
        let buffer = ocl::builders::BufferBuilder::new()
          .queue(backend.queue())
          .len(v.len())
          .build()
          .unwrap();
        buffer.write(&*v)
          .enq()
          .unwrap();
        Tensor::new(DeviceBuffer::I32(buffer), self.shape.clone())
      },
    }
  }
}

impl Tensor<DeviceBuffer> {
  #[inline]
  fn to_host(&self) -> Tensor<HostBuffer> {
    match &self.buffer {
      DeviceBuffer::F32(buffer) => {
        let mut v = Vec::<f32>::with_capacity(buffer.len());
        unsafe { v.set_len(v.capacity()); }
        buffer.read(&mut v)
          .enq()
          .unwrap();
        Tensor::new(HostBuffer::F32(v), self.shape.clone())
      },
      DeviceBuffer::I32(buffer) => {
        let mut v = Vec::<i32>::with_capacity(buffer.len());
        unsafe { v.set_len(v.capacity()); }
        buffer.read(&mut v)
          .enq()
          .unwrap();
        Tensor::new(HostBuffer::I32(v), self.shape.clone())
      }
    }
  }
}

pub trait Layer: 'static {
  fn eval<'b, 'x>(&mut self, backend: &'b Backend, kernels: &mut Vec<ocl::Kernel>, xs: Vec<&'x Tensor<DeviceBuffer>>) -> Vec<&Tensor<DeviceBuffer>>;
}
  
pub struct Net {
  layer: Box<Layer>
} 

impl Net {
  #[inline]
  pub fn new(layer: Box<Layer>) -> Self { Self{layer} }
  #[inline]
  pub fn eval<'b>(&mut self, backend: &'b Backend, host_xs: Vec<Tensor<HostBuffer>>) -> Vec<Tensor<HostBuffer>> {
    use iter::FromIterator;
    let device_xs = Vec::from_iter(host_xs.iter()
      .map(|x| x.to_device(backend)));
    let device_xs_ref: Vec<&Tensor<DeviceBuffer>> = Vec::from_iter(device_xs.iter()
      .map(|x| x));
    let mut kernels = Vec::new();
    let device_ys = (*self.layer)
      .eval(backend, &mut kernels, device_xs_ref);
    kernels.into_iter()
      .for_each(|k| unsafe { k.enq().unwrap(); });
    Vec::from_iter(device_ys.iter()
      .map(|y| y.to_host()))
  }
}
  
#[cfg(test)]
mod tests {
  use super::{Backend, Shape, Tensor, HostBuffer};
  
  #[test]
  fn test_backend() {
    let backend = Backend::new();
  }
  
  #[test]
  fn test_tensor_f32() {
    let backend = Backend::new();
    let x = Tensor::new(HostBuffer::F32(vec![1.; 100]), Shape::new([100]));
    let y = x.to_device(&backend).to_host();
    assert_eq!(x, y);
  }
}
  
      


