

pub mod opencl {
  pub trait Dot<R> {
    type Output;
    fn dot<'b>(&self, rhs: &'b R) -> Self::Output;
  }
  
  impl<S1, S2> Dot<ndarray::ArrayBase<S2, ndarray::Ix1>> for ndarray::ArrayBase<S1, ndarray::Ix2>
    where S1: ndarray::Data<Elem=f32>,
          S2: ndarray::Data<Elem=f32> {
    type Output = ndarray::Array1<f32>;
    fn dot<'b>(&self, rhs: &'b ndarray::ArrayBase<S2, ndarray::Ix1>) -> Self::Output {
      debug_assert_eq!(self.len(), rhs.len());
      let mut y = unsafe { Self::Output::uninitialized(1) };
      let pro_que = ocl::ProQue::builder()
        .src("")
        .dims(y.len())
        .build()
        .unwrap();
      let y_buffer = unsafe { pro_que.buffer_builder::<f32>()
        .use_host_slice(y.as_slice_mut().unwrap())
        .len(1)
        .build()
        .unwrap() };
      let x1_buffer = pro_que.buffer_builder::<f32>()
        .copy_host_slice(self.as_slice().unwrap())
        .build()
        .unwrap();
      let x2_buffer = pro_que.buffer_builder::<f32>()
        .copy_host_slice(rhs.as_slice().unwrap())
        .build()
        .unwrap();
      let kernel = pro_que.kernel_builder("vdot")
        .arg(&y_buffer)
        .arg(&x1_buffer)
        .arg(&x2_buffer)
        .build()
        .unwrap();
      unsafe { kernel.enq().unwrap(); }
      y_buffer.read(y.as_slice_mut().unwrap())
        .enq()
        .unwrap();  
      y
    }
  }
}
