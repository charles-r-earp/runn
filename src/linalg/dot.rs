em::emu!{
  vdot_partial(local_y [f32], global_x1 [f32], global_x2 [f32]) {
    local_y[get_local_id(0)] = global_x1[get_global_id(0)] * global_x2[get_global_id(0)];
  }
  vdot_sum(global_y, global_
}

pub fn emu() -> &'static str { EMU }

pub mod opencl {
  use super::emu;
  
  pub trait Dot<R> {
    type Output;
    fn dot<'b>(&self, rhs: &'b R) -> Self::Output;
  }
  
  impl<S1, S2> Dot<ndarray::ArrayBase<S2, ndarray::Ix1>> for ndarray::ArrayBase<S1, ndarray::Ix1>
    where S1: ndarray::Data<Elem=f32>,
          S2: ndarray::Data<Elem=f32> {
    type Output = ndarray::Array1<f32>;
    fn dot<'b>(&self, rhs: &'b ndarray::ArrayBase<S2, ndarray::Ix1>) -> Self::Output {
      debug_assert_eq!(self.len(), rhs.len());
      let mut y = Self::Output::zeros(self.len());
      let pro_que = ocl::ProQue::builder()
        .src(emu())
        .dims(y.len())
        .build()
        .unwrap();
      let y_buffer = unsafe { pro_que.buffer_builder::<f32>()
        .use_host_slice(y.as_slice_mut().unwrap())
        .len(self.len())
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
      let kernel = pro_que.kernel_builder("mul")
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
  
  #[cfg(test)] 
  mod tests {
    use super::emu;
    #[test]
    fn vdot() {
      use super::Dot;
      let x1: ndarray::Array1<f32> = ndarray::arr1(&[1., 2., 3.]);
      let x2: ndarray::Array1<f32> = ndarray::arr1(&[4., 5., 6.]);
      let res = super::Dot::dot(&x1, &x2);
      //assert!(res.len() == 1);
      let y: ndarray::Array1<f32> = ndarray::arr1(&[1.*4. + 2.*10. + 3.*6.]);
      assert_eq!(res, y, "{}", emu());
    }
  }
}
