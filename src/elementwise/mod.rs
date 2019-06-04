pub mod relu;
pub mod mul;

pub(crate) mod opencl {
  pub(crate) fn unary<'s, 'n, 'y, 'x, T>(src: &'s str, name: &'n str, y: &'y mut [T], x: &'x [T])
    where T: ocl::OclPrm {
    debug_assert_eq!(y.len(), x.len());
    let pro_que = ocl::ProQue::builder()
        .src(src)
        .dims(y.len())
        .build()
        .unwrap();
    let y_buffer = unsafe { pro_que.buffer_builder::<T>()
      .use_host_slice(y)
      .build()
      .unwrap() };
    let x_buffer = pro_que.buffer_builder::<T>()
      .copy_host_slice(x)
      .build()
      .unwrap();
    let kernel = pro_que.kernel_builder(name)
      .arg(&y_buffer)
      .arg(&x_buffer)
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    y_buffer.read(y)
      .enq()
      .unwrap();  
  }
  pub(crate) fn unary_inplace<'s, 'n, 'y, T>(src: &'s str, name: &'n str, y: &'y mut [T])
    where T: ocl::OclPrm {
    let pro_que = ocl::ProQue::builder()
        .src(src)
        .dims(y.len())
        .build()
        .unwrap();
    let y_buffer = unsafe { pro_que.buffer_builder::<T>()
      .use_host_slice(y)
      .build()
      .unwrap() };
    let kernel = pro_que.kernel_builder(name)
      .arg(&y_buffer)
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    y_buffer.read(y)
      .enq()
      .unwrap();  
  }
  pub(crate) fn binary<'s, 'n, 'y, 'x1, 'x2, T>(src: &'s str, name: &'n str, y: &'y mut [T], x1: &'x1 [T], x2: &'x2 [T])
    where T: ocl::OclPrm {
    debug_assert_eq!(y.len(), x1.len());
    debug_assert_eq!(x1.len(), x2.len());
    let pro_que = ocl::ProQue::builder()
        .src(src)
        .dims(y.len())
        .build()
        .unwrap();
    let y_buffer = unsafe { pro_que.buffer_builder::<T>()
      .use_host_slice(y)
      .build()
      .unwrap() };
    let x1_buffer = pro_que.buffer_builder::<T>()
      .copy_host_slice(x1)
      .build()
      .unwrap();
    let x2_buffer = pro_que.buffer_builder::<T>()
      .copy_host_slice(x2)
      .build()
      .unwrap();
    let kernel = pro_que.kernel_builder(name)
      .arg(&y_buffer)
      .arg(&x1_buffer)
      .arg(&x2_buffer)
      .build()
      .unwrap();
    unsafe { kernel.enq().unwrap(); }
    y_buffer.read(y)
      .enq()
      .unwrap();  
  }
}
