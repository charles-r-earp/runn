pub mod relu;

pub(crate) mod opencl {
  pub(crate) fn scalar_func<'s, 'n, 'y, 'x, T>(src: &'s str, name: &'n str, y: &'y mut [T], x: &'x [T])
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
  pub(crate) fn scalar_func_inplace<'s, 'n, 'y, T>(src: &'s str, name: &'n str, y: &'y mut [T])
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
}
