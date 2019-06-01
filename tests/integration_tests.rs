#[cfg(feature="native")]
mod native_tests {
  use runn::native::Exec;
  
  #[test]
  fn dot_array2_f32() {
    use runn::DotExec;
    use ndarray::{Array2, arr2};
    let x: Array2<f32> = arr2(&[[1., -2., 3., -4.],
                                [5., -6., 7., -8.]]);
    let w: Array2<f32> = arr2(&[[-1., -2.],
                                [ -3., -4.],
                                [5., 6.],
                                [7., 8.]]);
    assert_eq!(Exec.dot(x.view(), w.view()), x.dot(&w));
  }
}
