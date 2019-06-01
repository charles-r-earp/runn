# runn
Provides Rust with useful functions for Neural Networks, with CUDA and OpenCL support.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/charles-r-earp/runn/LICENSE)

[![Build Status](https://travis-ci.com/charles-r-earp/runn.svg?branch=master)](https://travis-ci.com/charles-r-earp/runn)

## Status
In Early Developement

## Features
- native
  - dot (ndarray::ArrayBase) (passthrough to ndarray) 
    
Note: The native Exec currently doesn't have much purpose, it is meant to be a fallback when CUDA or OpenCL are not available. Code built on runn can then be executed on many different targets.

## Usage 

    \\Cargo.toml
    [dependencies]
    runn = { git = "https://github.com/charles-r-earp/runn", features=["native"] }
    
    \\main.rs
    use runn::{Executor, DotExec, native};
    use ndarray::{Array2, arr2};
    
    fn main() {
      let exec = native::Exec;
      println!("{}", exec);
      println!("{:?}", exec);
      let x: Array2<f32> = arr2(&[[1., 2.]]);
      let w: Array2<f32> = arr2(&[[3.],
                                  [4.]]); 
      let y = exec.dot(x, w);
      println!("{:?}", y);
    }
