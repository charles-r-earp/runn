# runn
Useful functions for Neural Networks for CPU / GPU targets.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/charles-r-earp/runn/LICENSE)

[![Build Status](https://travis-ci.com/charles-r-earp/runn.svg?branch=master)](https://travis-ci.com/charles-r-earp/runn)

# Requirements 
Install OpenCL such that clinfo lists a device. Currently the first platform -> first device is selected. 

# Features 
Currently emu only supports OpenCL and f32 is the only floating point type. Traits are implemented on ndarray::ArrayBase.

- Activations
  - Relu (+Inplace, +Grad)


