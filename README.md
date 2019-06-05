# runn
Provides a core implementation layer for Neural Networks and other computational tasks. 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/charles-r-earp/runn/LICENSE)

[![Build Status](https://travis-ci.com/charles-r-earp/runn.svg?branch=master)](https://travis-ci.com/charles-r-earp/runn)

# Status
Experimental!!! Focusing on a proof of concept, api is not stable.

Currently can setup and execute a Sigmoid op. Time is only lost copying the data into the Tensor, writing it to the device, and (possibly?) reconstructing the array. The break even point is about 100k on an NVIDIA GTX 1060 GPU. With a suffienctly long chain of ops, even small ones, this should be reduced.  

Multithreading on the host side seems to be a no-go, run the tests with:

    cargo test -- --test-threads 1
    
This will need to be addressed, with Arc or some other atomic structure. The backend is a static mut, which just gets copied to secondary threads. 

# Requirements 
Install OpenCL such that clinfo lists a device. 

# Goal
Use emu to write OpenCL kernels, that are namespaced and compiled into a single binary by ocl, for one (or more) devices. A function can be evaluated with a set of inputs, entirely on the device, only writing the inputs and reading the outputs, all intermediate values are only allocated on the device. Compiling once, and minimizing reads and writes is critical to performance. 



