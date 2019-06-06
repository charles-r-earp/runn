# runn
Provides a core implementation layer for Neural Networks and other computational tasks. 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/charles-r-earp/runn/LICENSE)

[![Build Status](https://travis-ci.com/charles-r-earp/runn.svg?branch=master)](https://travis-ci.com/charles-r-earp/runn)

# Status
Experimental!!! Rebuilt several times.

Currently can create a net with a sigmoid activation function for f32. The backend is created once, which compiles the opencl code. Then a net can be constructed, and fed a vec of Tensor inputs. This is then loaded onto the device, then the layers of the net push their kernels into a Vec, which then is enqueued and processed all at once. Then the outputs are copied back to host Tensors and returned. 

# Requirements 
Install OpenCL such that clinfo lists your device (gpu or cpu).





