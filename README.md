# GPUMultiscaleCLEAN
Multi-scale clean for the GPU and/or CPU

This code is written by Mark Harris, based on work from the ASKAP project, and 
modified for multi-scale by Levi Barnes. The original code can be found here:
https://github.com/askap-benchmarks/

This code requires CUB. It can be found here:
https://github.com/NVlabs/CUB
Specify CUB_HOME in the Makefile to build.

Execution of the tMultiscaleClean and tHogbomClean benchmarks will require the existence of 
the point spread function (PSF) image and the dirty image (the image to be cleaned) the 
working directory. These can be downloaded from here:

* http://www.atnf.csiro.au/people/Ben.Humphreys/dirty.img
* http://www.atnf.csiro.au/people/Ben.Humphreys/psf.img


Build

To build GPU version, specify CUDA=1 on the make line  

	> make CUDA=1

To run with GPU-only (i.e. without error checking), specify skipgolden on the command line

	> ./tMultiScaleClean skipgolden
