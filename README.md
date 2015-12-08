# GPUMultiscaleCLEAN
Multi-scale clean for the GPU

This code is written by Mark Harris, unabashedly stolen from the ASKAP project, and 
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


