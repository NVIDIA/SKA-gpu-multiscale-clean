/// @copyright (c) 2011 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>

// Include own header file first
#include "MultiScaleCuda.h"

// System includes
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <stdio.h>

// Local includes
#include "Parameters.h"

#include <cub.cuh>

using namespace std;

// Some constants for findPeak
int findPeakNBlocks = 26;
static const int findPeakWidth = 1024;

struct Position {
    __host__ __device__
    Position(int _x, int _y) : x(_x), y(_y) { };
    int x;
    int y;
};

__host__
static void checkerror(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}
__host__
static void checkerror(cudaError_t err, int line, const char* filename)
{
    if (err != cudaSuccess)
    {
        std::cout << "CUDA Error (line "<<line<<" of "<<filename<<": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__host__ __device__ inline
static Position idxToPos(const size_t idx, const int width)
{
    const int y = idx / width;
    const int x = idx % width;
    return Position(x, y);
}

__host__ __device__ inline
static size_t posToIdx(const int width, const Position& pos)
{
    return (pos.y * width) + pos.x;
}

// For CUB
struct MaxOp
{
    __host__ __device__ inline
    Peak operator()(const Peak &a, const Peak &b)
    {
        return (abs(b.val) > abs(a.val)) ? b : a;
    }
};

__global__ 
void d_findPeak(Peak *peaks, const float* __restrict__ image, int size)
{
    Peak threadMax = {0.0, 0};   
        
    // parallel raking reduction (independent threads)
    for (int i = findPeakWidth * blockIdx.x + threadIdx.x; 
        i < size; 
        i += gridDim.x * findPeakWidth) {
        if (abs(image[i]) > abs(threadMax.val)) {
            threadMax.val = image[i];
            threadMax.pos = i;
        }
    }

    // Use CUB to find the max for each thread block.
    typedef cub::BlockReduce<Peak, findPeakWidth> BlockMax;
    __shared__ typename BlockMax::TempStorage temp_storage;
    threadMax = BlockMax(temp_storage).Reduce(threadMax, MaxOp());

    if (threadIdx.x == 0) peaks[blockIdx.x] = threadMax;
}

__host__
static Peak findPeak(Peak *d_peaks, const float* d_image, size_t size)
{
    // Find peak
    d_findPeak<<<findPeakNBlocks, findPeakWidth>>>(d_peaks, d_image, size);   
    
    // Get the peaks array back from the device
    Peak peaks[findPeakNBlocks];
    cudaError_t err = cudaMemcpy(peaks, d_peaks, findPeakNBlocks * sizeof(Peak), cudaMemcpyDeviceToHost);
    checkerror(err, __LINE__, __FILE__);
    
    Peak p = peaks[0];
    // serial final reduction
    for (int i = 1; i < findPeakNBlocks; ++i) {
        if (abs(peaks[i].val) > abs(p.val))
            p = peaks[i];
    }

    return p;
}

__global__
void d_subtractPSF(const float* __restrict__ d_psf,
    const int psfWidth,
    float* d_residual,
    const int residualWidth,
    const int startx, const int starty,
    int const stopx, const int stopy,
    const int diffx, const int diffy,
    const float absPeakVal, const float gain)
{   
    const int x =  startx + threadIdx.x + (blockIdx.x * blockDim.x);
    const int y =  starty + threadIdx.y + (blockIdx.y * blockDim.y);

    // Because workload is not always a multiple of thread block size, 
    // need to ensure only threads in the work area actually do work
    if (x <= stopx && y <= stopy) {
        d_residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
            * d_psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
    }
}

__host__
static void subtractPSF(const float* d_psf, const int psfWidth,
        float* d_residual, const int residualWidth,
        const size_t peakPos, const size_t psfPeakPos,
        const float absPeakVal, const float gain)
{  
    // The x,y coordinate of the peak in the residual image
    const int rx = idxToPos(peakPos, residualWidth).x;
    const int ry = idxToPos(peakPos, residualWidth).y;

    // The x,y coordinate for the peak of the PSF (usually the centre)
    const int px = idxToPos(psfPeakPos, psfWidth).x;
    const int py = idxToPos(psfPeakPos, psfWidth).y;

    // The PSF needs to be overlayed on the residual image at the position
    // where the peaks align. This is the offset between the above two points
    const int diffx = rx - px;
    const int diffy = ry - py;

    // The top-left-corner of the region of the residual to subtract from.
    // This will either be the top right corner of the PSF too, or on an edge
    // in the case the PSF spills outside of the residual image
    const int startx = max(0, rx - px);
    const int starty = max(0, ry - py);

    // This is the bottom-right corner of the region of the residual to
    // subtract from.
    const int stopx = min(residualWidth - 1, rx + (psfWidth - px - 1));
    const int stopy = min(residualWidth - 1, ry + (psfWidth - py - 1));

    const dim3 blockDim(32, 4);

    // Note: Both start* and stop* locations are inclusive.
    const int blocksx = ceil((stopx-startx+1.0f) / static_cast<float>(blockDim.x));
    const int blocksy = ceil((stopy-starty+1.0f) / static_cast<float>(blockDim.y));

    dim3 gridDim(blocksx, blocksy);

    d_subtractPSF<<<gridDim, blockDim>>>(d_psf, psfWidth, d_residual, residualWidth,
        startx, starty, stopx, stopy, diffx, diffy, absPeakVal, gain);
    cudaError_t err = cudaGetLastError();
    checkerror(err);
}

__host__
MultiScaleCuda::MultiScaleCuda(size_t psfSize, size_t n_scale_in, size_t residualSize)
{
    reportDevice();

    n_scale = n_scale_in;
    cudaError_t err;
    err = cudaMalloc((void **) &d_psf_all, psfSize * sizeof(float) * n_scale);
    checkerror(err);
    err = cudaMalloc((void **) &d_residual_all, residualSize * sizeof(float) * n_scale);
    checkerror(err);
    err = cudaMalloc((void **) &d_model, residualSize * sizeof(float));
    checkerror(err);
    err = cudaMalloc((void **) &d_peaks_all, findPeakNBlocks * sizeof(Peak) * n_scale);
    checkerror(err);
    //TODO use less memory by keeping only the upper diagonal
    err = cudaMalloc((void **) &d_cross_all, psfSize * sizeof(float) * n_scale*n_scale);
    checkerror(err);
    d_psf = (float**)malloc(sizeof(float*)*n_scale);
    d_residual = (float**)malloc(sizeof(float*)*n_scale);
    d_peaks = (Peak**)malloc(sizeof(Peak*)*n_scale);
    d_cross = (float***)malloc(sizeof(float**)*n_scale);
    for (size_t s=0;s<n_scale;s++) {
       d_psf[s] = d_psf_all+s*psfSize;
       d_residual[s] = d_residual_all+s*residualSize;
       d_peaks[s] = d_peaks_all+s*findPeakNBlocks;
       d_cross[s] = (float**)malloc(sizeof(float*)*n_scale);
       //TODO Here we can just rearrange the pointers to point only to the
       //     upper diag
       for (size_t ss=0;ss<n_scale;ss++) {
          d_cross[s][ss] = d_cross_all + (s*n_scale+ss)*psfSize;
       }
    }
}

__host__
MultiScaleCuda::~MultiScaleCuda()
{
    // Free device memory
    cudaFree(d_psf_all);
    cudaFree(d_residual_all);
    cudaFree(d_model);
    cudaFree(d_peaks_all);
    cudaFree(d_cross_all);
    free(d_psf);
    free(d_residual);
    free(d_peaks);
    for(size_t s=0;s<n_scale;s++) {
      free(d_cross[s]);
    }
    free(d_cross);
}
cudaError_t cudaMemcpyWrap(void* pout, const void* pin, size_t sz, cudaMemcpyKind type) {
    cout << "memcpy " << pin << " to " << pout << endl;
    cout.flush();
    return cudaMemcpy(pout, pin, sz, type);
}

__host__
void MultiScaleCuda::deconvolve(const vector<float>& dirty,
        const size_t dirtyWidth,
        const vector<float>* psf,
        const size_t psfWidth,
        const vector<float>* cross,
        vector<float>& model,
        vector<float>* residual)
{
    cudaError_t err;

    // Copy host vectors to device arrays
    for (size_t s=0;s<n_scale;s++) {
       err = cudaMemcpy(d_psf[s], &psf[s][0], psf[0].size() * sizeof(float), cudaMemcpyHostToDevice);
       checkerror(err, __LINE__, __FILE__);
       err = cudaMemcpy(d_residual[s], &dirty[0], residual[0].size() * sizeof(float), cudaMemcpyHostToDevice);
       checkerror(err, __LINE__, __FILE__);
       for (size_t ss=0;ss<n_scale;ss++) {
          err = cudaMemcpy(d_cross[s][ss], &cross[s*n_scale+ss][0], psf[0].size() * sizeof(float), cudaMemcpyHostToDevice);
          checkerror(err, __LINE__, __FILE__);
       }
    }
 
    //Zero the model
    cudaMemset(d_model, 0, model.size() * sizeof(float));

    // Find peak of PSF
    Peak *psfPeak = new Peak[n_scale];
    for (size_t s=0;s<n_scale;s++) {
      psfPeak[s] = findPeak(d_peaks[s], d_psf[s], psf[s].size());

      cout << "Found peak of PSF: " << "Maximum = " << psfPeak[s].val 
        << " at location " << idxToPos(psfPeak[s].pos, psfWidth).x << ","
        << idxToPos(psfPeak[s].pos, psfWidth).y << " for scale " << s << endl;
      assert(psfPeak[s].pos <= psf[s].size());
    }

    for (unsigned int i = 0; i < g_niters; ++i) {
        // Find peak in the residual image
        Peak absPeak;
        absPeak.val=-INT_MAX;
#if 1
        for (size_t s=0;s<n_scale;s++) {
           //TODO multiply by scale-dependent scale factor
           //TODO think of synonym for "scale", reword preceeding nonsense
           Peak thisPeak = findPeak(d_peaks[s], d_residual[s], residual[0].size());
           if (thisPeak > absPeak) absPeak=thisPeak;
        }
#else
        //TODO We can do these searches in one step if we scale each residual by the factor
        //     We will also need to un-scale the absPeak.val as we add it to the model and
        //     re-scale for each new component as we subtract
        absPeak = findPeak(d_peaks[0], d_residual[0], n_scale*residual[0].size());
        absPeak.pos %= residual[0].size();
#endif

        assert(absPeak.pos <= residual[0].size());
        //cout << "Iteration: " << i + 1 << " - Maximum = " << peak.val
        //    << " at location " << idxToPos(peak.pos, dirtyWidth).x << ","
        //    << idxToPos(peak.pos, dirtyWidth).y << endl;


        // Check if threshold has been reached
        if (abs(absPeak.val) < g_threshold) {
            cout << "Reached stopping threshold" << endl;
            break;
        }

        // Subtract the PSF from the residual image (this function will launch
        // an kernel asynchronously, need to sync later
        for (size_t s=0; s<n_scale; s++) {
           //TODO Do we need to find a center for d_cross?
           subtractPSF(d_cross[absPeak.scale][s], psfWidth, d_residual[s], dirtyWidth, absPeak.pos, psfPeak[s].pos, absPeak.val, g_gain);
        }

        // Add to model
        //Note, the minus sign in front of -absPeak.val makes this an addition instead
        subtractPSF(d_psf[absPeak.scale], psfWidth, d_model, dirtyWidth, absPeak.pos, psfPeak[absPeak.scale].pos, 
                    -absPeak.val, g_gain);
    }

    // Copy device array back into the host vector
    for (size_t s=0;s<n_scale;s++) {
       err = cudaMemcpy(&residual[s][0], d_residual[s], residual[0].size() * sizeof(float), cudaMemcpyDeviceToHost);
       checkerror(err);
    }
    err = cudaMemcpy(&model[0], d_model, model.size() * sizeof(float), cudaMemcpyDeviceToHost);
    checkerror(err);
    delete []psfPeak;
}

__host__
void MultiScaleCuda::reportDevice(void)
{
    // Report the type of device being used
    int device;
    cudaDeviceProp devprop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devprop, device);
    std::cout << "    Using CUDA Device " << device << ": "
        << devprop.name << std::endl;

    // Allocate 2 blocks per multiprocessor
    findPeakNBlocks = 2 * devprop.multiProcessorCount;
}
