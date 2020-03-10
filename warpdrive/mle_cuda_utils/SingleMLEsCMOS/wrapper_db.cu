#include "GPUgaussMLEv2.h"

extern "C" void kernel_MLEFit_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim) 
{
	kernel_MLEFit<<<dimGrid, dimBlock>>>(d_data, PSFSigma, sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits,d_varim, d_gainim);
}

extern "C" void kernel_MLEFit_sigma_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim) 
{
	kernel_MLEFit_sigma<<<dimGrid, dimBlock>>>(d_data, PSFSigma, sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits,d_varim, d_gainim);
}

extern "C" void kernel_MLEFit_z_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma_x, float Ax, float Ay, float Bx, 
		float By, float gamma, float d, float PSFSigma_y, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim) 
{
	kernel_MLEFit_z<<<dimGrid, dimBlock>>>(d_data, PSFSigma_x, Ax, Ay, Bx, By, gamma, d, PSFSigma_y, sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits,d_varim);
}

extern "C" void kernel_MLEFit_sigmaxy_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim) 
{
	kernel_MLEFit_sigmaxy<<<dimGrid, dimBlock>>>(d_data, PSFSigma, sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits,d_varim);
}

extern "C" void kernel_MLEFit_pix_threads_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim, int calcCRB) 
{
//	//kernel_MLEFit<<<dimGrid, dimBlock>>>(d_data, PSFSigma, sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits,d_varim, d_gainim);
	kernel_MLEFit_pix_threads<<<dimGrid, dimBlock>>>(d_data, PSFSigma, sz, iterations, 
        d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim, d_gainim, calcCRB);
}

extern "C" void kernel_MLEFit_pix_threads_astig_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim, int calcCRB) 
{
//	//kernel_MLEFit<<<dimGrid, dimBlock>>>(d_data, PSFSigma, sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits,d_varim, d_gainim);
	kernel_MLEFit_pix_threads_astig<<<dimGrid, dimBlock>>>(d_data, PSFSigma, sz, iterations, 
        d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim, d_gainim, calcCRB);
}