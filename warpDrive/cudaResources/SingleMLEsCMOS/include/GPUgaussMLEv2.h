#ifndef GPUGAUSSMLEV2_H
#define GPUGAUSSMLEV2_H

__global__ void kernel_MLEFit(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim);

__global__ void kernel_MLEFit_pix_threads(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim, int calcCRB);

__global__ void kernel_MLEFit_pix_threads_astig(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim, int calcCRB);

__global__ void kernel_MLEFit_sigma(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim);

__global__ void kernel_MLEFit_z(float *d_data, float PSFSigma_x, float Ax, float Ay, float Bx, 
		float By, float gamma, float d, float PSFSigma_y, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim);

__global__ void kernel_MLEFit_sigmaxy(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim);

#endif