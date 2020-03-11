#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mex.h>
#include <cuda_runtime.h>

// Thread block size
#define BSZ 64
//#define BSZ 64
#define MEM 3872
#define IMSZ 11
#define IMSZBIG 21
#define NK 128 //number of blocks to run in each kernel
#define pi 3.141592f
#define NV_P 4
#define NV_PS 5
#define NV_PZ 5
#define NV_PS2 6

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

extern "C" void kernel_MLEFit_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim); 

extern "C" void kernel_MLEFit_sigma_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim);

extern "C" void kernel_MLEFit_z_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma_x, float Ax, float Ay, float Bx, 
		float By, float gamma, float d, float PSFSigma_y, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim);

extern "C" void kernel_MLEFit_sigmaxy_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim); 


void CUDAERRROR(const char *instr) {
    cudaError_t errornum;
    const char *str;
    if (errornum=cudaGetLastError()) {
        str=cudaGetErrorString(errornum);
        mexPrintf("%s\n", str);
        mexPrintf("You should clear this function in MATLAB for proper operation.\n", str);
    }
}

void mexFunction(int nlhs, mxArray *plhs[],	int	nrhs, const	mxArray	*prhs[]) {
    int blockx;
    int threadx;
    const mwSize *datasize;
    float PSFSigma, Ax, Ay, Bx, By, gamma, d, PSFSigma_y;
    int iterations, Nfits;
    float *data, *d_data, *varim, *d_varim, *gainim, *d_gainim;
    float *d_Parameters,*d_CRLBs,*d_LogLikelihood;
    size_t Ndim, Nfitraw, fittype, sz;

    //input checks
    if (nrhs<6)
        mexErrMsgTxt("Data input must include: data,PSFSigma,iterations,fittype,varim,gainim !\n");
    
    if (mxGetClassID(prhs[0])!=mxSINGLE_CLASS)
        mexErrMsgTxt("Data must be comprised of single floats!\n");
    
    datasize=mxGetDimensions(prhs[0]);
    Ndim=mxGetNumberOfDimensions(prhs[0]);
    
    if (Ndim==2)Nfitraw=1;else Nfitraw=datasize[2];
    
    if (datasize[0] > IMSZBIG)
        mexErrMsgTxt("X,Y dimension of data must be smaller than 21.\n");
    if (datasize[1]!=datasize[0])
        mexErrMsgTxt("Fit Box must be square");
    
	sz=datasize[0];
    // mexPrintf("c sizeX: %d Ndims: %d\n",sz,Ndim);
    
    //put as many images as fit into a block
    //float szsqaure = sz*sz;
    int BlockSize = (int) floor((float)15000/4/sz/sz/3);
    BlockSize = max(4, BlockSize);
    BlockSize = min(BSZ, BlockSize);
    
    //mexPrintf("cdasdas %d\n",BlockSize);
    
    Nfits= (int) BlockSize*(int) ceil( (float) Nfitraw/BlockSize);
    Nfits= (int) Nfitraw;
    //mexPrintf("c sizeX: %d Nfits: %d\n", sz, Nfits);
    //mexPrintf("Nfits: %d, Nfitraw: %d\n", Nfits, Nfitraw);
    
    //get variables
    data=(float *) mxGetData(prhs[0]);
    PSFSigma=(float)mxGetScalar(prhs[1]); //matlab-dip_image convention
    iterations=(int) mxGetScalar(prhs[2]);
    fittype = (int) mxGetScalar(prhs[3]);
    varim=(float *) mxGetData(prhs[4]);
	gainim=(float *) mxGetData(prhs[5]);

    if (nrhs>6)
        Ax = (float) mxGetScalar(prhs[6]);
    if (nrhs>7)
        Ay = (float) mxGetScalar(prhs[7]);
    if (nrhs>8)
        Bx = (float) mxGetScalar(prhs[8]);
    if (nrhs>9)
        By = (float) mxGetScalar(prhs[9]);
    if (nrhs>10)
        gamma = (float) mxGetScalar(prhs[10]);
    if (nrhs>11)
        d = (float) mxGetScalar(prhs[11]);
    if (nrhs>12)
        PSFSigma_y = (float) mxGetScalar(prhs[12]);
    else
        PSFSigma_y =PSFSigma;
    
    //create device variable for data and copy to device
    cudaMalloc((void**)&d_data, sz*sz*Nfits*sizeof(float));
    cudaMemset(d_data, 0, sz*sz*Nfits*sizeof(float));
    cudaMemcpy(d_data, data, sz*sz*Nfitraw*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_varim, sz*sz*Nfits*sizeof(float));
    cudaMemset(d_varim, 0, sz*sz*Nfits*sizeof(float));
    cudaMemcpy(d_varim, varim, sz*sz*Nfitraw*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_gainim, sz*sz*Nfits*sizeof(float));
    cudaMemset(d_gainim, 0, sz*sz*Nfits*sizeof(float));
    cudaMemcpy(d_gainim, gainim, sz*sz*Nfitraw*sizeof(float), cudaMemcpyHostToDevice);

    //mexPrintf("gain: %f, var: %f\n", gainim[0], varim[0]);
    //create output for parameters and CRLBs
    switch(fittype){
        case 1: // (x,y,bg,I)
            cudaMalloc((void**)&d_Parameters,   NV_P*Nfits*sizeof(float));
            cudaMalloc((void**)&d_CRLBs,        NV_P*Nfits*sizeof(float));
            plhs[0]=mxCreateNumericMatrix(Nfitraw, NV_P, mxSINGLE_CLASS, mxREAL);
            plhs[1]=mxCreateNumericMatrix(Nfitraw, NV_P, mxSINGLE_CLASS, mxREAL);
            break;
        case 2: // (x,y,bg,I,Sigma)
			cudaMalloc((void**)&d_Parameters,   NV_PS*Nfits*sizeof(float));
            cudaMalloc((void**)&d_CRLBs,        NV_PS*Nfits*sizeof(float));
            plhs[0]=mxCreateNumericMatrix(Nfitraw, NV_PS, mxSINGLE_CLASS, mxREAL);
            plhs[1]=mxCreateNumericMatrix(Nfitraw, NV_PS, mxSINGLE_CLASS, mxREAL);
            break; 
        case 3: // (x,y,bg,I,z)
            mexErrMsgTxt("3D fitting is not supported in this version. Please contact the authors for updates.");
			break;
        case 4: // (x,y,bg,I,Sx,Sy)
            mexErrMsgTxt("Astigmatism fitting is not supported in this version. Please contact the authors for updates.");   
			break;
    }
    cudaMalloc((void**)&d_LogLikelihood,        Nfits*sizeof(float));
    plhs[2]=mxCreateNumericMatrix(Nfitraw, 1, mxSINGLE_CLASS, mxREAL);
    
    //setup kernel
    blockx = (int) ceil( (float)Nfits/(float)BlockSize);
    threadx= BlockSize;
    
    dim3 dimBlock(threadx);
    dim3 dimGrid(blockx);
    
    printf("threadx: %d,blockx: %d,Nfitraw: %d\n", threadx, blockx, Nfitraw);
    
    switch(fittype) {
        case 1: //fit x,y,bg,I
            kernel_MLEFit_wrapper(dimGrid, dimBlock, d_data, PSFSigma, (int) sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim, d_gainim);
            break;
            
        case 2: //fit x,y,bg,I,sigma
            kernel_MLEFit_sigma_wrapper(dimGrid, dimBlock, d_data, PSFSigma, (int) sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim, d_gainim);
            break;
            
		
		//******************* 3D localization using z fit does not work for this version of sCMOS software **********************//
        case 3: //fit x,y,bg,I,z  
            kernel_MLEFit_z_wrapper(dimGrid, dimBlock, d_data, PSFSigma, Ax, Ay, Bx, By, gamma, d, PSFSigma_y, (int) sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim);
            //mexPrintf("A:%f B:%f gamma:%f d:%f \n",Ax,Bx,gamma,d);
            break;
       //******************* Sigma fit does not work for this version of sCMOS software *****************************************//     
        case 4: //fit x,y,bg,I,sigmax,sigmay
            kernel_MLEFit_sigmaxy_wrapper(dimGrid, dimBlock, d_data, PSFSigma, (int) sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim);
            
    }
    CUDAERRROR("kernel");
    
    //copy to matlab output
    switch(fittype){
        case 1: // (x,y,bg,I)
            cudaMemcpy((float *)mxGetData(plhs[0]), d_Parameters, NV_P*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy((float *)mxGetData(plhs[1]), d_CRLBs, NV_P*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            break;
        case 2: // (x,y,bg,I,Sigma)
        case 3: // (x,y,bg,I,z)
            cudaMemcpy((float *)mxGetData(plhs[0]), d_Parameters, NV_PS*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy((float *)mxGetData(plhs[1]), d_CRLBs, NV_PS*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            break;
        case 4: // (x,y,bg,I,Sx,Sy)
            cudaMemcpy((float *)mxGetData(plhs[0]), d_Parameters, NV_PS2*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy((float *)mxGetData(plhs[1]), d_CRLBs, NV_PS2*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            break;
    }
    cudaMemcpy((float *)mxGetData(plhs[2]), d_LogLikelihood, Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
    
    //cleanup
    cudaFree(d_Parameters);
    cudaFree(d_CRLBs);
    cudaFree(d_LogLikelihood);
	cudaFree(d_varim);
	cudaFree(d_gainim);
	cudaFree(d_data);
    return;
}
