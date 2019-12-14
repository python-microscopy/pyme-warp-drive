/*Compile the mex file: (old fashion). VS2008 version does not these steps.
 *
 * nvmex -f Z:\CUDA\nvmexoptsxp64.bat       GPUgaussMLE.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
 * nvmex -f Z:\CUDA\nvmexopts_r2008a.bat    GPUgaussMLE.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
 * nvmex -f Z:\CUDA\nvmexopts_9.bat    GPUgaussMLE.cu -IC:\cuda\include -LC:\cuda\lib -lcudart -output GPUgaussMLEfangMATLAB
 * nvmex -f Z:\cCode\CUDA\nvmexoptsXP32_VS2008_r2008.bat GPUgaussMLE.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
 * nvmex -f Z:\cCode\CUDA\nvmexoptsxp64_VS2008SDK7.bat GPUgaussMLEf1.cu -IC:\cuda\include -LC:\cuda\lib64 -lcudart
 *
 * [Parameters CRLBs LL]=GPUgaussMLE(data,PSFSigma,iterations)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include "cuda_runtime.h"
#include "MatInvLib.h"
#include "GPUgaussLib.cuh"
#include "GPUgaussMLEv2.h"
// Thread block size
#define BSZ 64
//#define BSZ 64
#define MEM 1300
#define IMSZ 11
#define IMSZBIG 21
#define NK 128 //number of blocks to run in each kernel
#define pi 3.141592f
#define NV_P 4
#define NV_PS 5
#define NV_PZ 5
#define NV_PS2 6


//theta is: {x,y,N,bg}
__global__ void kernel_MLEFit(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim){
    
    __shared__ float s_data[MEM];
	__shared__ float s_varim[MEM];
	__shared__ float s_gainim[MEM];
    float M[NV_P*NV_P], Diag[NV_P], Minv[NV_P*NV_P];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
	//float gain=0.45;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=NV_P;
    float dudt[NV_P];
    float d2udt2[NV_P];
    float NR_Numerator[NV_P], NR_Denominator[NV_P];
    float theta[NV_P];
    float maxjump[NV_P]={1e0, 1e0, 1e2, 2e0}; // maxjump is revised such that the intensity can converge to the correct value
    float gamma[NV_P]={1.0, 1.0, 0.5, 1.0};
    float Nmax;

    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;
    
    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;
    //load data
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++)
	{
        s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
		s_varim[sz*sz*tx+sz*jj+ii]=d_varim[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
		s_gainim[sz*sz*tx+sz*jj+ii]=d_gainim[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
	}
    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=max(0.0, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);
    
    for (kk=0;kk<iterations;kk++) {//main iterative loop
        
        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}
        
        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
            PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
            PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);
            
            model=theta[3]+theta[2]*PSFx*PSFy+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
            data=s_data[sz*sz*tx+sz*jj+ii]+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
            
            //calculating derivatives
            kernel_DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], &d2udt2[0]);
            kernel_DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], &d2udt2[1]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0;
            dudt[3] = 1.0;
            d2udt2[3] = 0.0;
            
            cf=0.0;
            df=0.0;
            if (model>10e-3) cf=data/model-1;
            if (model>10e-3) df=data/pow(model, 2);
            cf=min(cf, 10e4);
            df=min(df, 10e4);
            
            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        
        // The update
        if (kk<2)
            for (ll=0;ll<NV;ll++)
                theta[ll]-=gamma[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        else
            for (ll=0;ll<NV;ll++)
                theta[ll]-=min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        
        // Any other constraints
        theta[2]=max(theta[2], 1.0);
        theta[3]=max(theta[3], 0.01);
        
    }
    
    // Calculating the CRLB and LogLikelihood
    Div=0.0;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
        PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);
        
        model=theta[3]+theta[2]*PSFx*PSFy+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
        data=s_data[sz*sz*tx+sz*jj+ii]+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
        
        //calculating derivatives
        kernel_DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;
        
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);
    
    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;
	//d_LogLikelihood[BlockSize*bx+tx] = 1;
    
    return;
}

__global__ void kernel_MLEFit_sigma(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim){
    
    __shared__ float s_data[MEM];
	__shared__ float s_varim[MEM];
	__shared__ float s_gainim[MEM];
    float M[NV_PS*NV_PS], Diag[NV_PS], Minv[NV_PS*NV_PS];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
	//float gain=0.45;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=NV_PS;
    float dudt[NV_PS];
    float d2udt2[NV_PS];
    float NR_Numerator[NV_PS], NR_Denominator[NV_PS];
    float theta[NV_PS];
    float maxjump[NV_PS]={1e0, 1e0, 1e2, 2e0, 5e-1};
    float gamma[NV_PS]={1.0, 1.0, 0.5, 1.0, 1.0};
    float Nmax;
    
    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;
    
    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;
      
    //copy in data
	for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
		s_varim[sz*sz*tx+sz*jj+ii]=d_varim[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
		s_gainim[sz*sz*tx+sz*jj+ii]=d_gainim[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
	}
    
    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=max(0.0, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);
    theta[4]=PSFSigma;
    
    for (kk=0;kk<iterations;kk++) {//main iterative loop
        
        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}
        
        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
            PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
            PSFy=kernel_IntGauss1D(jj, theta[1], theta[4]);
            
            model=theta[3]+theta[2]*PSFx*PSFy+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
            data=s_data[sz*sz*tx+sz*jj+ii]+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
            
            //calculating derivatives
            kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            kernel_DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], &d2udt2[4]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0;
            dudt[3] = 1.0;
            d2udt2[3] = 0.0;
            
            cf=0.0;
            df=0.0;
            if (model>10e-3) cf=data/model-1;
            if (model>10e-3) df=data/pow(model, 2);
            cf=min(cf, 10e4);
            df=min(df, 10e4);
            
            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        
        // The update
        if (kk<5)
            for (ll=0;ll<NV;ll++)
                theta[ll]-=gamma[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        else
            for (ll=0;ll<NV;ll++)
                theta[ll]-=min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        
        // Any other constraints
        theta[2]=max(theta[2], 1.0);
        theta[3]=max(theta[3], 0.01);
        theta[4]=max(theta[4], 0.5);
        theta[4]=min(theta[4], sz/2.0);
    }
    
    // Calculating the CRLB and LogLikelihood
    Div=0.0;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
        PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);
        
        model=theta[3]+theta[2]*PSFx*PSFy+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
        data=s_data[sz*sz*tx+sz*jj+ii]+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
        
        //calculating derivatives
        kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], NULL);
        kernel_DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;
        
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);
  
    
    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;
	//d_LogLikelihood[BlockSize*bx+tx] = 1;
    
    
    return;
}

//******************* 3D localization using z fit does not work for this version of sCMOS software **********************//
__global__ void kernel_MLEFit_z(float *d_data, float PSFSigma_x, float Ax, float Ay, float Bx, float By, float gamma, float d, float PSFSigma_y, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim){
    __shared__ float s_data[MEM];
    float M[5*5], Diag[5], Minv[5*5];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=5;
    float dudt[5];
    float d2udt2[5];
    float NR_Numerator[5], NR_Denominator[5];
    float theta[5];
    float maxjump[5]={1e0, 1e0, 1e2, 2e0, 1e-1};
    float g[5]={1.0, 1.0, 0.5, 1.0, 1.0};
    float Nmax;
    
    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;

    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;
    
    //copy in data
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++)
        s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
    
    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma_x, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=max(0.0, (Nmax-theta[3])*2*pi*PSFSigma_x*PSFSigma_y*sqrt(2.0f));
    theta[4]=0;
   
    for (kk=0;kk<iterations;kk++) {//main iterative loop
        
        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}
        
        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
            kernel_DerivativeIntGauss2Dz(ii, jj, theta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, dudt, d2udt2);
            
            model=theta[3]+theta[2]*PSFx*PSFy;
            data=s_data[sz*sz*tx+sz*jj+ii];
            
            //calculating remaining derivatives
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0;
            dudt[3] = 1.0;
            d2udt2[3] = 0.0;
            
            cf=0.0;
            df=0.0;
            if (model>10e-3) cf=data/model-1;
            if (model>10e-3) df=data/pow(model, 2);
            cf=min(cf, 10e4);
            df=min(df, 10e4);
            
            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        
        // The update
        if (kk<2)
            for (ll=0;ll<NV;ll++)
                theta[ll]-=g[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        else
            for (ll=0;ll<NV;ll++)
                theta[ll]-=min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        
        // Any other constraints
        theta[2]=max(theta[2], 1.0);
        theta[3]=max(theta[3], 0.01);
        
    }
    
    // Calculating the CRLB and LogLikelihood
    Div=0.0;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        
        kernel_DerivativeIntGauss2Dz(ii, jj, theta, PSFSigma_x,PSFSigma_y, Ax,Ay, Bx,By, gamma, d, &PSFx, &PSFy, dudt, NULL);
        
        model=theta[3]+theta[2]*PSFx*PSFy;
        data=s_data[sz*sz*tx+sz*jj+ii];
        
        //calculating remaining derivatives
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;
       
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) 
    kernel_MatInvN(M, Minv, Diag, NV);
  
   //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;
    return;
}

//************* sigma fit does not work for this version of sCMOS software***********************************//
__global__ void kernel_MLEFit_sigmaxy(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim){
 
    __shared__ float s_data[MEM];
    float M[6*6], Diag[6], Minv[6*6];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=6;
    float dudt[6];
    float d2udt2[6];
    float NR_Numerator[6], NR_Denominator[6];
    float theta[6];
    float maxjump[6]={1e0, 1e0, 1e2, 2e0, 1e-1,1e-1};
    float g[6]={1.0, 1.0, 0.5, 1.0, 1.0,1.0};
    float Nmax;
    
    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;
    
    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;
    
    //copy in data
    
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++)
        s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
    
    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=max(0.0, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);
    theta[4]=PSFSigma;
    theta[5]=PSFSigma;
    for (kk=0;kk<iterations;kk++) {//main iterative loop
        
        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}
        
        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
            PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
            PSFy=kernel_IntGauss1D(jj, theta[1], theta[5]);
            
            model=theta[3]+theta[2]*PSFx*PSFy;
            data=s_data[sz*sz*tx+sz*jj+ii];
            
            //calculating derivatives
            kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], &d2udt2[4]);
            kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], &d2udt2[5]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0;
            dudt[3] = 1.0;
            d2udt2[3] = 0.0;
            
            cf=0.0;
            df=0.0;
            if (model>10e-3) cf=data/model-1;
            if (model>10e-3) df=data/pow(model, 2);
            cf=min(cf, 10e4);
            df=min(df, 10e4);
            
            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        
        // The update
            for (ll=0;ll<NV;ll++)
                theta[ll]-=g[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

        // Any other constraints
        theta[2]=max(theta[2], 1.0);
        theta[3]=max(theta[3], 0.01);
        theta[4]=max(theta[4], PSFSigma/10.0);
        theta[5]=max(theta[5], PSFSigma/10.0);  
    }
    
    // Calculating the CRLB and LogLikelihood
    Div=0.0;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        
        PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
        PSFy=kernel_IntGauss1D(jj, theta[1], theta[5]);
        
        model=theta[3]+theta[2]*PSFx*PSFy;
        data=s_data[sz*sz*tx+sz*jj+ii];
        
        //calculating derivatives
        kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], NULL);
        kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], NULL);
        kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;
        
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);
   
    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;
    return;
}
