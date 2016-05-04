#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <mex.h>
#include <cuda_runtime.h>

#include "Python.h"
#include "numpy/arrayobject.h"

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

extern "C" void kernel_MLEFit_pix_threads_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim, int calcCRB);

extern "C" void kernel_MLEFit_pix_threads_astig_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim, int calcCRB);

extern "C" void kernel_MLEFit_sigma_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim);

extern "C" void kernel_MLEFit_z_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma_x, float Ax, float Ay, float Bx, 
		float By, float gamma, float d, float PSFSigma_y, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim);

extern "C" void kernel_MLEFit_sigmaxy_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim); 


//PyMODINIT_FUNC initpyMLEfit(void);

int CUDAERRROR(const char *instr) {
    cudaError_t errornum;
    const char *str;
    errornum=cudaGetLastError();
    if (errornum) {
        str=cudaGetErrorString(errornum);
        //mexPrintf("%s\n", str);
        PyErr_Format(PyExc_RuntimeError, "CUDA Error: %s", str);
        //mexPrintf("You should clear this function in MATLAB for proper operation.\n", str);
        //cudaDeviceReset();
        return 1;
    }  else return 0;
}

//void mexFunction(int nlhs, mxArray *plhs[],	int	nrhs, const	mxArray	*prhs[]) {
static PyObject * doMLEFits(PyObject *self, PyObject *args, PyObject *keywds){
    /*mex function is called with an array of left hand side arguments (returns), and an array of right hand side
    arguments (parameters)*/
    int blockx;
    int threadx;
    //const mwSize *datasize;

    //FIXME - make parameter
    const int calcCRB = 1;

    int size_x, size_y, Nfitraw, NDim;

    float PSFSigma;//, Ax, Ay, Bx, By, gamma, d, PSFSigma_y;
    int iterations, fittype;
    float *data, *d_data=0, *varim, *d_varim=0, *gainim, *d_gainim=0;
    float *d_Parameters=0,*d_CRLBs=0,*d_LogLikelihood=0;
    size_t sz, Nfits;

    PyObject *oData=0;
    PyObject *oVarim=0;
    PyObject *oGainim=0;

    PyArrayObject *aData=0;
    PyArrayObject *aVarim=0;
    PyArrayObject *aGainim=0; 

    PyArrayObject *aOutParams = 0;
    PyArrayObject *aOutCRLBs = 0;
    PyArrayObject *aOutLogLikelihood = 0;

    PyObject *outputTuple = 0;

    npy_intp outputSize[2];

    //#define CLEANUP() {}
    #define CLEANUP() {                           \
        cudaFree(d_Parameters);                   \
        cudaFree(d_CRLBs);                        \
        cudaFree(d_LogLikelihood);                \
        cudaFree(d_varim);                        \
        cudaFree(d_gainim);                       \
        cudaFree(d_data);                         \
                                                  \
        /*ensure we free up any references we have*/ \
        Py_XDECREF(aData);                        \
        Py_XDECREF(aVarim);                       \
        Py_XDECREF(aGainim);                      \
        Py_XDECREF(aOutParams);                   \
        Py_XDECREF(aOutCRLBs);                    \
        Py_XDECREF(aOutLogLikelihood);            \
        return NULL;                              \
        };

    //float * 

    //If device is in an error state, reset
    //this happens when, e.g. a memory error occurs on a previous call
    if (cudaGetLastError()) {
        cudaDeviceReset();
        //cudaGetLastError();
    }


    static char *kwlist[] = {"data", "PSFSigma", "iterations","fittype", "varim","gainim", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OfiiOO", kwlist, 
         &oData, &PSFSigma, &iterations, &fittype, &oVarim, &oGainim))
        return NULL; 

    //aData = (PyArrayObject *) PyArray_ContiguousFromObject(oData, PyArray_FLOAT, 2, 3);
    aData = (PyArrayObject*)PyArray_FROMANY(oData,PyArray_FLOAT, 2,3, NPY_ARRAY_F_CONTIGUOUS);
    if (aData == NULL) 
    {
      PyErr_Format(PyExc_RuntimeError, "Malformed data input");   
      CLEANUP()
    }

    //aVarim = (PyArrayObject *) PyArray_ContiguousFromObject(oVarim, PyArray_FLOAT, 2, 3);
    aVarim = (PyArrayObject*)PyArray_FROMANY(oVarim,PyArray_FLOAT, 2,3, NPY_ARRAY_F_CONTIGUOUS);
    if (aVarim == NULL) 
    {
      PyErr_Format(PyExc_RuntimeError, "Malformed variance image input");   
      CLEANUP()
    }

    //aGainim = (PyArrayObject *) PyArray_ContiguousFromObject(oGainim, PyArray_FLOAT, 2, 3);
    aGainim = (PyArrayObject*)PyArray_FROMANY(oGainim,PyArray_FLOAT, 2,3, NPY_ARRAY_F_CONTIGUOUS);
    if (aGainim == NULL) 
    {
      PyErr_Format(PyExc_RuntimeError, "Malformed gain image input");   
      CLEANUP()
    }

    //input checks
    /*if (nrhs<6)
        mexErrMsgTxt("Data input must include: data,PSFSigma,iterations,fittype,varim,gainim !\n");
    
    if (mxGetClassID(prhs[0])!=mxSINGLE_CLASS)
        mexErrMsgTxt("Data must be comprised of single floats!\n");
    */

    //datasize=mxGetDimensions(prhs[0]);

    size_x = PyArray_DIM((PyObject*)aData, 0);
    size_y = PyArray_DIM((PyObject*)aData, 1);
    //Ndim=mxGetNumberOfDimensions(prhs[0]);
    NDim = PyArray_NDIM((PyObject*)aData);
    if (NDim > 2){
        Nfitraw = PyArray_DIM((PyObject*)aData, 2);
    } else Nfitraw = 1;
    
    //we either get called with a single 2D image/ROI or 3D array of images/ROIs
    //if (Ndim==2)Nfitraw=1;else Nfitraw=datasize[2];
    
    if (size_x > IMSZBIG){
        PyErr_Format(PyExc_RuntimeError,"X,Y dimension of data must be smaller than 21.\n");
        CLEANUP()
    }

    if (size_x!=size_y){
        PyErr_Format(PyExc_RuntimeError,"Fit Box must be square");
        CLEANUP()
    }
    
    //sz is the size (x) of the array
	sz=size_x;
    // mexPrintf("c sizeX: %d Ndims: %d\n",sz,Ndim);
    
    //put as many images as fit into a block

    //this calculation is based on being able to fit the data into the block/multi-processor (MP) level shared memory
    //the of the shared memory is size of 16KB for compute capability 1, 48KB (by default, although configurable)
    //for compute capability 2.x and 3.x
    //The M4000 has 96KB ram per MP (some of this will be allocated as L1 cache rather than shared memory)
    //the 15KB here looks like a reasonable, if conservative choice which will work on all GPUs, 
    //although it will potentially limit us to a relatively
    //small number of threads per MP (3 for the maximum ROI size of 21x21)
    //The number of SMMs for the M4000 is 13
    //The usage of large ammounts of shared memory per thread also limits the number of blocks that can execute concurrently
    //With 15k per block, only 4 blocks can execute concurrently on each M4000 SMM (less on earlier GPUs)
    //float szsqaure = sz*sz;
    int BlockSize = (int) floor((float)15000/(4*sz*sz*3));
    //BlockSize = max(4, BlockSize); //this causes a memory error for large ROIs
    BlockSize = max(2, BlockSize); //DB = change minimum block size to 2, otherwise we try to use too much memory
    BlockSize = min(BSZ, BlockSize);
    
    //mexPrintf("cdasdas %d\n",BlockSize);
    
    Nfits= (int) BlockSize*(int) ceil( (float) Nfitraw/BlockSize);
    
    Nfits= (int) Nfitraw;
    //mexPrintf("c sizeX: %d Nfits: %d\n", sz, Nfits);
    //mexPrintf("Nfits: %d, Nfitraw: %d\n", Nfits, Nfitraw);
    
    //get variables
    //data=(float *) mxGetData(prhs[0]);
    data = (float *)PyArray_DATA(aData);
    //PSFSigma=(float)mxGetScalar(prhs[1]); //matlab-dip_image convention
    //iterations=(int) mxGetScalar(prhs[2]);
    //fittype = (int) mxGetScalar(prhs[3]);
    varim=(float *) (float *)PyArray_DATA(aVarim);
	gainim=(float *) (float *)PyArray_DATA(aGainim);

    //These additional parameters seem to be for z-fitting
    //Leave them out for now
    /*
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
    */
    
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

    outputSize[0] = 0;
    outputSize[1] = 0;

    switch(fittype){
        case 1: // (x,y,bg,I)
            cudaMalloc((void**)&d_Parameters,   NV_P*Nfits*sizeof(float));
            cudaMalloc((void**)&d_CRLBs,        NV_P*Nfits*sizeof(float));
            
            outputSize[0] = NV_P;
            outputSize[1] = Nfitraw;

            //printf("case: fitype==1\n");

            //plhs[0]=mxCreateNumericMatrix(Nfitraw, NV_P, mxSINGLE_CLASS, mxREAL);
            //plhs[1]=mxCreateNumericMatrix(Nfitraw, NV_P, mxSINGLE_CLASS, mxREAL);
            break;
        case 2: // (x,y,bg,I,Sigma)
			cudaMalloc((void**)&d_Parameters,   NV_PS*Nfits*sizeof(float));
            cudaMalloc((void**)&d_CRLBs,        NV_PS*Nfits*sizeof(float));

            outputSize[0] = NV_PS;
            outputSize[1] = Nfitraw;
            //plhs[0]=mxCreateNumericMatrix(Nfitraw, NV_PS, mxSINGLE_CLASS, mxREAL);
            //plhs[1]=mxCreateNumericMatrix(Nfitraw, NV_PS, mxSINGLE_CLASS, mxREAL);
            break; 
        case 3: // (x,y,bg,I,z)
            PyErr_Format(PyExc_RuntimeError,"3D fitting is not supported in this version. Please contact the authors for updates.");
			CLEANUP()
        case 4: // (x,y,bg,I,Sx,Sy)
            PyErr_Format(PyExc_RuntimeError, "Astigmatism fitting is not supported in this version. Please contact the authors for updates.");   
			CLEANUP()
        case 5: //thread parallel version of 1
            cudaMalloc((void**)&d_Parameters,   NV_P*Nfits*sizeof(float));
            cudaMalloc((void**)&d_CRLBs,        NV_P*Nfits*sizeof(float));
            
            outputSize[0] = NV_P;
            outputSize[1] = Nfitraw;
            break;
        case 6: //thread parallel version of 4 (astigmatic)
            cudaMalloc((void**)&d_Parameters,   NV_PS2*Nfits*sizeof(float));
            cudaMalloc((void**)&d_CRLBs,        NV_PS2*Nfits*sizeof(float));
            
            outputSize[0] = NV_PS2;
            outputSize[1] = Nfitraw;
            break;
    }

    //printf("outputSize: %d,%d\n", outputSize[0], outputSize[1]);
    //printf("NV_P, Nfitraw, fittype: %d,%d, %d\n", NV_P, Nfitraw, fittype);
    
    //allocate numpy arrays for output
    aOutParams = (PyArrayObject*) PyArray_New(&PyArray_Type, 2,outputSize,NPY_FLOAT, NULL, NULL, 0, 1, NULL);
    aOutCRLBs = (PyArrayObject*) PyArray_New(&PyArray_Type, 2,outputSize,NPY_FLOAT, NULL, NULL, 0, 1, NULL);
    if ((aOutParams == 0) || (aOutCRLBs == 0)){
        PyErr_Format(PyExc_RuntimeError,"Could not allocate output arrays");
        CLEANUP()
    }

    cudaMalloc((void**)&d_LogLikelihood,        Nfits*sizeof(float));
    //plhs[2]=mxCreateNumericMatrix(Nfitraw, 1, mxSINGLE_CLASS, mxREAL);

    outputSize[0] = 1;
    aOutLogLikelihood = (PyArrayObject*) PyArray_New(&PyArray_Type, 2,outputSize,NPY_FLOAT, NULL, NULL, 0, 1, NULL);
    if ((aOutParams == 0) || (aOutCRLBs == 0)){
        PyErr_Format(PyExc_RuntimeError,"Could not allocate output arrays");
        CLEANUP()
    }
    
    //setup kernel
    blockx = (int) ceil( (float)Nfits/(float)BlockSize);
    threadx= BlockSize;
    
    dim3 dimBlock(threadx);
    dim3 dimGrid(blockx);
    
    //printf("threadx: %d,blockx: %d,Nfitraw: %d\n", threadx, blockx, Nfitraw);
    
    switch(fittype) {
        case 1: //fit x,y,bg,I
            kernel_MLEFit_wrapper(dimGrid, dimBlock, d_data, PSFSigma, (int) sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim, d_gainim);
            break;
            
        case 2: //fit x,y,bg,I,sigma
            kernel_MLEFit_sigma_wrapper(dimGrid, dimBlock, d_data, PSFSigma, (int) sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim, d_gainim);
            break;
            
		
		//******************* 3D localization using z fit does not work for this version of sCMOS software **********************//
        case 3: //fit x,y,bg,I,z  
            //kernel_MLEFit_z_wrapper(dimGrid, dimBlock, d_data, PSFSigma, Ax, Ay, Bx, By, gamma, d, PSFSigma_y, (int) sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim);
            //mexPrintf("A:%f B:%f gamma:%f d:%f \n",Ax,Bx,gamma,d);
            break;
       //******************* Sigma fit does not work for this version of sCMOS software *****************************************//     
        case 4: //fit x,y,bg,I,sigmax,sigmay
            //kernel_MLEFit_sigmaxy_wrapper(dimGrid, dimBlock, d_data, PSFSigma, (int) sz, iterations, d_Parameters, d_CRLBs, d_LogLikelihood, Nfits, d_varim);
            break;

        case 5: // do thread parallel version
            {
                dim3 dimBlock(size_x, size_y);
                dim3 dimGrid(Nfitraw);
                kernel_MLEFit_pix_threads_wrapper(dimGrid, dimBlock, d_data, PSFSigma, (int) sz, iterations, 
                    d_Parameters, d_CRLBs, d_LogLikelihood,Nfits, d_varim, d_gainim, calcCRB);
            }
            break;

        case 6: // do thread parallel version
            {
                dim3 dimBlock(size_x, size_y);
                dim3 dimGrid(Nfitraw);
                kernel_MLEFit_pix_threads_astig_wrapper(dimGrid, dimBlock, d_data, PSFSigma, (int) sz, iterations, 
                    d_Parameters, d_CRLBs, d_LogLikelihood,Nfits, d_varim, d_gainim, calcCRB);
            }
            break;
            
    }

    if (CUDAERRROR("kernel")){
        CLEANUP()
    }
    
    //copy to matlab output
    switch(fittype){
        case 1: // (x,y,bg,I)
            cudaMemcpy((float *)PyArray_DATA(aOutParams), d_Parameters, NV_P*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy((float *)PyArray_DATA(aOutCRLBs), d_CRLBs, NV_P*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            break;
        case 2: // (x,y,bg,I,Sigma)
        case 3: // (x,y,bg,I,z)
            cudaMemcpy((float *)PyArray_DATA(aOutParams), d_Parameters, NV_PS*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy((float *)PyArray_DATA(aOutCRLBs), d_CRLBs, NV_PS*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            break;
        case 4: // (x,y,bg,I,Sx,Sy)
            cudaMemcpy((float *)PyArray_DATA(aOutParams), d_Parameters, NV_PS2*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy((float *)PyArray_DATA(aOutCRLBs), d_CRLBs, NV_PS2*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            break;
        case 5:
            cudaMemcpy((float *)PyArray_DATA(aOutParams), d_Parameters, NV_P*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy((float *)PyArray_DATA(aOutCRLBs), d_CRLBs, NV_P*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            break;
        case 6:
            cudaMemcpy((float *)PyArray_DATA(aOutParams), d_Parameters, NV_PS2*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy((float *)PyArray_DATA(aOutCRLBs), d_CRLBs, NV_PS2*Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);
            break;
    }
    cudaMemcpy((float *)PyArray_DATA(aOutLogLikelihood  ), d_LogLikelihood, Nfitraw*sizeof(float), cudaMemcpyDeviceToHost);

    if (CUDAERRROR("cudaMemcpy")){
        CLEANUP()
    }
    
    //cleanup
    cudaFree(d_Parameters);
    cudaFree(d_CRLBs);
    cudaFree(d_LogLikelihood);
	cudaFree(d_varim);
	cudaFree(d_gainim);
	cudaFree(d_data);

    Py_XDECREF(aData);
    Py_XDECREF(aVarim);
    Py_XDECREF(aGainim);

    outputTuple = Py_BuildValue("OOO",(PyObject*) aOutParams,(PyObject*) aOutCRLBs,(PyObject*) aOutLogLikelihood);
    
    Py_XDECREF(aOutParams);
    Py_XDECREF(aOutCRLBs);
    Py_XDECREF(aOutLogLikelihood);

    return outputTuple;
};

static PyMethodDef pyMLEfitMethods[] = {
    {"doMLEFits",  (PyCFunction) doMLEFits, METH_VARARGS | METH_KEYWORDS,
    "Perform GPU based MLE fitting\nArguments are: data,PSFSigma,iterations,fittype,varim,gainim"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initpyMLEfit(void)
{
    PyObject *m;

    m = Py_InitModule("pyMLEfit", pyMLEfitMethods);
    import_array();

}
