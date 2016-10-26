//////////////////////////////////////////////////////////////////////////
// Creates C++ MEX-file for OpenCV routine matchTemplate. 
// Here matchTemplate uses normalized cross correlation method to search 
// for matches between an image patch and an input image.
//
// Copyright 2014 The MathWorks, Inc.
//////////////////////////////////////////////////////////////////////////

#include "opencvmex.hpp"

#define _DO_NOT_EXPORT
#if defined(_DO_NOT_EXPORT)
#define DllExport  
#else
#define DllExport __declspec(dllexport)
#endif


///////////////////////////////////////////////////////////////////////////
// Main entry point to a MEX function
///////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  
    
    // Convert mxArray inputs into OpenCV types
    cv::Ptr<cv::Mat> ImgCV = ocvMxArrayToImage_uint8(prhs[0], true);
    
    cv::Mat outCV(227,227, CV_8UC3);
    
    // Run the OpenCV template matching routine
    cv::resize(*ImgCV, outCV, cv::Size(227,227));
    
    // Put the data back into the output MATLAB array
    plhs[0] = ocvMxArrayFromImage_uint8(outCV);
}

