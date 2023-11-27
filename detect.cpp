#include "mex.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("detect:nrhs", "One input argument required.");
    }

    if (nlhs != 1) {
        mexErrMsgIdAndTxt("detect:nlhs", "One output argument required.");
    }

    char* imagePath = mxArrayToString(prhs[0]);

    try {
        dlib::array2d<dlib::bgr_pixel> img;
        dlib::load_image(img, imagePath);

        dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
        std::vector<dlib::rectangle> faces = faceDetector(img);

        plhs[0] = mxCreateDoubleMatrix(faces.size(), 4, mxREAL);
        double* output = mxGetPr(plhs[0]);

        for (size_t i = 0; i < faces.size(); ++i) {
            output[i] = faces[i].left();
            output[i + faces.size()] = faces[i].top();
            output[i + 2 * faces.size()] = faces[i].width();
            output[i + 3 * faces.size()] = faces[i].height();
        }
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("detect:exception", "An error occurred during face detection: %s", e.what());
  }

    mxFree(imagePath);
}
