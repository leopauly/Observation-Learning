///////////////////////////////////////////////////////////////////////////
//
// Courtesy : Sample programs @ STEREOLABS
// Customised for data collection : @leopauly (cnlp@leeds.ac.uk)
//
///////////////////////////////////////////////////////////////////////////

/***********************************************************************************************
 ** Data collection program for collecting rgb images and depth images
 ** z: for starting recording ; f: for ending recording
 ***********************************************************************************************/

 // ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Sample includes
#include <SaveDepth.hpp>

// includes
#include <vector>


using namespace sl;

cv::Mat slMat2cvMat(Mat& input);
void printHelp();

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;
    int enter=0,count=0;
    std::string filename_rgb,filename_depth;


    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_HD720;
    //init_params.depth_mode = DEPTH_MODE_QUALITY;
    //init_params.coordinate_units = UNIT_CENTIMETER;
    init_params.camera_fps = 30 ;
    //init_params.depth_minimum_distance = 30 ;
    int f=0;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS) {
        printf("%s\n", errorCode2str(err).c_str());
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Display help in console
    printHelp();

    // Set runtime parameters after opening the camera
    //RuntimeParameters runtime_parameters;
    //runtime_parameters.sensing_mode = SENSING_MODE_FILL;

    // Prepare new image size to retrieve half-resolution images
    Resolution image_size = zed.getResolution();
    int new_width = image_size.width;
    int new_height = image_size.height;

    //print(zed.getCameraSettings())

    // To share data between sl::Mat and cv::Mat, use slMat2cvMat()
    // Only the headers and pointer to the sl::Mat are copied, not the data itself
    Mat image_zed;
    //cv::Mat image_ocv = slMat2cvMat(image_zed);
    Mat depth_image_zed(new_width, new_height, MAT_TYPE_8U_C4);
    //cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
    //Mat point_cloud;
    std::vector<cv::Mat> store;

    // Loop until 'q' is pressed
    char key = ' ';
    //zed.setCameraSettings(CAMERA_SETTINGS_EXPOSURE, 30, false);
    //zed.setCameraSettings(CAMERA_SETTINGS_EXPOSURE, -1, true);
    while (f != 100) {

        if (zed.grab() == SUCCESS) {

            // Retrieve the left image, depth image in half-resolution
            //std::cout << zed.getCameraFPS() << '\n';
            zed.retrieveImage(image_zed, VIEW_LEFT);
            zed.retrieveImage(image_zed, VIEW_RIGHT);
            zed.retrieveImage(depth_image_zed, VIEW_DEPTH, MEM_CPU, new_width, new_height);

            // Retrieve the RGBA point cloud in half-resolution
            // zed.retrieveMeasure(point_cloud, MEASURE_XYZRGBA, MEM_CPU, new_width, new_height);

            // Display image and depth using cv:Mat which share sl:Mat data

            //cv::imshow("Image", image_ocv);
            //cv::imshow("Depth", depth_image_ocv);

            //if (key=='z')
            //{std::cout << "Entered" << '\n';
            //enter=1;}
            //if (key=='f')
            //{enter=0;}


            // For recording
            //if (enter==1)
            //{
              std::cout << "Recording....." << '\n';

              //store.push_back(image_ocv);
              //std::cout << image_ocv << '\n';

            //}


            // Handle key event
            //key =cv::waitKey(10); //std::cout <<"Playing"<<key << '\n';
            // std::cout << f << '\n';
            //f=f+1;
        }
    }

    //cv::imwrite("leo.png",store.pop_back());
    zed.close();
    for (std::vector<cv::Mat>::iterator it = store.begin() ; it != store.end(); ++it)
    {
    std::stringstream a;
    a << count;
    filename_rgb="./rgb/"+a.str()+"rgb.png";
    filename_depth="./depth/"+a.str()+"depth.png";
    cv::imwrite(filename_rgb,*it);

    std::cout << count << '\n';
    //cv::imwrite(filename_depth,depth_image_ocv);
    count=count+1;
    }
    return 0;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

/**
* This function displays help in console
**/
void printHelp() {
    std::cout << " Press 'z' to start recording" << std::endl;
    std::cout << " Press 'f' to end recording" << std::endl;
    std::cout << " Press 'q' to quite" << std::endl;
}
