
#include <iostream>
#include <string>  
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc , char** argv)
{

  //Initialization of the data
  Mat img = imread("../office_noisy.png", CV_LOAD_IMAGE_GRAYSCALE);

  Mat u0, laplacian;
  double D = 1., dt = 0.1, t = 0., tMax = 100, eps = 1e-8;
 
  img.convertTo(u0, CV_64F); //Convert the image source in CV_32F (i.e float with a range of [0;1])

  //Starting the process of diffusion
 
  while(t < tMax)
    {
      //Compute the laplacian of the current image
      Laplacian(u0, laplacian, CV_64F);
      u0 += dt * D * laplacian; // Apply the diffusion
      t += dt;
     
    }
 
  //Show the output images
  double min, max;
  minMaxLoc(u0, &min, &max);
  u0.convertTo(u0, CV_8U, 255/(max-min), -255.0*min/(max-min));
 	  
  //Display the images during the process of diffusion
  imshow("Linear diffusion output", u0);
  waitKey(0);


  return 0;
}

