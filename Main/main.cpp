//#define DEBUG_PRETREATMENT
#include "Armor/ArmorDetector.h"
using namespace rm;

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace  cv;
int main()
{
    namedWindow("video",WINDOW_AUTOSIZE);

   VideoCapture capture(0);//这句话可以替代上面两个语句，效果是一致的。

   if (!capture.isOpened())
   {
       std::cerr << "Couldn't open capture." << std::endl;
       return -1;
   }

   while(1)
   {
       Mat frame;
       capture  >>  frame;

       if (frame.empty())
       {
           break;
       }
       else
       {
           ArmorParam armorParam;
           ArmorDetector test(armorParam);
           test.loadImg(frame);
           test.detect();
           imshow("video", frame);
       }

       if (waitKey(33) >= 0) break;
       }
        waitKey(0);

       capture.release();
       destroyAllWindows();
       return 0;
}

