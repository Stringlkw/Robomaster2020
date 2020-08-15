#ifndef WINDMILL_H
#define WINDMILL_H

#include <time.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class WindMill
{
private:
    int _enemy_color; //敌人装甲板颜色
    Mat _srcImg; //原图像
    Mat _roiImg; //roi from the result of last frame
    Mat _grayImg; //灰度图
    Mat binBrightImg; //二值图
    Mat templ[9]; //模板图片

    //风车参数
    const float maxHWRatio = 0.7153846;
    const float maxArea = 2000;
    const float minArea = 500;


    //模板图片载入
    void loadTempImage(String filepath);



    //图像预处理
    void ISP();


public:
    WindMill()
    {
        _enemy_color = 2;
    }
    void detect();


};

#endif // WINDMILL_H
