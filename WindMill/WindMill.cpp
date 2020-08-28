#include"WindMill.h"

static bool CircleInfo2(std::vector<cv::Point2f>& pts, cv::Point2f& center, float& radius)
{
    center = cv::Point2d(0, 0);
    radius = 0.0;
    if (pts.size() < 3) return false;

    double sumX = 0.0;
    double sumY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    double sumX3 = 0.0;
    double sumY3 = 0.0;
    double sumXY = 0.0;
    double sumX1Y2 = 0.0;
    double sumX2Y1 = 0.0;
    const double N = (double)pts.size();
    for (int i = 0; i < pts.size(); ++i)
    {
        double x = pts.at(i).x;
        double y = pts.at(i).y;
        double x2 = x * x;
        double y2 = y * y;
        double x3 = x2 * x;
        double y3 = y2 * y;
        double xy = x * y;
        double x1y2 = x * y2;
        double x2y1 = x2 * y;

        sumX += x;
        sumY += y;
        sumX2 += x2;
        sumY2 += y2;
        sumX3 += x3;
        sumY3 += y3;
        sumXY += xy;
        sumX1Y2 += x1y2;
        sumX2Y1 += x2y1;
    }
    double C = N * sumX2 - sumX * sumX;
    double D = N * sumXY - sumX * sumY;
    double E = N * sumX3 + N * sumX1Y2 - (sumX2 + sumY2) * sumX;
    double G = N * sumY2 - sumY * sumY;
    double H = N * sumX2Y1 + N * sumY3 - (sumX2 + sumY2) * sumY;

    double denominator = C * G - D * D;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double a = (H * D - E * G) / (denominator);
    denominator = D * D - G * C;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double b = (H * C - E * D) / (denominator);
    double c = -(a * sumX + b * sumY + sumX2 + sumY2) / N;

    center.x = a / (-2);
    center.y = b / (-2);
    radius = std::sqrt(a * a + b * b - 4 * c) / 2;
    return true;
}

//获取两点间的距离
double getDistance(Point pointA, Point pointB)
{
    double distance;
    distance = powf((pointA.x - pointB.x), 2) + powf((pointA.y - pointB.y), 2);
    distance = sqrtf(distance);

    return distance;
}

//模板匹配
double TemplateMatch(cv::Mat image, cv::Mat tepl, cv::Point& point, int method)
{
    int result_cols = image.cols - tepl.cols + 1;
    int result_rows = image.rows - tepl.rows + 1;
    cv::Mat result = cv::Mat(result_cols, result_rows, CV_32FC1);
    cv::matchTemplate(image, tepl, result, method);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    switch (method)
    {
    case TM_SQDIFF:
    case TM_SQDIFF_NORMED:
        point = minLoc;
        return minVal;

    default:
        point = maxLoc;
        return maxVal;
    }
}


void WindMill:: loadTempImage(String filepath)
{
    for (int i = 1; i <= 8; i++)
    {
        templ[i] = imread(filepath + to_string(i) + ".jpg", IMREAD_GRAYSCALE);
    }
}


void WindMill::ISP()
{
    vector<Mat> channels;
    split(_roiImg, channels); //分离色彩通道
    if (_roiImg.empty())
    {
        return;
    }
    //预处理删除己方装甲板颜色
    if (_enemy_color == 2)
    {
        _grayImg = channels.at(2) - channels.at(0); //Get red-blue image;
    }
    else
    {
        _grayImg = channels.at(0) - channels.at(2); //Get blue-red image;
    }


    threshold(_grayImg, binBrightImg, 100, 255, THRESH_BINARY);
    //imshow("gray", _grayImg);


    int KernelSize = 2;
    Mat kernel = getStructuringElement(cv::MORPH_RECT, Size(KernelSize * 2 + 1, KernelSize * 2 + 1),
                                       Point(KernelSize, KernelSize));
    dilate(binBrightImg, binBrightImg, kernel); //膨胀

    //形态学闭运算,去除小孔
    KernelSize = 1;
    kernel = getStructuringElement(cv::MORPH_RECT, Size(KernelSize * 2 + 1, KernelSize * 2 + 1),
                                   Point(KernelSize, KernelSize));
    morphologyEx(binBrightImg, binBrightImg, MORPH_CLOSE, kernel);
}


Point2f WindMill:: detect()
    {
        vector<Point2f> cirV;
        Point2f cc = Point2f(0, 0);

        String filepath = "template/template";
        loadTempImage(filepath);

        //载入视频
        VideoCapture capture("red.avi");
        //deoCapture capture(0);
        if (!capture.isOpened())
        {
            cout << "fail to open video" << endl;
            return Point2f(0, 0);
        }
        capture >> _roiImg;
        while (!_roiImg.empty())
        {
            capture >> _roiImg;
            ISP();

            //风车扇叶识别
            //查找轮廓
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(binBrightImg, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            RotatedRect rect_tmp;
            bool findTarget = false;
            //遍历轮廓
            if (hierarchy.size())
            {
                for (int i = 0; i >= 0; i = hierarchy[i][0])
                {
                    //找出轮廓的最小外接矩形
                    rect_tmp = minAreaRect(contours[i]);
                    Point2f P[4];
                    rect_tmp.points(P);

                    //为透视变换做准备
                    Point2f srcRect[4];
                    Point2f dstRect[4];
                    double width;
                    double height;

                    //矫正提取叶片的宽高
                    width = getDistance(P[0], P[1]);
                    height = getDistance(P[1], P[2]);
                    if (width > height)
                    {
                        srcRect[0] = P[0];
                        srcRect[1] = P[1];
                        srcRect[2] = P[2];
                        srcRect[3] = P[3];
                    }
                    else
                    {
                        swap(width, height);
                        srcRect[0] = P[1];
                        srcRect[1] = P[2];
                        srcRect[2] = P[3];
                        srcRect[3] = P[0];
                    }

                    //通过面积筛选
                    double area = height * width;
                    if (area > 5000)
                    {
                        dstRect[0] = Point2f(0, 0);
                        dstRect[1] = Point2f(width, 0);
                        dstRect[2] = Point2f(width, height);
                        dstRect[3] = Point2f(0, height);
                        // 应用透视变换，矫正成规则矩形
                        Mat transform = getPerspectiveTransform(srcRect, dstRect);
                        Mat perspectMat;
                        warpPerspective(binBrightImg, perspectMat, transform, binBrightImg.size());
                        // 提取扇叶图片
                        Mat testim;
                        testim = perspectMat(Rect(0, 0, width, height));

                        cv::Point matchLoc;
                        double value;
                        Mat tmp1;
                        resize(testim, tmp1, Size(42, 20));

                        vector<double> Vvalue1;
                        vector<double> Vvalue2;
                        for (int j = 1; j <= 6; j++)
                        {
                            value = TemplateMatch(tmp1, templ[j], matchLoc, TM_CCOEFF_NORMED);
                            Vvalue1.push_back(value);
                        }
                        for (int j = 7; j <= 8; j++)
                        {
                            value = TemplateMatch(tmp1, templ[j], matchLoc, TM_CCOEFF_NORMED);
                            Vvalue2.push_back(value);
                        }
                        int maxv1 = 0, maxv2 = 0;

                        for (int t1 = 0; t1 < 6; t1++)
                        {
                            if (Vvalue1[t1] > Vvalue1[maxv1])
                            {
                                maxv1 = t1;
                            }
                        }
                        for (int t2 = 0; t2 < 2; t2++)
                        {
                            if (Vvalue2[t2] > Vvalue2[maxv2])
                            {
                                maxv2 = t2;
                            }
                        }


                        //预测是否是要打击的扇叶
                        if (Vvalue1[maxv1] > Vvalue2[maxv2] && Vvalue1[maxv1] > 0.6)
                        {
                            findTarget = true;
                            //查找装甲板
                            if (hierarchy[i][2] >= 0)
                            {
                                RotatedRect rect_tmp = minAreaRect(contours[hierarchy[i][2]]);
                                Point2f Pnt[4];
                                rect_tmp.points(Pnt);


                                float width = rect_tmp.size.width;
                                float height = rect_tmp.size.height;
                                if (height > width)
                                    swap(height, width);
                                float area = width * height;

                                if (height / width > maxHWRatio || area > maxArea || area < minArea)
                                {
                                    continue;
                                }
                                Point centerP = rect_tmp.center;
                                circle(_roiImg, centerP, 1, Scalar(0, 0, 255));
                                if (cirV.size() < 30)
                                {
                                    cirV.push_back(centerP);
                                }
                                else
                                {
                                    float R;
                                    //得到拟合的圆心
                                    CircleInfo2(cirV, cc, R);
                                    cirV.erase(cirV.begin());
                                    if (cc.x != 0 && cc.y != 0)
                                    {
                                        Mat rot_mat = getRotationMatrix2D(cc, 30, 1);

                                        float sinA = rot_mat.at<double>(0, 1); //sin(60);
                                        float cosA = rot_mat.at<double>(0, 0); //cos(60);
                                        float xx = -(cc.x - centerP.x);
                                        float yy = -(cc.y - centerP.y);
                                        Point2f resPoint = Point2f(cc.x + cosA * xx - sinA * yy,
                                                                   cc.y + sinA * xx + cosA * yy);
                                        circle(_roiImg, resPoint, 1, Scalar(0, 255, 0), 5);
                                        return resPoint;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            imshow("result", _roiImg);
            waitKey(1);
        }
    }




