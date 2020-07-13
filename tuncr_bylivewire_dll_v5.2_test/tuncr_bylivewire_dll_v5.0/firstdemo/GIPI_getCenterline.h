/*
Copyright (C), 2020,WHU.
   File name:GIPI_getCenterline.h     /
   Created by:Lizhe
   Version:1.0.0        
   Date:2020-6-30
   Description:
		本文件用于声明和定义中心线提取模块以及裂缝信息计算模块中使用到的函数与结构体。
   
   Dependencies：

   ChangeLog:
   （1）2020-6-30 changed by Lizhe
        将这两个模块中使用到的函数全部放在本头文件中声明，有助于后续的模块化分工与代码管理。
		  
   // 修改历史记录列表，每条修改记录应包括修改日期、修改
   // 者及修改内容简述  

CalculataWidth_gray
*/

#ifndef _LW_OCV_
#define _LW_OCV_

#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include<string>
#include <cstdio>
#include<algorithm>

using namespace std;
using namespace cv;
typedef struct Ancpoint {
	Point anc;
	double width;
}Ancpoint;
struct Sllines {
	double wid;
	double k;
	double b;
};
///////////////////////////////////////////////////
cv::Mat calcImgGrad(const cv::Mat& img);
cv::Mat calcCanny(const cv::Mat& img);
cv::Mat normalize(const cv::Mat &img);
//void use_superpixels(cv::Mat &image);
//cv::Mat calCurvateCostFun(const cv::Mat &img);
cv::Mat calcLiveWireCostFun(const cv::Mat& img);//计算用livewire的损失图
cv::Mat normImage(const cv::Mat& img);
bool thresholdIntegral(cv::Mat &inputMat, cv::Mat &outputMat, double T);
//void findPoint(const cv::Mat& img, cv::Point &stpoint);
void fneartPoint(cv::Mat& img, cv::Point &stpoint);
void correctPoint(const cv::Mat& img, cv::Point &stpoint);//校正起始点，从附近损失最小的地方开始

double findMinId(vector<double> m);
//double searchRadiu(const cv::Mat &imgGt, cv::Point curpt, int searchdir);
//double calcuwidth(const cv::Mat &img, vector<cv::Point >path, vector<cv::Point >pathgt, int sd);//利用裂缝的中心线和边缘线计算裂缝宽度

//int calcrackareabypath_andorimg(const cv::Mat & smImg, cv::Mat& cimg, const vector<int> & smcoors, const vector<cv::Point>& path);

///////////////////////////////////////////////////
// Structure definitin of the active list entries
struct SEntry {
    short    sX;         // X-coordinate
    short    sY;         // Y-coordinate
    long     lLinInd;    // Linear index from x and y for 1D-array
    float    flG;        // The current cost from seed to (X,Y)^T
};

// Inline function to determin minimum of two numbers
inline long ifMin(long a, long b)
{
    return a < b ? a : b;
}

// Inline function to determin maximum of two numbers
inline long ifMax(long a, long b)
{
    return a > b ? a : b;
}

// Inline function to calculate linear index from subscript indices.
//inline long ifLinInd(short sX, short sY, short sNY)
inline long ifLinInd(short sX, short sY, short sNX)
{
//    return long(sX)*long(sNY) + long(sY);
    return long(sY)*long(sNX) + long(sX);
}

//
//  FUNCTION fFindMinG
//
//  Get the Index of the vector entry with the smallest dQ in pV
//
long fFindMinG(SEntry *pSList, long lLength);

//
//  FUNCTION fFindLinInd
//
//  Get the Index of the list entry in *pSList lLinInd == lInd
//
long fFindLinInd(SEntry *pSList, long lLength, long lInd);

void calcLiveWireP(const cv::Mat& imgS,
                   int *smpoints,
                   cv::Mat& iPX, cv::Mat& iPY,
                   double dRadius = 10000.0, int LISTMAXLENGTH=1000000);

void calcLiveWireGetPath(const cv::Mat& ipx,
                         const cv::Mat& ipy,
                         cv::Point pxy, int dex, int dey,
                         std::vector<cv::Point>& path, int iMAXPATH = 1000000);
bool calpaths(const cv::Mat &imgF, const vector<int> & smcoors, vector<cv::Point>& path, vector<cv::Point>& crackpoints, int &searchdirection);
int calcrackareabypath(const cv::Mat & connimg, cv::Mat& cimg, const vector<int> & smcoors, const vector<cv::Point>& path);
void cvHilditchThin1(const cv::Mat & src, cv::Mat &spur);
int callengthbyspur(const cv::Mat &spur);
void backtoorisize(const cv::Mat &smallerImg, cv::Mat & smImg, const vector<int> & smcoors);
cv::Point calcIdealAnchor(const cv::Mat& imgS, cv::Point pxy, int rad=4);

//--------宽度计算函数
void CalculataWidth(cv::Mat &spur, const cv::Mat preimg, vector<Ancpoint>& ap);
void CalculataWidth_var(cv::Mat &spur, cv::Mat preimg, cv::Mat orimg, vector<Ancpoint>& ap);
void CalculataWidth_gray(cv::Mat &spur, cv::Mat preimg, cv::Mat orimg, vector<Ancpoint>& ap);
void CalculataWidth_var_addgray(cv::Mat &spur, cv::Mat preimg, cv::Mat  orimg, vector<Ancpoint>& ap);
double line_fit(vector<Ancpoint>& ap/*, const cv::Mat modimg*/);
double averageap(vector<Ancpoint>& ap);
//--------
#endif
