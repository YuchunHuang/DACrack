#ifdef FIRSTDEMO_EXPORTS
#define FIRSTDEMO_API __declspec(dllexport)
#else
#define FIRSTDEMO_API __declspec(dllimport)
#endif
#pragma once
#include"dcs_logIf.h"
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include<string>
using namespace std;
vector<double> scales;
double sc_var = -1;
double scalea = 1;
double calMean(vector<double> cpradius);
double calvar(vector<double> vec);
void SplitString(const string& s, vector<string>& v, const string& c);
bool calsmallimg(const cv::Mat &srcImg, cv::Mat & smImg, int * pointscoor, vector<int>&smcoors);
bool calpaths(const cv::Mat &imgF, const vector<int> & smcoors, vector<cv::Point>& path, vector<cv::Point>& crackpoints, int &searchdirection);
int calcrackareabypath(const cv::Mat & connimg, cv::Mat& cimg, const vector<int> & smcoors, const vector<cv::Point>& path);
int callengthbyspur(const cv::Mat &spur);
void backtoorisize(const cv::Mat &smallerImg, cv::Mat & smImg, const vector<int> & smcoors);
extern "C" bool FIRSTDEMO_API GetIssuePointsInfo(const char * src, const char * txtroad, int * pointscoor, int * outpointscoor, double &cr_len, double &cr_wid, bool finb=false);//将计算得到裂缝点写入到txtroad表示的txt文件中
extern "C" bool FIRSTDEMO_API InputWidthInfo(const char * src, const char * txtroad, int * pointscoor, int * outpointscoor, double &cr_len, double &cr_wid, double trwid);
extern "C" void FIRSTDEMO_API CheckCopyright();