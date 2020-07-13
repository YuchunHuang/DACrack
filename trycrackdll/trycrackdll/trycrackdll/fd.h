#ifdef FIRSTDEMO_EXPORTS
#define FIRSTDEMO_API __declspec(dllexport)
#else
#define FIRSTDEMO_API __declspec(dllimport)
#endif
#pragma once
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include<string>
#include <ctime>
#include <stack> 
#include<string>
using namespace cv;
using namespace std;
using namespace cv::ocl;
double calMean(vector<double> cpradius);
void SplitString(const string& s, vector<string>& v, const string& c);
extern "C" void FIRSTDEMO_API  livewire_core(const char * src, const char * txtroad, int * pointscoor);
