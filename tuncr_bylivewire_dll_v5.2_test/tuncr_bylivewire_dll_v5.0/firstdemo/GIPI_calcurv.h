#pragma once
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
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
typedef struct connectArea
{
	int area;
	Rect rectangle;
	int lable;
	double  perimeter;
	//double R;
	string information;
	Point weight;//Ŀ������
				 //double T;//�쳤��
}CArea;
struct seednaindex {
	int seedsnums;
	int index;
};
struct dists {
	double distlab;
	double distxy;
};
bool verticalmeanfilter(const cv::Mat& src, cv::Mat &dst, double sens);
//-------------------qvlv
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);
cv::Mat padarray(const cv::Mat &Image, int RowPad, int ColPad);
void addzero(const cv::Mat &input, cv::Mat& dst, int addrows, int addcols);
cv::Mat expmartix(const cv::Mat& input);
cv::Mat cosmartix(const cv::Mat& input);
cv::Mat sinmartix(const cv::Mat &input);
cv::Mat absmartix(const cv::Mat &input);
cv::Mat sqrtmartix(const cv::Mat & input);
cv::Mat sign(const cv::Mat & input);
cv::Mat bigger(const cv::Mat & input, const cv::Mat & input2);
cv::Mat repl(const cv::Mat & old, const cv::Mat & newi, const cv::Mat & model);
cv::Mat orepl(const cv::Mat & old, double newo, const cv::Mat & model);
cv::Mat mat2gray(const cv::Mat & input);
cv::Mat im2int(const cv::Mat & input);
double summar(const cv::Mat & input);
void selectpart(const cv::Mat &input, cv::Mat &output, int up, int down, int left, int right);
void Conv2d_fft_first(const cv::Mat & image, const  cv::Mat & h, const  cv::Mat  & h2, const cv::Mat & h3, cv::Mat &output, cv::Mat &output2, cv::Mat &output3);
void Conv2d_fft_second(const cv::Mat &  image, const cv::Mat & h, const  cv::Mat & h2, cv::Mat &output, cv::Mat &output2);
char * CurvatureIndex2D(const cv::Mat & I1, cv::Mat & dst, int nrScales, int  preservePolarity, double min_scale);
bool Curvature(const cv::Mat& inputimg, cv::Mat& retimg, int n);
bool calCurCost(const cv::Mat & src, cv::Mat& dst);
void BinaryImageAdvance(cv::Mat bmp_org, cv::Mat &bmp_binaried, int theshold);
void FeatureCaputure(cv::Mat  img, cv::Mat & dst, vector<CArea> &ConArea);
//void Repaint(Mat srcimg, string smsrc, Rect Shape, string dst);
//-----------------------
bool maskcur(const cv::Mat &thesholdimg, cv::Mat & curimg);
int ifacceptcorpoint(const vector<CArea> &ConArea, const cv::Point& p);
void corpoint_afcur(const cv::Mat &curimg, vector<int>&smcoors);