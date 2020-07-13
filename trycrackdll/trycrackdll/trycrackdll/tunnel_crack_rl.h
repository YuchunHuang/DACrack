// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 FIRSTDEMO_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// FIRSTDEMO_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
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
typedef struct connectArea
{
	int area;
	Rect rectangle;
	int lable;
	double  perimeter;
	double R;
	string information;
	Point weight;//目标重心
	double T;//伸长度
}CArea;
struct seednaindex {
	int seedsnums;
	int index;
};
struct dists {
	double distlab;
	double distxy;
};
typedef struct Ancpoint {
	Point anc;
	int width;
}Ancpoint;
const double pi = 3.14159265358979323846;
void medBlur(string src, string dst, int ksize);
void DrawRectangle(cv::Mat& img, cv::Rect box);
void theshold(string src, string dst);
void BinaryImageAdvance(Mat bmp_org, Mat bmp_binaried, int theshold);
void FeatureCaputure(string src, string dst, vector<CArea> ConArea, int perjudge);
//-------------------qvlv
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);
cv::Mat padarray(cv::Mat Image, int RowPad, int ColPad);
cv::Mat addzero(cv::Mat input, int addrows, int addcols);
cv::Mat expmartix(cv::Mat input);
cv::Mat cosmartix(cv::Mat input);
cv::Mat sinmartix(cv::Mat input);
cv::Mat absmartix(cv::Mat input);
cv::Mat sqrtmartix(cv::Mat input);
cv::Mat sign(cv::Mat input);
cv::Mat bigger(cv::Mat input, cv::Mat input2);
cv::Mat repl(cv::Mat old, cv::Mat newi, cv::Mat model);
cv::Mat orepl(cv::Mat old, double newo, cv::Mat model);
cv::Mat mat2gray(cv::Mat input);
cv::Mat im2int(cv::Mat input);
double summar(cv::Mat input);
cv::Mat selectpart(cv::Mat input, int up, int down, int left, int right);
void Conv2d_fft_first(cv::Mat image, cv::Mat h, cv::Mat h2, cv::Mat h3, cv::Mat &output, cv::Mat &output2, cv::Mat &output3);
void Conv2d_fft_second(cv::Mat image, cv::Mat h, cv::Mat h2, cv::Mat &output, cv::Mat &output2);
char * CurvatureIndex2D(cv::Mat I1, string dst, int nrScales, int  preservePolarity, double min_scale);
char * Curvature(string inputimg, string dst, int n);
void cutspurs(string src, string dst);
void Repaint(Mat srcimg, string smsrc, Rect Shape, string dst);
//-----------------------

//----------------实现两种方法的处理
//void w1_comslic(string cpath, string coutimg, string coutimg2);

//--------------
void process(const char * src, const char * txtroad, int * rectrange);

extern "C" void FIRSTDEMO_API  GetIssueRectInfo(const char * src, const char * txtroad, int * rectrange);

void cvHilditchThin1(const char * src1, const char * dst1);
void CalculataWidth(const char * src, const char * src2, vector<Ancpoint>* ap);
void BinaryImageAdvance2(Mat bmp_org, Mat bmp_binaried, int theshold);
int findamaxid(vector<int>cnums);
string  kuaiorhz(string calcutype, double & square);
template<typename T>
vector<size_t> sort_indexes(const vector<T> &v);
void FeatureCaputure_deletesmalls(const char * src, const char * dst, vector<CArea>& ConArea);
vector<size_t> FeatureCaputure(const char * src, vector<CArea>& ConArea);
void FeatureCaputure_big(const char * src, const char * dst, vector<CArea>& ConArea);
//double fmax(vector<int> b);
void getDir(string carpeta, vector<string> &f);
int calculatesq_deg(vector<CArea> ConArea);
string judgedimensions(string type, float width, int aveareas);
void calcuparas(string path, vector<double>& issues, vector<Point> & crackpixels, int xb, int yb);
void DrawRectangle2(cv::Mat& img, cv::Rect box);
void quantization(string src2v, const char * imgroad, const char * txtroad, int * rectrange);

