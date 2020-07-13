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
#include <stdio.h>  
#include <stdlib.h> 
using namespace std;
vector<double> scales ;
struct Calibra_info{
	const char * imgroad;
	double scale;
};
double sc_var=-1;
double scalea = 1;
double calMean(vector<double> cpradius);
double calvar(vector<double> vec);
void SplitString(const string& s, vector<string>& v, const string& c);
bool calsmallimg(const cv::Mat &srcImg, cv::Mat & smImg, int * pointscoor, vector<int>&smcoors);
bool writeCrackInfo(string txtroad, double clength, double cwidth, double csquare, vector<cv::Point> crackpoints, int dex, int dey, int Xb, int Yb);
extern "C" bool FIRSTDEMO_API GetIssuePointsInfo(const char * src, const char * txtroad, int * pointscoor, int * outpointscoor, double &cr_len, double &cr_wid, bool finb=false);//������õ��ѷ��д�뵽txtroad��ʾ��txt�ļ���
extern "C" bool FIRSTDEMO_API InputWidthInfo(/*int cameraid,*/const char * src, const char * txtroad, int * pointscoor,double trwid);
//����������ţ�ͼƬ������Χ����ͼƬ·����ʵ�ʿ�ȣ�txtroad���춨����ļ���
//�궨����������д��ǰ·�����ṹ�屣��
//���ID��ͼ��λ�ã��춨����
extern "C" void FIRSTDEMO_API CheckCopyright();