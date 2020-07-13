// firstdemo.cpp : 定义 DLL 应用程序的导出函数。

//
#include "stdafx.h"
#include "GIPI_getCenterline.h"
#include"getissuespi.h"
#include"GIPI_calcurv.h"
// readtxt.cpp : 定义控制台应用程序的入口点。
//

#include <iostream>
#include <fstream>
#include<string>
#include <vector>
#include  <stdio.h>
#include  <io.h>
using namespace std;
using namespace cv;
//形成一个标定结果文件，存放每个相机的参数
//记录裂缝灰度值的变化规律，形成一个裂缝宽度对应分数的文件
double calMean(vector<double> cpradius) {
	double cps = 0;
	for (int i = 0; i < cpradius.size(); i++) {
		cps += cpradius[i];
	}
	cps = cps / cpradius.size();
	return cps;
}
double calvar(vector<double> vec) {
	double cps = 0;
	for (int i = 0; i < vec.size(); i++) {
		cps += vec[i];
	}
	cps = cps / vec.size();
	double var = 0;
	for (int i = 0; i < vec.size(); i++) {
		var+=(vec[i]-cps)*(vec[i] - cps);
	}
	var /= vec.size();
	return var;
}
void SplitString(const string& s, vector<string>& v, const string& c)
{
	string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}
//void corPointByBinimg(const cv::Mat &srcImg, int * pointscoor) {//单独进行点的校正
//	//int x1Gt, y1Gt, x2Gt, y2Gt;
//	int imgheight = srcImg.rows;
//	int imgwidth = srcImg.cols;
//	int x1, y1;// , x2, y2;
//	x1 = pointscoor[0];
//	y1 = pointscoor[1];
//	/*x2 = pointscoor[2];
//	y2 = pointscoor[3];*/
//	cv::Point p1 = cv::Point(x1, y1);
//	//cv::Point p2 = cv::Point(x2, y2);
//	cv::Mat imgF = normalize(srcImg);
//	cv::Mat src;
//	cvtColor(srcImg, src, CV_BGR2GRAY);
//	cv::Mat thimg = cv::Mat::zeros(src.rows, src.cols, src.type());
//	thresholdIntegral(src, thimg, 0.15);//区域二值化
//	vector<CArea> ConArea;
//	cv::Mat connimg = cv::Mat::zeros(src.rows, src.cols, src.type());
//	FeatureCaputure(thimg, connimg, ConArea);//连通分析，留下长裂缝
//	fneartPoint(connimg, p1);//p1校正到最近的裂缝点
//	//fneartPoint(connimg, p2);//p2校正到最近的裂缝点
//	pointscoor[0] = x1;
//	pointscoor[1] = y1;
//	thimg.release();
//	connimg.release();
//	/*pointscoor[2] = x2;
//	pointscoor[3] = y2;*/
//}
bool calsmallimg(const cv::Mat &srcImg, cv::Mat & smImg, int * pointscoor, vector<int>&smcoors)
{//原图，裁剪出的小图，输入的点坐标，输入点在小图中的坐标
	int x1, y1, x2, y2;
	//int x1Gt, y1Gt, x2Gt, y2Gt;
	int imgheight = srcImg.rows;
	int imgwidth = srcImg.cols;
	x1 = pointscoor[0];
	y1 = pointscoor[1];
	x2 = pointscoor[2];
	y2 = pointscoor[3];
	cv::Point p1 = cv::Point(x1, y1);
	cv::Point p2 = cv::Point(x2, y2);
	//cv::Mat imgF = normalize(srcImg);//归一化原图，使得灰度范围在0-1之间
	//----校正点
	cv::Mat src;
	cvtColor(srcImg, src, CV_BGR2GRAY);//将原图转换为单通道灰度图
									   //cv::Mat verimg = cv::Mat::zeros(src.rows, src.cols, src.type());
									   //verimg=verticalmeanfilter(src, 50);//双边滤波
	cv::Mat thimg = cv::Mat::zeros(src.rows, src.cols, src.type());//预先声明一个图像用来存放二值化结果
	thresholdIntegral(src, thimg, 0.05);//区域二值化
	/*cv::imwrite("thimg.jpg", thimg);
	cv::imshow("ss", thimg);
	cv::waitKey(0);*/
	vector<CArea> ConArea;//定义连通域的数组
	cv::Mat connimg = cv::Mat::zeros(src.rows, src.cols, src.type());//预先声明一个图像用来存放连通分析结果
																	 //clock_t start, end;
																	 //start = clock();
	FeatureCaputure(thimg, connimg, ConArea);//连通分析，留下长裂缝
	/*cv::imwrite("connimgbbb.jpg", connimg);
	cv::imshow("connimg", connimg);
	cv::waitKey(0);*/ 										 //end = clock();
	//-------------------
	//cv::imwrite("connimgbig.jpg",connimg);										 //cout << "连通分析" << (double)(end - start) / CLOCKS_PER_SEC << "s\n";
	if (ifacceptcorpoint(ConArea, p1) >= 1)
		fneartPoint(connimg, p1);//p1校正到最近的裂缝点
	/*else
		return false;*/
	if (ifacceptcorpoint(ConArea, p2) >= 1)
		fneartPoint(connimg, p2);//p2校正到最近的裂缝点
	/*else
		return false;*/
	x1 = p1.x;
	y1 = p1.y;
	x2 = p2.x;
	y2 = p2.y;
	pointscoor[0] = x1;
	pointscoor[1] = y1;
	pointscoor[2] = x2;
	pointscoor[3] = y2;

	int smp1x = 0, smp1y = 0;
	int smp2x = 0, smp2y = 0;//将p1点转换到小图后的坐标
	int sdr = 0;
	int addrange = 50;
	if (abs(x2 - x1) >= abs(y2 - y1))
	{
		sdr = 0;//y方向
	}
	else if (abs(y2 - y1)>abs(x2 - x1)) {
		sdr = 1;//x方向
	}
	//	searchdirection = sdr;
	int dilatesize_u = 0;//边界
	int dilatesize_d = 0;
	int dex = 0;//坐标改正值，小矩形左上顶点在大图中的位置
	int dey = 0;
	if (sdr == 0) {
		if (y2 >= y1) {
			dilatesize_d = y2 + addrange<imgheight ? y2 + addrange : imgheight - 1;
			dilatesize_u = y1 - addrange>0 ? y1 - addrange : 0;
		}
		else {
			dilatesize_d = y1 + addrange<imgheight ? y1 + addrange : imgheight - 1;
			dilatesize_u = y2 - addrange>0 ? y2 - addrange : 0;
		}
		int xmin = 0, xmax = 0;
		if (x1 < x2) {
			/*xmin = x1;
			xmax = x2;*/
			xmax = x2 + addrange<imgwidth ? x2 + addrange : imgwidth - 1;
			xmin = x1 - addrange>0 ? x1 - addrange : 0;
		}
		else {
			/*xmin = x2;
			xmax = x1;*/
			xmax = x1 + addrange<imgwidth ? x1 + addrange : imgwidth - 1;
			xmin = x2 - addrange>0 ? x2 - addrange : 0;
		}
		srcImg(Range(dilatesize_u, dilatesize_d + 1), Range(xmin, xmax + 1)).copyTo(smImg);//抠出小图用于计算
																						   //imgGt(Range(dilatesize_u, dilatesize_d), Range(xmin, xmax)).copyTo(smallGt);
		smp1x = x1 - xmin;
		smp1y = y1 - dilatesize_u;
		smp2x = x2 - xmin;
		smp2y = y2 - dilatesize_u;
		dex = xmin;
		dey = dilatesize_u;
	}
	else {
		if (x2 >= x1) {
			dilatesize_d = x2 + addrange<imgwidth ? x2 + addrange : imgwidth - 1;
			dilatesize_u = x1 - addrange>0 ? x1 - addrange : 0;
		}
		else {
			dilatesize_d = x1 + addrange<imgwidth ? x1 + addrange : imgwidth - 1;
			dilatesize_u = x2 - addrange>0 ? x2 - addrange : 0;
		}
		int ymin = 0, ymax = 0;
		if (y1 < y2) {
			/*ymin = y1;
			ymax = y2;*/
			ymax = y2 + addrange<imgheight ? y2 + addrange : imgheight - 1;
			ymin = y1 - addrange>0 ? y1 - addrange : 0;
		}
		else {
			/*ymin = y2;
			ymax = y1;*/
			ymax = y1 + addrange<imgheight ? y1 + addrange : imgheight - 1;
			ymin = y2 - addrange>0 ? y2 - addrange : 0;
		}
		srcImg(Range(ymin, ymax + 1), Range(dilatesize_u, dilatesize_d + 1)).copyTo(smImg);
		//imgGt(Range(ymin, ymax), Range(dilatesize_u, dilatesize_d)).copyTo(smallGt);
		smp1x = x1 - dilatesize_u;
		smp1y = y1 - ymin;
		smp2x = x2 - dilatesize_u;
		smp2y = y2 - ymin;
		dex = dilatesize_u;
		dey = ymin;
	}
	smcoors[0] = smp1x;//原来点在
	smcoors[1] = smp1y;
	smcoors[2] = smp2x;
	smcoors[3] = smp2y;
	thimg.release();
	connimg.release();
	return true;
}

 //根据图片名进行坐标转换
bool coorsChangebyImgname(string paths,int & Xb,int &Yb) {
	//string paths = src;
	string::size_type iPos = paths.find_last_of('\\') + 1;
	string filename = paths.substr(iPos, paths.length() - iPos);
	//获取不带后缀的文件名	
	string name = filename.substr(0, filename.rfind("."));
	//cout << name << endl;
	vector<string> v;
	SplitString(name, v, "_");
	//根据图像名称获得目标图像的x,y起始坐标
	Yb = atoi(v[0].c_str());
	Yb *= 1000;
	Xb = atoi(v[1].c_str());
	Xb *= 1000;
	//-----------------根据图片名进行坐标转换
	return true;
}
bool writeCrackInfo(string txtroad,double clength,double cwidth,double csquare, vector<cv::Point> crackpoints,int dex,int dey,int Xb,int Yb) {
	if (clength != 0 && cwidth != 0) {
		ofstream out(txtroad, ios::app);
		string cl;
		string cw;
		string ca;
		cl = to_string(clength);
		cw = to_string(cwidth);
		ca = to_string(csquare);
		string crackissues = cl + " " + cw + " " + ca;
		string crpoints = "[";
		if (crackpoints.size() > 0) {
			for (int i = 0; i < crackpoints.size() - 1; i++) {
				int tempx = crackpoints[i].x + dex;
				int tempy = crackpoints[i].y + dey;
				tempx += Xb;
				tempy += Yb;
				string tx = to_string(tempx);
				string ty = to_string(tempy);
				int tempx2 = crackpoints[i + 1].x + dex;
				int tempy2 = crackpoints[i + 1].y + dey;
				tempx2 += Xb;
				tempy2 += Yb;
				string tx2 = to_string(tempx2);
				string ty2 = to_string(tempy2);
				crpoints = crpoints + "[" + tx + "," + ty + "," + tx2 + "," + ty2 + "]";
			}
		}
		crpoints = crpoints + "]";
		out << crackissues << " " << crpoints << endl;
		return true;
	}
	return false;
}
bool GetIssuePointsInfo(const char * src, const char * txtroad, int * pointscoor, int * outpointscoor, double &cr_len, double &cr_wid,bool finb) {//Interface// bool变量表明函数是否调用标定后的scale																																//拆分功能，明确单点输入进行校正，双点输入进行计算
	LOG_INIT("gipi.log", LEVEL_INFO);//初始化日志文件
	//--------1
	std::vector<cv::Point>  path, crackpoints;
	//path是通过livewire计算出的裂缝线上的所有点。crackpoints是path每隔5个点的抽样，减少点的数量写入文件方便在UI界面展示，因path数据量过大，不宜写入文件
	//std::vector<cv::Point>  pathgt, crackpointsgt;
	vector<int> smcoors(4, 0);//输入的两个点在抠出的小图中的坐标(x1,y1,x2,y2)
	
	cv::Mat srcImg = imread(src, 1);//原始输入图
	cv::Mat smImg;//根据输入输出点抠出的小图
	int pscoor[4] = { pointscoor[0],pointscoor[1],pointscoor[2],pointscoor[3] };
	//*pscoor = pointscoor;
	bool calsuc = calsmallimg(srcImg, smImg, pscoor, smcoors);//抠出小图
	//cv::imwrite("smimg.jpg", smImg);
	if(smImg.channels()>1)
		cvtColor(smImg, smImg, CV_BGR2GRAY);
	if (calsuc == false) {
		LOG_INFO("===calculate smallimg and correct point first time failed===");
		return false;
	}
	else {
		LOG_INFO("===calculate smallimg and correct point first time success===");
	}
	//-------完成小图截取的计算
	//-------2.曲率计算
	int dex = min(pscoor[0], pscoor[2]) - min(smcoors[0], smcoors[2]);
	int dey = min(pscoor[1], pscoor[3]) - min(smcoors[1], smcoors[3]);
	//-------参数传指针能够修改pscoor的值，因此pscoor的值变成了根据二值图进行第一次点修正的结果。
	//dex,dey是小图左上角点在大图中的位置
	cv::Mat imgCur;//计算曲率后的结果图
	calCurCost(smImg, imgCur);//计算曲率图，包含双边滤波，按列均值，曲率计算
	LOG_INFO("曲率计算成功");
	cv::Mat thsholdimg, curimg_dev;//二值化后图像，曲率结果图的反
	curimg_dev = 255 - imgCur;//用255减曲率图取反
	thsholdimg = cv::Mat::zeros(imgCur.rows, imgCur.cols, imgCur.type());
	thresholdIntegral(curimg_dev, thsholdimg, 0.05);//二值化函数
	LOG_INFO("二值化成功");
	vector<CArea> ConArea;//定义连通域的数组
	cv::Mat connimg = cv::Mat::zeros(curimg_dev.rows, curimg_dev.cols, curimg_dev.type());//预先声明一个图像用来存放连通分析结果
	FeatureCaputure(thsholdimg, connimg, ConArea);//连通分析，留下长裂缝
	LOG_INFO("连通分析成功");
	/*cv::imwrite("imgCur2.jpg", imgCur);
	cv::imwrite("thsholdimg2.jpg", thsholdimg);
	cv::imwrite("connimg010.jpg", connimg);*/
	//cv::imshow("thsholdimg", thsholdimg);
	/*cv::imshow("connimg", connimg);
	cvWaitKey(0);*/
	maskcur(connimg, imgCur);//利用连通后的结果做掩膜，将属于前景的曲率值保留
	LOG_INFO("曲率图掩膜提取成功");
	//---------------
	int x1, y1, x2, y2;
	x1 = smcoors[0];
	y1 = smcoors[1];
	x2 = smcoors[2];
	y2 = smcoors[3];
	cv::Point p1 = cv::Point(x1, y1);
	cv::Point p2 = cv::Point(x2, y2);
	//if (ifacceptcorpoint(ConArea, p1) >= 1&& ifacceptcorpoint(ConArea, p2) >= 1) {
	corpoint_afcur(imgCur, smcoors);//根据曲率图校正livewire起点终点，第二次校正
	LOG_INFO("第二次校正点成功");
	//}
	cv::Mat nomcurimg_dev = normalize(255 - imgCur);
	int searchdirection = 0;
	//cv::imshow("nodev",nomcurimg_dev);
	calpaths(nomcurimg_dev, smcoors, path, crackpoints, searchdirection);//计算livewire路径
	LOG_INFO("裂缝路径计算成功");
	cv::Mat cimg;//裂缝区域图
	int area = calcrackareabypath(connimg, cimg, smcoors, path);//种子生长。同时裁掉裂缝在二值图中延长出来的部分，否则计算出的裂缝比实际值长
	cv::Mat spur;//裂缝骨架图
	cvHilditchThin1(cimg, spur);
	LOG_INFO("骨架提取成功");
	cv::Mat spurinbig = cv::Mat::zeros(connimg.rows, connimg.cols, connimg.type());
	backtoorisize(spur, spurinbig, smcoors);
	outpointscoor[0] = dex + smcoors[0];
	outpointscoor[2] = dex + smcoors[2];
	outpointscoor[1] = dey + smcoors[1];
	outpointscoor[3] = dey + smcoors[3];
	//-------------
	int Xb = 0, Yb = 0;
	coorsChangebyImgname(src, Xb, Yb);
	//-----------------根据图片名进行坐标转换
	/*----------------图片名称命名方式以64_32.jpg为例子，64,32分别代表该图片在全幅图像中的位置，64行，32列。
	每个小图的size是1000*1000，即该图片中点的坐标起始于(64000,32000)，我们计算出的裂缝点坐标要加上(64000,32000)
	才是在全幅图像中的真实位置，将真实坐标写入文件中，由UI界面读取坐标进行展示
	-----------------*/
	double clength = double(callengthbyspur(spur));
	/*cv::imshow("connimg", connimg);
	cv::imshow("cimg", cimg);
	cv::imshow("spurinbig", spurinbig);
	cvWaitKey(0);*/
	/*cv::imwrite("thsholdimg0002.jpg", thsholdimg);
	cv::imwrite("smImg0002.jpg", smImg);
	cv::imwrite("cimg0002.jpg", cimg);
	cv::imwrite("spurinbig0002.jpg", spurinbig);*/
	clength /= 1000.0;
	vector<Ancpoint> ap;//存放裂缝点坐标以及每个点的宽度值的数据结构
	//CalculataWidth(spurinbig, connimg, ap);
	//CalculataWidth_var(spurinbig, connimg, smImg, ap);//利用方差计算裂缝宽度
	CalculataWidth_gray(spurinbig, connimg, smImg, ap);								  //为什么用connimg不用cimg的backtoorisize，connimg的信息更全，端点处仍然保有裂缝信息，而cimg丢失了一些信息
	//CalculataWidth_var_addgray(spurinbig, connimg, smImg, ap);
	//double cwidth = line_fit(ap);//利用直线拟合，将裂缝点依次拟合成多段直线，选取最宽的一段裂缝的宽度作为整体宽度
	double cwidth = averageap(ap);
	
	cwidth *= 0.5;//分辨率是每个像素代表0.5*0.5mm
	cwidth *= finb ? scalea : 1;//如果使用标定后参数，就乘以scalea
	double csquare = clength*0.2;
	cr_len = clength;
	if (clength == 0)
		cr_wid = 0;
	else {
		cr_wid = cwidth;
	}
	thsholdimg.release();
	connimg.release();
	spurinbig.release();
	spur.release();
	//cwimg.release();
	//--------------
	bool writecisuc=writeCrackInfo(txtroad,clength,cwidth,csquare,crackpoints,dex,dey,Xb,Yb);
	if(writecisuc){
		LOG_INFO("ImgInfoTemp size :%s", "信息写入文件成功");
	}
	else {
		LOG_INFO("未识别到裂缝");
	}
	/*std::vector<Ancpoint> tmp;
	ap.swap(tmp);*/
	//----------写文件操作，将计算出的裂缝信息写入文件，包括长宽面积、裂缝点坐标
	LOG_INFO("一次裂缝识别完成");
	return true;
}
bool InputWidthInfo(const char * src, const char * txtroad, int * pointscoor,double trwid) {
	double cr_len = 0;
	double cr_wid = 0;
	int  outpointscoor[4] = { 0,0,0,0 };
	bool ifget=GetIssuePointsInfo(src, txtroad, pointscoor, outpointscoor, cr_len, cr_wid,false);
	double oldvar = sc_var;
	bool next=true;
	if (ifget) {
		double sc = trwid / cr_wid;
		scales.push_back(sc);
		if (scales.size() > 1) {
			double var = calvar(scales);
			sc_var = var;
		}
		if (scales.size() > 1 && oldvar != -1 && sc_var != -1 && sc_var - oldvar <= 2) {
			scalea = calMean(scales);
			next = false;
			Calibra_info calinfo;
			calinfo.imgroad = src;
			calinfo.scale = scalea;
			/*FILE *fp;
			fp = fopen(,"w+");*/
			FILE* pf;
			if ((pf = fopen("11_11.txt", "ab+")) == NULL)
			{
				cout << "不能打开文件！" << endl;
			}
			fwrite(&calinfo, sizeof(calinfo), 1, pf);
			rewind(pf);
			fclose(pf);
			FILE *fp;
			Calibra_info sb;
			fp = fopen("11_11.txt", "rb");
			if (!fp)
			{
				printf("errror!\n");
				exit(-1);
			}
			fread(&sb, sizeof(sb), 1, fp);
			cout << sb.imgroad<<endl;
			cout << sb.scale << endl;
			fclose(fp);
		}	
	}
	return next;
}
void CheckCopyright() {
	ofstream out("copyright.txt", ios::app);
	string copyright = "Copyright 2020 WHU";
	out << copyright;
	out.close();
}