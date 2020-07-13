// firstdemo.cpp : ���� DLL Ӧ�ó���ĵ���������

//
#include "stdafx.h"
#include "GIPI_getCenterline.h"
#include"getissuespi.h"
#include"GIPI_calcurv.h"
// readtxt.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include <iostream>
#include <fstream>
#include<string>
#include <vector>
#include  <stdio.h>
#include  <io.h>
using namespace std;
using namespace cv;
//�γ�һ���궨����ļ������ÿ������Ĳ���
//��¼�ѷ�Ҷ�ֵ�ı仯���ɣ��γ�һ���ѷ��ȶ�Ӧ�������ļ�
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
//void corPointByBinimg(const cv::Mat &srcImg, int * pointscoor) {//�������е��У��
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
//	thresholdIntegral(src, thimg, 0.15);//�����ֵ��
//	vector<CArea> ConArea;
//	cv::Mat connimg = cv::Mat::zeros(src.rows, src.cols, src.type());
//	FeatureCaputure(thimg, connimg, ConArea);//��ͨ���������³��ѷ�
//	fneartPoint(connimg, p1);//p1У����������ѷ��
//	//fneartPoint(connimg, p2);//p2У����������ѷ��
//	pointscoor[0] = x1;
//	pointscoor[1] = y1;
//	thimg.release();
//	connimg.release();
//	/*pointscoor[2] = x2;
//	pointscoor[3] = y2;*/
//}
bool calsmallimg(const cv::Mat &srcImg, cv::Mat & smImg, int * pointscoor, vector<int>&smcoors)
{//ԭͼ���ü�����Сͼ������ĵ����꣬�������Сͼ�е�����
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
	//cv::Mat imgF = normalize(srcImg);//��һ��ԭͼ��ʹ�ûҶȷ�Χ��0-1֮��
	//----У����
	cv::Mat src;
	cvtColor(srcImg, src, CV_BGR2GRAY);//��ԭͼת��Ϊ��ͨ���Ҷ�ͼ
									   //cv::Mat verimg = cv::Mat::zeros(src.rows, src.cols, src.type());
									   //verimg=verticalmeanfilter(src, 50);//˫���˲�
	cv::Mat thimg = cv::Mat::zeros(src.rows, src.cols, src.type());//Ԥ������һ��ͼ��������Ŷ�ֵ�����
	thresholdIntegral(src, thimg, 0.05);//�����ֵ��
	/*cv::imwrite("thimg.jpg", thimg);
	cv::imshow("ss", thimg);
	cv::waitKey(0);*/
	vector<CArea> ConArea;//������ͨ�������
	cv::Mat connimg = cv::Mat::zeros(src.rows, src.cols, src.type());//Ԥ������һ��ͼ�����������ͨ�������
																	 //clock_t start, end;
																	 //start = clock();
	FeatureCaputure(thimg, connimg, ConArea);//��ͨ���������³��ѷ�
	/*cv::imwrite("connimgbbb.jpg", connimg);
	cv::imshow("connimg", connimg);
	cv::waitKey(0);*/ 										 //end = clock();
	//-------------------
	//cv::imwrite("connimgbig.jpg",connimg);										 //cout << "��ͨ����" << (double)(end - start) / CLOCKS_PER_SEC << "s\n";
	if (ifacceptcorpoint(ConArea, p1) >= 1)
		fneartPoint(connimg, p1);//p1У����������ѷ��
	/*else
		return false;*/
	if (ifacceptcorpoint(ConArea, p2) >= 1)
		fneartPoint(connimg, p2);//p2У����������ѷ��
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
	int smp2x = 0, smp2y = 0;//��p1��ת����Сͼ�������
	int sdr = 0;
	int addrange = 50;
	if (abs(x2 - x1) >= abs(y2 - y1))
	{
		sdr = 0;//y����
	}
	else if (abs(y2 - y1)>abs(x2 - x1)) {
		sdr = 1;//x����
	}
	//	searchdirection = sdr;
	int dilatesize_u = 0;//�߽�
	int dilatesize_d = 0;
	int dex = 0;//�������ֵ��С�������϶����ڴ�ͼ�е�λ��
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
		srcImg(Range(dilatesize_u, dilatesize_d + 1), Range(xmin, xmax + 1)).copyTo(smImg);//�ٳ�Сͼ���ڼ���
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
	smcoors[0] = smp1x;//ԭ������
	smcoors[1] = smp1y;
	smcoors[2] = smp2x;
	smcoors[3] = smp2y;
	thimg.release();
	connimg.release();
	return true;
}

 //����ͼƬ����������ת��
bool coorsChangebyImgname(string paths,int & Xb,int &Yb) {
	//string paths = src;
	string::size_type iPos = paths.find_last_of('\\') + 1;
	string filename = paths.substr(iPos, paths.length() - iPos);
	//��ȡ������׺���ļ���	
	string name = filename.substr(0, filename.rfind("."));
	//cout << name << endl;
	vector<string> v;
	SplitString(name, v, "_");
	//����ͼ�����ƻ��Ŀ��ͼ���x,y��ʼ����
	Yb = atoi(v[0].c_str());
	Yb *= 1000;
	Xb = atoi(v[1].c_str());
	Xb *= 1000;
	//-----------------����ͼƬ����������ת��
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
bool GetIssuePointsInfo(const char * src, const char * txtroad, int * pointscoor, int * outpointscoor, double &cr_len, double &cr_wid,bool finb) {//Interface// bool�������������Ƿ���ñ궨���scale																																//��ֹ��ܣ���ȷ�����������У����˫��������м���
	LOG_INIT("gipi.log", LEVEL_INFO);//��ʼ����־�ļ�
	//--------1
	std::vector<cv::Point>  path, crackpoints;
	//path��ͨ��livewire��������ѷ����ϵ����е㡣crackpoints��pathÿ��5����ĳ��������ٵ������д���ļ�������UI����չʾ����path���������󣬲���д���ļ�
	//std::vector<cv::Point>  pathgt, crackpointsgt;
	vector<int> smcoors(4, 0);//������������ڿٳ���Сͼ�е�����(x1,y1,x2,y2)
	
	cv::Mat srcImg = imread(src, 1);//ԭʼ����ͼ
	cv::Mat smImg;//�������������ٳ���Сͼ
	int pscoor[4] = { pointscoor[0],pointscoor[1],pointscoor[2],pointscoor[3] };
	//*pscoor = pointscoor;
	bool calsuc = calsmallimg(srcImg, smImg, pscoor, smcoors);//�ٳ�Сͼ
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
	//-------���Сͼ��ȡ�ļ���
	//-------2.���ʼ���
	int dex = min(pscoor[0], pscoor[2]) - min(smcoors[0], smcoors[2]);
	int dey = min(pscoor[1], pscoor[3]) - min(smcoors[1], smcoors[3]);
	//-------������ָ���ܹ��޸�pscoor��ֵ�����pscoor��ֵ����˸��ݶ�ֵͼ���е�һ�ε������Ľ����
	//dex,dey��Сͼ���Ͻǵ��ڴ�ͼ�е�λ��
	cv::Mat imgCur;//�������ʺ�Ľ��ͼ
	calCurCost(smImg, imgCur);//��������ͼ������˫���˲������о�ֵ�����ʼ���
	LOG_INFO("���ʼ���ɹ�");
	cv::Mat thsholdimg, curimg_dev;//��ֵ����ͼ�����ʽ��ͼ�ķ�
	curimg_dev = 255 - imgCur;//��255������ͼȡ��
	thsholdimg = cv::Mat::zeros(imgCur.rows, imgCur.cols, imgCur.type());
	thresholdIntegral(curimg_dev, thsholdimg, 0.05);//��ֵ������
	LOG_INFO("��ֵ���ɹ�");
	vector<CArea> ConArea;//������ͨ�������
	cv::Mat connimg = cv::Mat::zeros(curimg_dev.rows, curimg_dev.cols, curimg_dev.type());//Ԥ������һ��ͼ�����������ͨ�������
	FeatureCaputure(thsholdimg, connimg, ConArea);//��ͨ���������³��ѷ�
	LOG_INFO("��ͨ�����ɹ�");
	/*cv::imwrite("imgCur2.jpg", imgCur);
	cv::imwrite("thsholdimg2.jpg", thsholdimg);
	cv::imwrite("connimg010.jpg", connimg);*/
	//cv::imshow("thsholdimg", thsholdimg);
	/*cv::imshow("connimg", connimg);
	cvWaitKey(0);*/
	maskcur(connimg, imgCur);//������ͨ��Ľ������Ĥ��������ǰ��������ֵ����
	LOG_INFO("����ͼ��Ĥ��ȡ�ɹ�");
	//---------------
	int x1, y1, x2, y2;
	x1 = smcoors[0];
	y1 = smcoors[1];
	x2 = smcoors[2];
	y2 = smcoors[3];
	cv::Point p1 = cv::Point(x1, y1);
	cv::Point p2 = cv::Point(x2, y2);
	//if (ifacceptcorpoint(ConArea, p1) >= 1&& ifacceptcorpoint(ConArea, p2) >= 1) {
	corpoint_afcur(imgCur, smcoors);//��������ͼУ��livewire����յ㣬�ڶ���У��
	LOG_INFO("�ڶ���У����ɹ�");
	//}
	cv::Mat nomcurimg_dev = normalize(255 - imgCur);
	int searchdirection = 0;
	//cv::imshow("nodev",nomcurimg_dev);
	calpaths(nomcurimg_dev, smcoors, path, crackpoints, searchdirection);//����livewire·��
	LOG_INFO("�ѷ�·������ɹ�");
	cv::Mat cimg;//�ѷ�����ͼ
	int area = calcrackareabypath(connimg, cimg, smcoors, path);//����������ͬʱ�õ��ѷ��ڶ�ֵͼ���ӳ������Ĳ��֣������������ѷ��ʵ��ֵ��
	cv::Mat spur;//�ѷ�Ǽ�ͼ
	cvHilditchThin1(cimg, spur);
	LOG_INFO("�Ǽ���ȡ�ɹ�");
	cv::Mat spurinbig = cv::Mat::zeros(connimg.rows, connimg.cols, connimg.type());
	backtoorisize(spur, spurinbig, smcoors);
	outpointscoor[0] = dex + smcoors[0];
	outpointscoor[2] = dex + smcoors[2];
	outpointscoor[1] = dey + smcoors[1];
	outpointscoor[3] = dey + smcoors[3];
	//-------------
	int Xb = 0, Yb = 0;
	coorsChangebyImgname(src, Xb, Yb);
	//-----------------����ͼƬ����������ת��
	/*----------------ͼƬ����������ʽ��64_32.jpgΪ���ӣ�64,32�ֱ�����ͼƬ��ȫ��ͼ���е�λ�ã�64�У�32�С�
	ÿ��Сͼ��size��1000*1000������ͼƬ�е��������ʼ��(64000,32000)�����Ǽ�������ѷ������Ҫ����(64000,32000)
	������ȫ��ͼ���е���ʵλ�ã�����ʵ����д���ļ��У���UI�����ȡ�������չʾ
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
	vector<Ancpoint> ap;//����ѷ�������Լ�ÿ����Ŀ��ֵ�����ݽṹ
	//CalculataWidth(spurinbig, connimg, ap);
	//CalculataWidth_var(spurinbig, connimg, smImg, ap);//���÷�������ѷ���
	CalculataWidth_gray(spurinbig, connimg, smImg, ap);								  //Ϊʲô��connimg����cimg��backtoorisize��connimg����Ϣ��ȫ���˵㴦��Ȼ�����ѷ���Ϣ����cimg��ʧ��һЩ��Ϣ
	//CalculataWidth_var_addgray(spurinbig, connimg, smImg, ap);
	//double cwidth = line_fit(ap);//����ֱ����ϣ����ѷ��������ϳɶ��ֱ�ߣ�ѡȡ����һ���ѷ�Ŀ����Ϊ������
	double cwidth = averageap(ap);
	
	cwidth *= 0.5;//�ֱ�����ÿ�����ش���0.5*0.5mm
	cwidth *= finb ? scalea : 1;//���ʹ�ñ궨��������ͳ���scalea
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
		LOG_INFO("ImgInfoTemp size :%s", "��Ϣд���ļ��ɹ�");
	}
	else {
		LOG_INFO("δʶ���ѷ�");
	}
	/*std::vector<Ancpoint> tmp;
	ap.swap(tmp);*/
	//----------д�ļ�����������������ѷ���Ϣд���ļ�����������������ѷ������
	LOG_INFO("һ���ѷ�ʶ�����");
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
				cout << "���ܴ��ļ���" << endl;
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