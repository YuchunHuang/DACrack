// trycrackdll.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
//#include"tunnel_crack_rl.h"
#include"getissuespi.h"
#include"dcs_log.h"
#include<queue>
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
using namespace cv;
queue <cv::Point> points;
/*
00_00
00_02
00_78
01_00
01_1
01_05
01_5
2_3
04_09
*/
const char * src = "smimgs//01_5.jpg";
const char * troad = "630.txt";
cv::Mat src2;
int times = 0;
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	cv::Point sp;
	double c_length = 0;
	double c_width = 0;
	
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		/*�԰�ɫ��Ϊǰ�����д���
		�����Ϊ������
		1.����Сͼ��У���������ǰ����
		calsmallimg(src2, pointscoor, smcoors).copyTo(smImg);
		���а���{
		��ֵ��
		��ͨ����
		}
		���������ܱ����ѷ죬���ܺõ�ȥ�������㡣

		2.��Сͼ�������ʴ���У���������������
		ֱ�Ӷ�ԭͼ���������ֵ����Ч����
		*/
		sp =cv:: Point(x, y);//��ȡ��ǰ��
		points.push(sp);//һ�������ŵ�����ĵ�
		times++;
		if (points.size() == 3)
			points.pop();
		int  pointscoor[4];

		bool sx = true;
		int  outpoints[4] = { 0,0,0,0 };
		if (points.size() > 1) {
			if (!points.empty()) {
				pointscoor[0] = points.front().x;
				pointscoor[1] = points.front().y;
				pointscoor[2] = points.back().x;
				pointscoor[3] = points.back().y;
			}
			double len = 0;
			double wid = 0;
			//cout << "input 0 or 1 ,0 ������1 �궨" << endl;
			int a = 0;
			bool finb=false;
			if (times <= 1)
				a = 1;
			else {
				a = 0;
				finb = true;
			}
			//cin >> a;
			if (a == 0) {
				sx = GetIssuePointsInfo(src, troad, pointscoor, outpoints, len, wid/*,finb*/);//������õ��ѷ��д�뵽txtroad��ʾ��txt�ļ���
				if (sx == true)
				{																		//c_length += len;
				//c_width += wid;
					cout << "length: " << len;
					cout << "width: " << wid << endl;
					//g_pSyslog.WriteLog();
				}
				else {
					cout << "δ��⵽�ѷ�" << endl;
				}
			}
			//CheckCopyright();
			else if (a == 1) {
				cout << "Please input true width";
				double twid=3;
				//cin >> twid;
				bool sx = InputWidthInfo(src, troad, pointscoor, twid);
				if (sx == false)
					cout << "�ɽ����궨" << endl;
			}
		}
		if (sx == true) {
			cv::Point gz2(outpoints[2], outpoints[3]);
			circle(src2, sp, 2, cv::Scalar(0, 0, 255), -1);//Rԭʼ��
			circle(src2, gz2, 2, cv::Scalar(255, 0, 0), -1);//B�ڶ���������
		}
		cv::imshow("dst", src2);
	}
}

int main()
{
	//src = imread(src, 0);
	//LOG_INIT("main.log", LEVEL_INFO);
	//LOG_INFO("hello");
	//LOG_WARNING()
	src2 =cv::imread(src, 1);
	//src2.copyTo(newsrc);
	//cvNamedWindow("smImg", 1);
	//cvNamedWindow("src", 1);
	//cv::imshow("src", src);
	cvNamedWindow("dst", 1);
	cv::imshow("dst", src2);
	cvSetMouseCallback("dst", on_mouse, 0);
	cvWaitKey(0);
	cvDestroyAllWindows();
	_CrtDumpMemoryLeaks();
    return 0;
}

