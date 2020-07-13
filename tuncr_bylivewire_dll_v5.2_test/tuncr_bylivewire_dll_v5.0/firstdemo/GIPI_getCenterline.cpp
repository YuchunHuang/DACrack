#include "stdafx.h"
#include "GIPI_getCenterline.h"
#include "GIPI_calcurv.h"
const double pi = 3.14159265358979323846;
cv::Mat calcImgGrad(const cv::Mat &img) {
    cv::Mat ret;
    cv::Mat gx, gy, ga;
    cv::Sobel(img, gx, CV_64F, 1,0);
    cv::Sobel(img, gy, CV_64F, 0,1);
    cv::cartToPolar(gx,gy, ret,ga);
    double vMax;
    cv::minMaxLoc(ret, NULL, &vMax, NULL,NULL);
    ret=1.0-ret/vMax;
    return ret;
}


cv::Mat calcCanny(const cv::Mat &img) {
    cv::Mat ret, tmp, imgb;
    img.copyTo(imgb);
    //    cv::blur(img, imgb, cv::Size(3,3));
    //    cv::adaptiveBilateralFilter(img,imgb,cv::Size(7,7),200);
    int param=9;
    cv::bilateralFilter(img,imgb,param,param*2,param/2.0);
    cv::Scalar vMean = cv::mean(imgb);
   // std::cout << "mean = " << vMean << std::endl;
    //    tmp.reshape(0,1);
    //    cv::sort(tmp,tmp,CV_SORT_ASCENDING);
    //    double vMedian  = (double)tmp.at<uchar>(0,tmp.cols/2);
    double vMedian  = vMean[0];
    double sigma    = 0.3;
    double lower    = std::max(0.0, (1.0 - sigma) * vMedian);
    double upper    = std::min(255.0, (1.0 + sigma) * vMedian);
  //  std::cout << "lower=" << lower << ", upper=" << upper << std::endl;
    cv::Canny(imgb, ret, lower, upper, 3);
    ret.convertTo(ret,CV_64F);
    double vMax;
    cv::minMaxLoc(ret, NULL, &vMax, NULL,NULL);
    return (1.0 - ret/vMax);
}
cv::Mat normalize(const cv::Mat &img) {
	cv::Mat imgg = img;
	if (img.channels() == 3) {
		cv::cvtColor(img, imgg, CV_BGR2GRAY);
	}
	cv::Mat ret;
	cv::normalize(imgg, ret, 0, 1, CV_MINMAX, CV_64F);
	return ret;
}
//void use_superpixels(cv::Mat &image) {
//	cv::Mat image_contours = cv::imread("dog_contours.bmp", CV_LOAD_IMAGE_GRAYSCALE);
//	for (int i = 0; i < image_contours.rows; i++) {
//		for (int j = 0; j < image_contours.cols; j++) {
//			if (image_contours.at<unsigned char>(i, j) == 255) {
//				image.at<double>(i, j) = 1;
//			}
//		}
//	}
//}
bool thresholdIntegral( cv::Mat &inputMat, cv::Mat &outputMat, double T)
{
	//区域二值化
	// accept only char type matrices
	CV_Assert(!inputMat.empty());
	CV_Assert(inputMat.depth() == CV_8U);
	CV_Assert(inputMat.channels() == 1);
	/*CV_Assert(!outputMat.empty());
	CV_Assert(outputMat.depth() == CV_8U);
	CV_Assert(outputMat.channels() == 1);*/

	// rows -> height -> y
	int nRows = inputMat.rows;
	// cols -> width -> x
	int nCols = inputMat.cols;

	// create the integral image
	cv::Mat sumMat;
	cv::integral(inputMat, sumMat);//计算积分图

	CV_Assert(sumMat.depth() == CV_32S);
	CV_Assert(sizeof(int) == 4);

	int S = MAX(nRows, nCols) / 8;
	//double T = 0.15;

	// perform thresholding
	int s2 = S / 2;
	int x1, y1, x2, y2, count, sum;

	// CV_Assert(sizeof(int) == 4);
	int *p_y1, *p_y2;
	uchar *p_inputMat, *p_outputMat;

	for (int i = 0; i < nRows; ++i)
	{
		y1 = i - s2;
		y2 = i + s2;

		if (y1 < 0) {
			y1 = 0;
		}
		if (y2 >= nRows) {
			y2 = nRows - 1;
		}

		p_y1 = sumMat.ptr<int>(y1);
		p_y2 = sumMat.ptr<int>(y2);
		p_inputMat = inputMat.ptr<uchar>(i);
		p_outputMat = outputMat.ptr<uchar>(i);

		for (int j = 0; j < nCols; ++j)
		{
			// set the SxS region
			x1 = j - s2;
			x2 = j + s2;

			if (x1 < 0) {
				x1 = 0;
			}
			if (x2 >= nCols) {
				x2 = nCols - 1;
			}

			count = (x2 - x1)*(y2 - y1);

			// I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
			sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

			if ((int)(p_inputMat[j] * count) < (int)(sum*(1.0 - T)))//取反
				p_outputMat[j] = 255;
			else
				p_outputMat[j] = 0;
		}
	}
	return true;
}
void fneartPoint(cv::Mat& img, cv::Point &stpoint) {
	//-----------------integral二值化原图，找到前景中曲率最大点修正之//xxxxxxxxxx
	//找到前景中最近点
	cv::Mat imgc;
	img.copyTo(imgc);
	if (img.channels() == 3) {
		cv::cvtColor(img, imgc, CV_BGR2GRAY);
	}
	else {
		img.copyTo(imgc);
	}
	/*cv::Mat outimg;
	thresholdIntegral(img, outimg, 0.15);*/
	int mx = stpoint.x;
	int my = stpoint.y;
	int rng = 10;
	int lx = mx - rng > 0 ? mx - rng : 0;
	int rx = mx + rng < img.cols ? mx + rng : img.cols - 1;
	int uy = my - rng > 0 ? my - rng : 0;
	int dy = my + rng < img.rows ? my + rng : img.rows - 1;
	//cv::Mat smallF;
	//imgc(Range(uy, dy), Range(lx, rx)).copyTo(smallF);//抠出小图用于计算
	//smallF.convertTo(smallF, CV_64FC1);
	//double *pdG = (double*)(smallF.data);
	unsigned char * pdG = imgc.data;

	int minindex = ifLinInd(mx, my, imgc.cols);
	int mindist = 20000;
	for (int j = uy; j <= dy; j++) {
		for (int i = lx; i <= rx; i++) {
			int index = ifLinInd(i, j, imgc.cols);
			if (pdG[index] > 200) {//前景
				int dist = sqrt((mx - i)*(mx - i) + (my - j)*(my - j));
				if (dist < mindist) {
					mindist = dist;
					minindex = index;
				}
			}
		}
	}
	int nx = (minindex % imgc.cols);
	int ny = (minindex / imgc.cols);
	stpoint = cv::Point(nx, ny);
}
//void findPoint( cv::Mat& img, cv::Point &stpoint) {
//	//integral二值化原图，找到前景中曲率最大点修正之
//	cv::Mat imgc;
//	img.copyTo(imgc);
//	if (img.channels() == 3) {
//		cv::cvtColor(img, imgc, CV_BGR2GRAY);
//	}
//	cv::Mat outimg;
//	thresholdIntegral(img, outimg, 0.15);
//	int mx = stpoint.x;
//	int my = stpoint.y;
//	int rng = 100;
//	int lx = mx - rng > 0 ? mx - rng : 0;
//	int rx = mx + rng < img.cols ? mx + rng : img.cols - 1;
//	int uy = my - rng > 0 ? mx - rng : 0;
//	int dy = my + rng < img.rows ? my + rng : img.rows - 1;
//	double *pdG = (double*)(imgc.data);
//
//
//	int minindex = ifLinInd(mx, my, imgc.cols);
//	int mindist = 20000;
//	for (int i = lx; i <= rx; i++) {
//		for (int j = uy; j <= dy; j++) {
//			int index = ifLinInd( i, j, imgc.cols);
//			if (pdG[index] < 5) {//前景
//				int dist = sqrt((mx - i)*(mx - i) + (my - j)*(my - j));
//				if (dist < mindist) {
//					mindist = dist;
//					minindex = index;
//				}
//			}
//		}
//	}
//	int nx = minindex % imgc.cols;
//	int ny = minindex / imgc.cols;
//	stpoint = cv::Point(nx, ny);
//}
void correctPoint(const cv::Mat& img ,cv::Point &stpoint) {
	int px = stpoint.x;
	int py = stpoint.y;
	cv::Mat imgc;
	img.copyTo(imgc);
	if (img.channels() == 3) {
		cv::cvtColor(img, imgc, CV_BGR2GRAY);
	}
	double *pdG = (double*)(imgc.data);
	

	vector <int > bestindexs;
	for (int it = 0; it < 5; it++) {
		vector<int >indexes;
		vector<double> costs;
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				if (px + i < imgc.cols&&px + i >= 0 && py + j < imgc.rows&&py + j >= 0) {
					int index = ifLinInd(px + i, py + j, imgc.cols);
					indexes.push_back(index);
					costs.push_back(pdG[index]);
				}
			}
		}
		int minid = int(findMinId(costs));
		bestindexs.push_back(indexes[minid]);
		if (it > 0) {
			if (bestindexs[it] == bestindexs[it - 1])
				break;
		}
		int nx = indexes[minid] % imgc.cols;
		int ny = indexes[minid] / imgc.cols;
		stpoint = cv::Point(nx, ny);
		costs.clear();
		indexes.clear();
	}
	bestindexs.clear();
}
//cv::Mat calcLWCurvateCostFun(const cv::Mat &img) {
//	cv::Mat imgg = img;
//	if (img.channels() == 3) {
//		cv::cvtColor(img, imgg, CV_BGR2GRAY);
//	}
//	imgg.convertTo(imgg, CV_64FC1, 1);
//	int R, C;
//	R = imgg.size().height;
//	C = imgg.size().width;
//	cv::Mat imgI;
//	imgI = imgg;
//	imgI.convertTo(imgI, CV_64FC1, 1);
//	imgI = -imgI;//裂缝取负数
//	double sigma;
//	int siz = 0;
//	double t = 1.5;
//	sigma = t*(pow(sqrt(2), t - 1));
//	siz = round(sigma * 4) + 1;
//	if (siz % 2 == 0) {
//		siz = siz + 1;
//	}
//
//	cv::Mat X, Y;
//	meshgrid(cv::Range(-siz, siz), cv::Range(-siz, siz), X, Y);
//	//cout << X;
//	X.convertTo(X, CV_64FC1, 1);
//	Y.convertTo(Y, CV_64FC1, 1);
//	double theta1 = 0;
//	double theta2 = pi / 3;
//	double theta3 = 2 * pi / 3;//方向
//	cv::Mat u, v;
//	u.convertTo(u, CV_64FC1, 1);
//	v.convertTo(v, CV_64FC1, 1);
//	u = X*cos(theta1) - Y*sin(theta1);
//	v = X*sin(theta1) + Y*cos(theta1);
//	Mat temp1 = -((u.mul(u) + v.mul(v)) / (2 * (sigma*sigma)));
//	Mat temp2 = (1 / (2 * pi*(sigma*sigma)))*expmartix(temp1);//生成高斯核
//	Mat G0;
//	G0.convertTo(G0, CV_64FC1, 1);
//	G0 = temp2;
//	Mat G20;// = (((u.mul(u). / sigma)))
//	G20.convertTo(G20, CV_64FC1, 1);
//	Mat uu;
//	uu.convertTo(uu, CV_64FC1, 1);
//	uu = u.mul(u);
//	G20 = (uu / pow(sigma, 4) - (1 / (sigma*sigma))).mul(G0);//二阶导数
//															 //cout << G20;
//	u = X*cos(theta2) - Y*sin(theta2);
//	uu = u.mul(u);
//	Mat G260;
//	G260 = (uu / pow(sigma, 4) - (1 / (sigma*sigma))).mul(G0);
//	u = X*cos(theta3) - Y*sin(theta3);
//	uu = u.mul(u);
//	Mat G2120;
//	G2120 = (uu / pow(sigma, 4) - (1 / (sigma*sigma))).mul(G0);
//	G20 = G20 - mean(G20);
//	G260 = G260 - mean(G260);
//	G2120 = G2120 - mean(G2120);
//	cv::Mat J20, J260, J2120;
//	Conv2d_fft_first(imgI, G20, G260, G2120, J20, J260, J2120);
//	cv::Mat Nr, Dr;
//	Nr = ((2 * sqrt(3)) / 9)*((J260.mul(J260)) - (J2120.mul(J2120)) + (J20.mul(J260)) - (J20.mul(J2120)));//C3
//	Dr = (2.0 / 9.0) *((2 * (J20.mul(J20))) - (J260.mul(J260)) - (J2120.mul(J2120)) + (J20.mul(J260)) - (2 * (J260.mul(J2120))) + (J20.mul(J2120)));//C2																																				//2019.3.12--9:33: (2/9)按int类型计算为0，改为（2.0/9.0）
//	cv::Mat angles = cv::Mat::zeros(imgI.rows, imgI.cols, imgI.type());
//	for (int i = 0; i < imgI.rows; i++) {
//		for (int j = 0; j < imgI.cols; j++) {
//			if (Dr.ptr<double>(i)[j] > 0) {
//				angles.ptr<double>(i)[j] = atan(Nr.ptr<double>(i)[j] / (Dr.ptr<double>(i)[j]));
//			}
//			if (Nr.ptr<double>(i)[j] >= 0 && Dr.ptr<double>(i)[j] < 0) {
//				angles.ptr<double>(i)[j] = atan(Nr.ptr<double>(i)[j] / (Dr.ptr<double>(i)[j])) + pi;
//			}
//			if (Nr.ptr<double>(i)[j] < 0 && Dr.ptr<double>(i)[j] < 0) {
//				angles.ptr<double>(i)[j] = atan(Nr.ptr<double>(i)[j] / (Dr.ptr<double>(i)[j])) - pi;
//			}
//			if (Nr.ptr<double>(i)[j] > 0 && Dr.ptr<double>(i)[j] == 0) {
//				angles.ptr<double>(i)[j] = pi / 2;//正无穷
//			}
//			if (Nr.ptr<double>(i)[j] < 0 && Dr.ptr<double>(i)[j] == 0) {
//				angles.ptr<double>(i)[j] = -pi / 2;//负无穷
//			}
//			//cout << angles.ptr<double>(i)[j]<<",";
//		}
//	}
//	angles = 0.5*angles;//区间【-pi/2,pi/2】。
//	double sigmal = 5 * sigma;
//	siz = round(sigmal * 4) + 1;
//	if (siz % 2 == 0) {
//		siz = siz + 1;
//	}
//	meshgrid(cv::Range(-siz, siz), cv::Range(-siz, siz), X, Y);
//	theta1 = 0;
//	//cout << X;
//
//	u = X*cos(theta1) - Y*sin(theta1);
//	v = X*sin(theta1) + Y*cos(theta1);
//	u.convertTo(u, CV_64FC1, 1);
//	v.convertTo(v, CV_64FC1, 1);
//	//    cout << u;
//	double G0p1, G0p2;
//
//	//uuvv.convertTo(uuvv, CV_64FC1,1);
//	G0p1 = 1.0 / (2.0 * pi*(sigmal*sigmal));
//	G0p2 = 2.0 * (sigmal*sigmal);
//	cv::Mat uuvv = -(u.mul(u) + v.mul(v));
//	uuvv.convertTo(uuvv, CV_64FC1, 1);
//	uuvv = (uuvv / (2.0 * sigmal*sigmal));
//	cv::Mat G0_half;
//	G0_half.convertTo(G0_half, CV_64FC1, 1);
//	G0_half = expmartix(uuvv);
//	//cout << G0_half;
//	G0_half = G0p1*G0_half;//生成高斯核
//						   //cout << G0_half;
//						   //(1 / (2 * pi*(sigmal*sigmal)))*expmartix(-((u.mul(u) + v.mul(v)) / (2 * (sigmal*sigmal))));
//	cv::Mat G0u(G0_half.rows, G0_half.cols, G0_half.type());// = G0_half.mul(u);// u.mul(G0_half);
//	for (int i = 0; i < G0u.rows; i++) {
//		for (int j = 0; j < G0u.cols; j++) {
//			G0u.ptr<double>(i)[j] = G0_half.ptr<double>(i)[j] * u.ptr<double>(i)[j];
//		}
//	}
//	//G0u = G0_half.mul(u);
//	//        cout << G0u;
//	G0u.convertTo(G0u, CV_64FC1, 1);
//	G0u = (-1)*((1 / sigmal)*(1 / sigmal))*G0u;
//	//        cout << G0u;
//	//(-1)*((1 / sigmal)*(1 / sigmal))*u.mul(G0_half);
//	theta1 = pi / 2;
//	u = X*cos(theta1) - Y*sin(theta1);
//	u.convertTo(u, CV_64FC1, 1);
//	v.convertTo(v, CV_64FC1, 1);
//	cv::Mat G90u(G0_half.rows, G0_half.cols, G0_half.type());
//	for (int i = 0; i < G0u.rows; i++) {
//		for (int j = 0; j < G0u.cols; j++) {
//			G90u.ptr<double>(i)[j] = G0_half.ptr<double>(i)[j] * u.ptr<double>(i)[j];
//		}
//	}
//	G90u = (-1)*((1 / sigmal)*(1 / sigmal))*G90u;
//	//        cout << G90u;
//	//= (-1)*((1 / sigmal)*(1 / sigmal))*u.mul(G0_half);
//	G0u = G0u - mean(G0u);
//	G90u = G90u - mean(G90u);
//	cv::Mat J0u, J90u;
//	Conv2d_fft_second(imgI, G0u, G90u, J0u, J90u);
//	//point[10] = clock();//计时点
//	//cout << "二阶导卷积" << nrScales - scale + 1 << "次" << (double)(point[10] - point[8]) / CLOCKS_PER_SEC << "s\n";
//	//-------------
//	J0u.convertTo(J0u, CV_64FC1, 1);
//	J90u.convertTo(J90u, CV_64FC1, 1);
//	cv::Mat J2, J1;
//	cv::Mat J2t1, J2t2, J2t3;
//	J2t1 = (1.0 + (2.0 * cosmartix(2.0 * angles))).mul(J20);
//	J2t2 = (1.0 - cosmartix(2.0 * angles) + (sqrt(3)*sinmartix(2.0 * angles))).mul(J260);
//	J2t3 = (1.0 - cosmartix(2.0 * angles) - (sqrt(3)*sinmartix(2.0 * angles))).mul(J2120);
//	J2t1.convertTo(J2t1, CV_64FC1, 1);
//	J2t2.convertTo(J2t2, CV_64FC1, 1);
//	J2t3.convertTo(J2t3, CV_64FC1, 1);
//	//cout << J2t3;
//	J2 = (1.0 / 3.0)*(J2t1 + J2t2 + J2t3);
//	J1 = (J0u.mul(cosmartix(angles))) + (J90u.mul(sinmartix(angles)));
//	J1.convertTo(J1, CV_64FC1, 1);
//	J2.convertTo(J2, CV_64FC1, 1);
//	//cout << J2;//-------2019.3.13--9:55：J2大了两个数量级，需要找到其中的问题
//	cv::Mat psi_scale;
//
//	cv::Mat psi1, psi2, psi3, psi4;
//	psi1 = (sigma*sigma)*(absmartix(J2));
//	//cout << psi1;
//	psi1.convertTo(psi1, CV_64FC1, 1);
//	psi2 = 1.0 + absmartix(J1).mul(absmartix(J1));
//	//        cout << psi2;
//	psi2.convertTo(psi2, CV_64FC1, 1);
//	psi3 = (psi2.mul(psi2)).mul(psi2);
//	//    cout << psi3;
//	psi3.convertTo(psi3, CV_64FC1, 1);
//	psi4 = 1.0 / sqrtmartix(psi3);
//	psi4.convertTo(psi4, CV_64FC1, 1);
//	//    cout << psi4;
//	//point[11] = clock();//计时点9
//	//cout << "计算曲率用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[11] - point[10]) / CLOCKS_PER_SEC << "s\n";
//	/*if (preservePolarity == 0) {
//	psi_scale = psi1.mul(psi4);
//
//	}*/
//	//else {
//	psi_scale = (J2.mul(psi4));
//	psi_scale = (sigma*sigma)*psi_scale;
//	//}
//	double vMax;
//	double vMin;
//	cv::minMaxLoc(psi_scale, &vMin, &vMax, NULL, NULL);
//	psi_scale = (psi_scale - vMin) / vMax - vMin;
//	return psi_scale;
//}
cv::Mat calcLiveWireCostFun(const cv::Mat &img) {
    cv::Mat imgg = img;
    if(img.channels()==3) {
        cv::cvtColor(img,imgg,CV_BGR2GRAY);
    }
    cv::Mat imgG = calcImgGrad(imgg);
    cv::Mat imgE = calcCanny(imgg);
    cv::Mat ret;
    double pG = 0.8;
    double pE = 0.2;
    //    std::cout << imgG.type() << "/" << imgE.type() << std::endl;
    ret = pG*imgG + pE*imgE;
	//use_superpixels(ret);
    return ret;
}
double findMinId(vector<double> m) {
	double min = 10000;
	int minid = 0;
	for (int i = 0; i < m.size(); i++) {
		if (m[i] < min) {
			min = m[i];
			minid = i;
		}
	}
	return minid;
}
double findMax(vector<double> m) {
	double max = 0;
	//int maxid = 0;
	for (int i = 0; i < m.size(); i++) {
		if (m[i] >max) {
			max = m[i];
			//maxid = i;
		}
	}
	return max;
}
int minmin(int a, int b) {
	int c = a > b ? b : a;
	return c;
}
double averageap(vector<Ancpoint>& ap) {
	double sum = 0;
	for (int i = 0; i < ap.size();i++)
		sum += ap[i].width;
	sum /= ap.size();
	return sum;
}
double line_fit(vector<Ancpoint>& ap) {
	
	//输入拟合点
	if (ap.size() == 0) {
		return 0;
	}
	int i = 0;//位置指针，指向当前处理的点
	vector<Sllines> sllines;
	int sd = 20;//起始长度。使用20个点拟合直线
	while (i < ap.size() - 1) {
		std::vector<cv::Point> points;
		int st = i;
		int r = st + sd;
		if (ap.size() - 1 - st < sd)
			r = ap.size() - 1;
		for (int j = st; j < r; j++) {
			points.push_back(cv::Point(ap[j].anc.x, ap[j].anc.y));
			i++;
		}
		if (i >= ap.size() - 1)
			break;
		cv::Vec4f line_para;
		cv::fitLine(points, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
		//lineparas.push_back(line_para);
		//获取点斜式的点和斜率
		cv::Point point0;
		point0.x = line_para[2];
		point0.y = line_para[3];
		double k = line_para[1] / line_para[0];
		double dis = 0;
		int addloc = i;
		//直线的解析式(y = k(x - x0) + y0)
		do {
			double nx = ap[i].anc.x;
			double ny = ap[i].anc.y;
			dis = abs((k*nx - (ny)+point0.y - k*point0.x) / sqrt(k*k + (-1)*(-1)));//点到直线距离
			if (dis >= 3) {
				break;
			}
			else {
				points.push_back(cv::Point(nx, ny));
				i++;
				if (i >= ap.size() - 1)
					break;
			}
			if (i - addloc == 5) {//每新添加5个点重新拟合一次
				cv::fitLine(points, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
				point0.x = line_para[2];
				point0.y = line_para[3];
				k = line_para[1] / line_para[0];
				addloc = i;
			}
		} while (dis < 3);
		//cv::Vec4f line_para;
		cv::fitLine(points, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
		//lineparas.push_back(line_para);
		//获取点斜式的点和斜率
		//	cv::Point point0;
		point0.x = line_para[2];
		point0.y = line_para[3];
		k = line_para[1] / line_para[0];
		double b = point0.y - k*point0.x;
		double widsum = 0;
		for (int m = st; m < i; m++) {
			widsum += ap[m].width;
		}
		double widn = widsum / points.size();
		//widn = (widn + 0.5) / 2.0;
		Sllines a;
		a.wid = widn;
		a.k = k;
		a.b = b;
		sllines.push_back(a);
	}
	double widthmax = 0;
	for (int i = 0; i < sllines.size(); i++) {
		if (sllines[i].wid > widthmax)
			widthmax = sllines[i].wid;

	}
	return widthmax;
}

bool calpaths(const cv::Mat &imgF, const vector<int> & smcoors, vector<cv::Point>& path, vector<cv::Point>& crackpoints, int &searchdirection) {
	cv::Mat iPX, iPY;
	int smpoints[4] = { smcoors[0],smcoors[1] ,smcoors[2] ,smcoors[3] };
	int smp1x = smpoints[0];
	int smp1y = smpoints[1];
	int smp2x = smpoints[2];
	int smp2y = smpoints[3];
	cv::Point smp1 = cv::Point(smp1x, smp1y);
	cv::Point smp2 = cv::Point(smp2x, smp2y);
	int * sps = smpoints;
	//calcLiveWireP(imgF, p1.x, p1.y, iPX, iPY, 200);
	double dist = sqrt((smp2x - smp1x)*(smp2x - smp1x) + (smp2y - smp1y)*(smp2y - smp1y));
	int rad = dist + 5;//加一个增量5，确保能计算在范围内
	cv::imwrite("imgF.jpg", imgF);
	//cvWaitKey(0);
	calcLiveWireP(imgF, sps, iPX, iPY, rad);//在局部区域计算iPX，iPY
											/*cv::imshow("iPX", iPX);
											cv::imshow("iPY", iPY);
											cvWaitKey(0);*/
	cv::imwrite("iPX.jpg", normImage(iPX));
	cv::imwrite("iPY.jpg", normImage(iPY));
	int dex = 0;
	int dey = 0;
	calcLiveWireGetPath(iPX, iPY, smp2, dex, dey, path);//iPX、iPY都以种子点为基准
														/*cv::imshow("iPX", iPX);
														cv::imshow("iPY", iPY);
														cvWaitKey(0);*/
														/*dex = smp1x - pointscoors[0];
														dey = smp1y - pointscoors[1];*/
	for (int i = 0; i < path.size() / 5 * 5; i = i + (int)(path.size() / 5)) {
		crackpoints.push_back(path[i]);
	}
	//crackpoints.push_back(p2);
	iPX.release();
	iPY.release();
	return true;
}

int calcrackareabypath(const cv::Mat & connimg, cv::Mat& cimg, const vector<int> & smcoors, const vector<cv::Point>& path) {
	int ymin = min(smcoors[1], smcoors[3]);
	int ymax = max(smcoors[1], smcoors[3]);
	int xmin = min(smcoors[0], smcoors[2]);
	int xmax = max(smcoors[0], smcoors[2]);
	if (xmax - xmin > ymax - ymin) {
		ymin = 0;
		ymax = connimg.rows - 1;
	}
	else {
		xmin = 0;
		xmax = connimg.cols - 1;
	}
	cv::Mat areaimg;
	connimg(Range(ymin, ymax + 1), Range(xmin, xmax + 1)).copyTo(areaimg);
	cimg = cv::Mat::zeros(areaimg.rows, areaimg.cols, areaimg.type());
	int areapoints = 0;
	int col = areaimg.cols;
	int row = areaimg.rows;
	int x[8] = { 1,1,1,0,0,-1,-1,-1 };
	int y[8] = { 0,1,-1,1,-1,0,1,-1 };

	unsigned char *pcon = areaimg.data;
	unsigned char *pcr = cimg.data;
	//unsigned char * pBmp_binaried = bmp_binaried.data;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (pcon[i*col + j] <= 128)
				pcon[i*col + j] = 0;
			else
				pcon[i*col + j] = 255;
		}
	}
	stack<cv::Point> crstack;
	for (auto i : path) {
		crstack.push(i);
		areapoints++;
	}
	while (!crstack.empty()) {
		/*int siz = crstack.size();
		for (int i = 0; i < siz; i++) {*/
		cv::Point np = crstack.top();
		if (np.x - xmin >= 0 && np.x - xmin < col&&np.y - ymin >= 0 && np.y - ymin < row)
		{
			pcr[(np.y - ymin)*col + (np.x - xmin)] = 255;
			crstack.pop();
		}
		else {
			crstack.pop();
			continue;
		}
		for (int j = 0; j < 8; j++) {
			if (np.x + x[j] - xmin >= 0 && np.x + x[j] - xmin < col&&np.y + y[j] - ymin >= 0 && np.y + y[j] - ymin < row) {
				int sx = np.x + x[j];
				int sy = np.y + y[j];
				if (pcon[(sy - ymin)*col + sx - xmin] == 255 && pcr[(sy - ymin)*col + sx - xmin] == 0) {
					crstack.push(cv::Point(sx, sy));
					areapoints++;
				}
			}
		}
	}
	return areapoints;
}
//int calcrackareabypath_andorimg(const cv::Mat & smImg, cv::Mat& cimg, const vector<int> & smcoors, const vector<cv::Point>& path) {
//	//在原图中生长
//	/*CV_Assert(connimg.rows == crarea.rows);
//	CV_Assert(connimg.cols == crarea.cols);*/
//	int ymin = min(smcoors[1], smcoors[3]);
//	int ymax = max(smcoors[1], smcoors[3]);
//	int xmin = min(smcoors[0], smcoors[2]);
//	int xmax = max(smcoors[0], smcoors[2]);
//	if (xmax - xmin > ymax - ymin) {
//		ymin = 0;
//		ymax = smImg.rows - 1;
//	}
//	else {
//		xmin = 0;
//		xmax = smImg.cols - 1;
//	}
//	cv::Mat areaimg;
//	smImg(Range(ymin, ymax + 1), Range(xmin, xmax + 1)).copyTo(areaimg);
//	if (areaimg.channels()>1)
//		cvtColor(areaimg, areaimg, CV_BGR2GRAY);
//	cv::Mat crarea = cv::Mat::zeros(areaimg.rows, areaimg.cols, areaimg.type());
//	int areapoints = 0;
//	int col = areaimg.cols;
//	int row = areaimg.rows;
//	int x[8] = { 1,1,1,0,0,-1,-1,-1 };
//	int y[8] = { 0,1,-1,1,-1,0,1,-1 };
//
//	unsigned char *pcon = areaimg.data;
//	unsigned char *pcr = crarea.data;
//	//unsigned char * pBmp_binaried = bmp_binaried.data;
//	/*for (int i = 0; i < row; i++)
//	{
//	for (int j = 0; j < col; j++)
//	{
//	if (pcon[i*col + j] <= 128)
//	pcon[i*col + j] = 0;
//	else
//	pcon[i*col + j] = 255;
//	}
//	}*/
//	stack<cv::Point> crstack;
//	for (auto i : path) {
//		crstack.push(i);
//		areapoints++;
//	}
//	while (!crstack.empty()) {
//		/*int siz = crstack.size();
//		for (int i = 0; i < siz; i++) {*/
//		cv::Point np = crstack.top();
//		pcr[(np.y - ymin)*col + (np.x - xmin)] = 255;
//		crstack.pop();
//		for (int j = 0; j < 8; j++) {
//			if (np.x + x[j] - xmin >= 0 && np.x + x[j] - xmin < col&&np.y + y[j] - ymin >= 0 && np.y + y[j] - ymin < row) {
//				int sx = np.x + x[j];
//				int sy = np.y + y[j];
//				if (abs(pcon[(sy - ymin)*col + sx - xmin] - pcon[(np.y - ymin)*col + np.x - xmin]) <= 5 && pcr[(sy - ymin)*col + sx - xmin] == 0) {//相邻两个像素灰度差小于5
//					crstack.push(cv::Point(sx, sy));
//					areapoints++;
//				}
//			}
//		}
//	}
//	//}
//	/*cv::imshow("smimg", areaimg);
//	cv::imshow("area", crarea);
//	cvWaitKey(0);*/
//	crarea.copyTo(cimg);
//	crarea.release();
//	return areapoints;
//}
void cvHilditchThin1(const cv::Mat & src, cv::Mat &spur) {
	CV_Assert(src.type() == CV_8UC1);
	cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, src.type());
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}
	int i, j;
	int width, height;
	//之所以减2，是方便处理8邻域，防止越界
	width = src.cols - 2;
	height = src.rows - 2;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	int A1;
	cv::Mat tmpimg;
	while (1)
	{
		dst.copyTo(tmpimg);
		ifEnd = false;
		img = tmpimg.data + step;
		for (i = 2; i < height; i++)
		{
			img += step;
			for (j = 2; j<width; j++)
			{
				uchar* p = img + j;
				A1 = 0;
				if (p[0] > 0)
				{
					if (p[-step] == 0 && p[-step + 1]>0) //p2,p3 01模式
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1]>0) //p3,p4 01模式
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1]>0) //p4,p5 01模式
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step]>0) //p5,p6 01模式
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1]>0) //p6,p7 01模式
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1]>0) //p7,p8 01模式
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1]>0) //p8,p9 01模式
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step]>0) //p9,p2 01模式
					{
						A1++;
					}
					p2 = p[-step]>0 ? 1 : 0;
					p3 = p[-step + 1]>0 ? 1 : 0;
					p4 = p[1]>0 ? 1 : 0;
					p5 = p[step + 1]>0 ? 1 : 0;
					p6 = p[step]>0 ? 1 : 0;
					p7 = p[step - 1]>0 ? 1 : 0;
					p8 = p[-1]>0 ? 1 : 0;
					p9 = p[-step - 1]>0 ? 1 : 0;
					//计算AP2,AP4
					int A2, A4;
					A2 = 0;
					//if(p[-step]>0)
					{
						if (p[-2 * step] == 0 && p[-2 * step + 1]>0) A2++;
						if (p[-2 * step + 1] == 0 && p[-step + 1]>0) A2++;
						if (p[-step + 1] == 0 && p[1]>0) A2++;
						if (p[1] == 0 && p[0]>0) A2++;
						if (p[0] == 0 && p[-1]>0) A2++;
						if (p[-1] == 0 && p[-step - 1]>0) A2++;
						if (p[-step - 1] == 0 && p[-2 * step - 1]>0) A2++;
						if (p[-2 * step - 1] == 0 && p[-2 * step]>0) A2++;
					}
					A4 = 0;
					//if(p[1]>0)
					{
						if (p[-step + 1] == 0 && p[-step + 2]>0) A4++;
						if (p[-step + 2] == 0 && p[2]>0) A4++;
						if (p[2] == 0 && p[step + 2]>0) A4++;
						if (p[step + 2] == 0 && p[step + 1]>0) A4++;
						if (p[step + 1] == 0 && p[step]>0) A4++;
						if (p[step] == 0 && p[0]>0) A4++;
						if (p[0] == 0 && p[-step]>0) A4++;
						if (p[-step] == 0 && p[-step + 1]>0) A4++;
					}
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)>1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)<7 && A1 == 1)
					{
						if (((p2 == 0 || p4 == 0 || p8 == 0) || A2 != 1) && ((p2 == 0 || p4 == 0 || p6 == 0) || A4 != 1))
						{
							dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
							ifEnd = true;
						}
					}
				}
			}
		}
		//已经没有可以细化的像素了，则退出迭代
		if (!ifEnd) break;
	}
	//imwrite(dst1, dst);
	dst.copyTo(spur);
}
int callengthbyspur(const cv::Mat &spur) {
	int len = 0;
	int row = spur.rows;
	int col = spur.cols;
	unsigned char *ps = spur.data;
	//unsigned char *pcr = crarea.data;
	//unsigned char * pBmp_binaried = bmp_binaried.data;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (ps[i*col + j] >= 200)
				len++;
		}
	}
	return len;
}
void backtoorisize(const cv::Mat &smallerImg, cv::Mat & smImg, const vector<int> & smcoors) {
	int ymin = min(smcoors[1], smcoors[3]);
	int ymax = max(smcoors[1], smcoors[3]);
	int xmin = min(smcoors[0], smcoors[2]);
	int xmax = max(smcoors[0], smcoors[2]);
	if (xmax - xmin > ymax - ymin) {
		ymin = 0;
		ymax = smImg.rows - 1;
	}
	else {
		xmin = 0;
		xmax = smImg.cols - 1;
	}
	smallerImg.copyTo(smImg(Range(ymin, ymax + 1), Range(xmin, xmax + 1)));
}//from smaller to small
cv::Mat normImage(const cv::Mat &img) {
    cv::Mat ret;
    cv::normalize(img, ret, 0,255, CV_MINMAX, CV_8U);
    return ret;
}


long fFindMinG(SEntry *pSList, long lLength)
{
    long    lMinPos = 0;
    float   flMin   = 1e15;
    SEntry  SE;
    for (long lI = 0; lI < lLength; lI++) {//找最小值索引
        SE = *pSList++;
        if (SE.flG < flMin) {
            lMinPos = lI;
            flMin = SE.flG;
        }
    }
    return lMinPos;
}


long fFindLinInd(SEntry *pSList, long lLength, long lInd)
{
    SEntry SE;

    for (long lI = 0; lI < lLength; lI++) {
        SE = *pSList++;
        if (SE.lLinInd == lInd) return lI;
    }
    return -1; // If not found, return -1
}


void calcLiveWireP(const cv::Mat &imgS, int * smpoints, cv::Mat &iPX, cv::Mat &iPY, double dRadius, int LISTMAXLENGTH)
{
    iPX.release();
    iPY.release();
    cv::Size siz = imgS.size();
    double *pdF	= (double*)(imgS.data);
    short   sNX = siz.width;
    short   sNY = siz.height;
	int px1 = smpoints[0];
	int py1 = smpoints[1];
    short   sXSeed = px1;
    short   sYSeed = py1;
    iPX = cv::Mat::zeros(siz, CV_8S);
    iPY = cv::Mat::zeros(siz, CV_8S);
    char *plPX    = (char*) (iPX.data);
    char *plPY    = (char*) (iPY.data);

    // Start of the real functionality
    long    lInd;
    long    lLinInd;
    long    lListInd = 0; // = length of list
    short   sXLowerLim;
    short   sXUpperLim;
    short   sYLowerLim;
    short   sYUpperLim;
    long    lNPixelsToProcess;
    long    lNPixelsProcessed = 0;

    float   flThisG;
    float   flWeight;

    SEntry  SQ, SR;

    cv::Mat lE = cv::Mat::zeros(siz, CV_8S);
    char*   plE= (char*)(lE.data);
    SEntry *pSList = new SEntry[LISTMAXLENGTH];
    lNPixelsToProcess = ifMin(long(3.14*dRadius*dRadius + 0.5), long(sNX)*long(sNY));

    // Initialize active list with zero cost seed pixel.
    SQ.sX       = sXSeed;
    SQ.sY       = sYSeed;
    //    SQ.lLinInd  = ifLinInd(sXSeed, sYSeed, sNY);
    SQ.lLinInd  = ifLinInd(sXSeed, sYSeed, sNX);//返回一维索引值（从左到右，从上到下）
    SQ.flG      = 0.0;
    pSList[lListInd++] = SQ;

    // While there are still objects in the active list and pixel limit not reached
    while ((lListInd) && (lNPixelsProcessed < lNPixelsToProcess)) {
        // ----------------------------------------------------------------
        // Determine pixel q in list with minimal cost and remove from
        // active list. Mark q as processed.
        lInd = fFindMinG(pSList, lListInd);//索引，最小cost点
        SQ   = pSList[lInd];
        lListInd--;
        pSList[lInd] = pSList[lListInd];//最小费用点被最后一个点赋值
        plE[SQ.lLinInd]  = 1;
        // ----------------------------------------------------------------
        // Determine neighbourhood of q and loop over it 3*3
        sXLowerLim = ifMax(      0, SQ.sX - 1);
        sXUpperLim = ifMin(sNX - 1, SQ.sX + 1);
        sYLowerLim = ifMax(      0, SQ.sY - 1);
        sYUpperLim = ifMin(sNY - 1, SQ.sY + 1);
		//全图范围
        for (short sX = sXLowerLim; sX <= sXUpperLim; sX++) {
            for (short sY = sYLowerLim; sY <= sYUpperLim; sY++) {
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - -3*3邻域
                // Skip if pixel was already processed
                //                lLinInd = ifLinInd(sX, sY, sNY);
                lLinInd = ifLinInd(sX, sY, sNX);
                if (plE[lLinInd]) continue;//如果是SQ点就跳过
                // - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                // Compute the new accumulated cost to the neighbour pixel
                if ((abs(sX - SQ.sX) + abs(sY - SQ.sY)) == 1) //上下左右点
					flWeight = 0.71; 
				else //斜对角点
					flWeight = 1;
                flThisG = SQ.flG + float(pdF[lLinInd])*flWeight;//3*3邻域内某点cost=SQ点的cost+该点灰度值*该点权重
                //邻域内其他点的cost大于等于SQ点的cost
				// - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                // Check whether r is already in active list and if the
                // current cost is lower than the previous
                lInd = fFindLinInd(pSList, lListInd, lLinInd);//（点列表，列表长度，当前点索引），
				//寻找当前点是否已经在点列表中，如果在就返回在点列表中的位置，否则返回-1
                if (lInd >= 0) {//如果在列表中
                    SR = pSList[lInd];
                    if (flThisG < SR.flG) {
                        SR.flG = flThisG;//更新cost
                        pSList[lInd] = SR;
                        plPX[lLinInd] = char(SQ.sX - sX);
                        plPY[lLinInd] = char(SQ.sY - sY);
                    }
                } 
				else {
                    // - - - - - - - - - - - - - - - - - - - - - - - - - -
                    // If r is not in the active list, add it!
                    SR.sX = sX;
                    SR.sY = sY;
                    SR.lLinInd = lLinInd;
                    SR.flG = flThisG;
                    pSList[lListInd++] = SR;
                    plPX[lLinInd] = char(SQ.sX - sX);//当前点与SQ点x坐标的差距（带符号）
                    plPY[lLinInd] = char(SQ.sY - sY);
                    // - - - - - - - - - - - - - - - - - - - - - - - - - -
                }
            }
            // End of the neighbourhood loop.
            // ----------------------------------------------------------------
        }
        lNPixelsProcessed++;
    }
    // End of while loop
	/*iPX.release();
	iPY.release();*/
	lE.release();
    delete pSList;
}


void calcLiveWireGetPath(const cv::Mat &ipx, const cv::Mat &ipy, cv::Point pxy, int dex, int dey, std::vector<cv::Point> &path, int iMAXPATH)
{
    path.clear();
    // Initialize the variables
    int iXS = pxy.x;
    int iYS = pxy.y;
    std::vector<int> iX(iMAXPATH,0);
    std::vector<int> iY(iMAXPATH,0);
    int iLength = 0;
    iX[iLength] = iXS;
    iY[iLength] = iYS;
    while ( (ipx.at<char>(iYS, iXS) != 0) || (ipy.at<char>(iYS, iXS) != 0) ) // We're not at the seed
    {
        iXS = iXS + ipx.at<char>(iYS, iXS);
        iYS = iYS + ipy.at<char>(iYS, iXS);
        iLength = iLength + 1;
        iX[iLength] = iXS;
        iY[iLength] = iYS;
    }//从目标点开始回溯到种子点
    for(int ii=iLength-1; ii>=0; ii--) {//-2?while循环中
        int tx = iX[ii] + dex;
        int ty = iY[ii] + dey;
        path.push_back(cv::Point(tx,ty));
    }
}


cv::Point calcIdealAnchor(const cv::Mat &imgS, cv::Point pxy, int rad) {
    int nc = imgS.cols;
    int nr = imgS.rows;
    int r0 = pxy.y;
    int c0 = pxy.x;
    //
    int rMin = r0-rad;
    int rMax = r0+rad;
    if(rMin<0) {
        rMin = 0;
    }
    if(rMax>=nr) {
        rMax = nr-1;
    }
    //
    int cMin = c0-rad;
    int cMax = c0+rad;
    if(cMin<0) {
        cMin = 0;
    }
    if(cMax>=nc) {
        cMax = nc-1;
    }
    //
    cv::Point ret(c0,r0);
    double valMin = imgS.at<double>(r0,c0);
    double tval;
    for(int rr=rMin; rr<rMax; rr++) {
        for(int cc=cMin; cc<cMax; cc++) {
            tval = imgS.at<double>(rr,cc);
            if(tval<valMin) {
                valMin = tval;
                ret.x = cc;
                ret.y = rr;
            }
        }
    }
    return ret;
}

//------宽度计算函数

void CalculataWidth(cv::Mat &spur, const cv::Mat preimg, vector<Ancpoint>& ap) {
	//Mat spur = imread(src, 0);
	unsigned char *ps = spur.data;
	//imwrite(src2, spur);//测试，看spur是否有数据
	//imshow("123", spur);
	//waitKey(0);
	//unsigned char * a;
	//vector<Ancpoint> ap;
	//Ancpoint Temp;
	//vector<Point> anc;
	Ancpoint temp;
	int n = 0;
	int height = spur.rows;
	int width = spur.cols;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (ps[i*width + j] > 128) {
				ps[i*width + j] = 255;
			}
			else {
				ps[i*width + j] = 0;
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (ps[i*width + j] == 255) {
				//a.push_back(pb[i*height + j]);
				temp.anc.x = j;
				temp.anc.y = i;
				temp.width = 1;
				ap.push_back(temp);
			}
		}
	}
	//cv::Mat preimg = imread(src2);
	if (preimg.channels()>1)
		cvtColor(preimg, preimg, CV_RGB2GRAY);
	unsigned char * pp = preimg.data;
	int *radius = new int[ap.size()];//设置扩散半径
	for (int i = 0; i < ap.size(); i++) {
		radius[i] = 1;
	}
	int tempno;
	for (int i = 0; i < ap.size(); i++) {//遍历每个点
		tempno = (ap[i].anc.y)*width + ap[i].anc.x;
		int radiubefore = radius[i];
		if (ap[i].anc.y > 0 && ap[i].anc.y < height && ap[i].anc.x>0 && ap[i].anc.x < width) {
			do {
				radiubefore = radius[i];
				int range = 2 * radius[i] + 1;
				//unsigned char * inround = new unsigned char[range*range];
				//int inrno = 0;
				int lastinno = tempno - width*radius[i] - radius[i];//最后一个在圆内的像素下标；
				for (int j = 0; j < range*range; j++) {
					int te = tempno - width*radius[i] - radius[i] + j / range  * width + j % range;
					int tey = te / width;
					int tex; //= tey==0?te % tey:te;
					if (tey == 0) {
						tex = te;
					}
					else {
						tex = te%width;
					}
					float deltax = tex - ap[i].anc.x;
					float deltay = tey - ap[i].anc.y;
					float dis = sqrt(pow(deltax, 2) + pow(deltay, 2));
					if (radius[i] >= dis) {//矩形内接圆，圆内的像素一定在矩形内
						lastinno = te;
						if (pp[te]) {
							continue;
						}
						else if (!pp[te]) {
							radius[i]--;
							break;

						}
					}
				}
				if (pp[lastinno]) {
					radius[i]++;
					if (ap[i].anc.y > radius[i] - 1 && ap[i].anc.y < height - radius[i] + 1 && ap[i].anc.x>radius[i] - 1 && ap[i].anc.x < width - radius[i] + 1) {
						continue;
					}
					else {
						radius[i]--;
						break;
					}
				}
			} while (radius[i] > radiubefore);
		}
		ap[i].width = radius[i];
	}
	/*几个问题：1.如何按照半径扩散，采用先按矩形扩散，再按照内接圆判断的方法
	2.对于图像边界上的点进行了相应的防止越界的判断。
	3.对半径加减的控制使用了before来进行比较，半径比原来大就继续循环，半径比原来小或相等就停止迭代，认为半径就是现在的值。
	*/
}
void CalculataWidth_var(cv::Mat &spur, cv::Mat preimg, cv::Mat  orimg, vector<Ancpoint>& ap) {
	/*cv::Mat smImg;
	orimg.copyTo(smImg);*/

	unsigned char *ps = spur.data;
	Ancpoint temp;
	int n = 0;
	int height = spur.rows;
	int width = spur.cols;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (ps[i*width + j] > 128) {
				ps[i*width + j] = 255;
			}
			else {
				ps[i*width + j] = 0;
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (ps[i*width + j] == 255) {
				//a.push_back(pb[i*height + j]);
				temp.anc.x = j;
				temp.anc.y = i;
				temp.width = 1;
				ap.push_back(temp);
			}
		}
	}
	//cv::Mat preimg = imread(src2);
	if (preimg.channels()>1)
		cvtColor(preimg, preimg, CV_RGB2GRAY);
	if (orimg.channels()>1)
		cvtColor(orimg, orimg, CV_RGB2GRAY);
	unsigned char * pp = preimg.data;
	unsigned char * po = orimg.data;

	double *radius = new double[ap.size()];//设置扩散半径
	for (int i = 0; i < ap.size(); i++) {
		radius[i] = 1;//初始化为1
	}
	int tempno;
	for (int i = 0; i < ap.size(); i++) {//遍历每个点
		tempno = (ap[i].anc.y)*width + ap[i].anc.x;
		//int radiubefore = radius[i];
		//if (ap[i].anc.y > 0 && ap[i].anc.y < height && ap[i].anc.x>0 && ap[i].anc.x < width) {
		bool contin = true;
		int stradiu = 1;
		double wid = 0;
		double tv = 0;
		do {
			int mindex = tempno;
			//radiubefore = radius[i];
			int range = 2 * stradiu + 1;
			//unsigned char * inround = new unsigned char[range*range];
			//int inrno = 0;
			//int lastinno = tempno - width*radius[i] - radius[i];//最后一个在圆内的像素下标；
			//int sum = 0;//裂缝点周围像素值的累积和
			//int num = 0;//元素数量
			vector<double> vas;
			double sum = 0;
			for (int j = 0; j < range*range; j++) {
				int te = tempno - width*stradiu - stradiu + j / range  * width + j % range;
				if (te < 0 || te>width*height - 1)//防止越界
					continue;
				int tey = te / width;
				int tex; //= tey==0?te % tey:te;
				if (tey == 0) {
					tex = te;
				}
				else {
					tex = te%width;
				}
				float deltax = tex - ap[i].anc.x;
				float deltay = tey - ap[i].anc.y;
				float dis = sqrt(pow(deltax, 2) + pow(deltay, 2));
				if (stradiu >= dis) {//矩形内接圆，圆内的像素一定在矩形内
									 //lastinno = te;
					sum += (double)po[te];
					vas.push_back((double)po[te]);
					if (pp[te]<10) {
						contin = false;
					}
				}
			}
			/*for (int k = 0; k < vas.size(); k++) {
			sum += vas[k];
			}*/
			//double mean = sum / vas.size();
			double mean = po[mindex];
			double sqsum = 0;
			for (int k = 0; k < vas.size(); k++) {
				sqsum += (vas[k] - mean)*(vas[k] - mean);
			}
			double var = sqsum / vas.size();
			double vi = 1.0 / var;
			wid += vi*stradiu;
			tv += vi;
			if (contin == true)
				stradiu++;
		} while (contin);
		radius[i] = wid / tv;
		ap[i].width = radius[i] * 2 - 1;
		//}
	}
	delete[]radius;
}
void CalculataWidth_var_addgray(cv::Mat &spur, cv::Mat preimg, cv::Mat  orimg, vector<Ancpoint>& ap) {
	/*cv::Mat smImg;
	orimg.copyTo(smImg);*/

	unsigned char *ps = spur.data;
	Ancpoint temp;
	int n = 0;
	int height = spur.rows;
	int width = spur.cols;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (ps[i*width + j] > 128) {
				ps[i*width + j] = 255;
			}
			else {
				ps[i*width + j] = 0;
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (ps[i*width + j] == 255) {
				//a.push_back(pb[i*height + j]);
				temp.anc.x = j;
				temp.anc.y = i;
				temp.width = 1;
				ap.push_back(temp);
			}
		}
	}
	//cv::Mat preimg = imread(src2);
	if (preimg.channels()>1)
		cvtColor(preimg, preimg, CV_RGB2GRAY);
	if (orimg.channels()>1)
		cvtColor(orimg, orimg, CV_RGB2GRAY);
	unsigned char * pp = preimg.data;
	unsigned char * po = orimg.data;

	double *radius = new double[ap.size()];//设置扩散半径
	for (int i = 0; i < ap.size(); i++) {
		radius[i] = 1;//初始化为1
	}
	int tempno;
	for (int i = 0; i < ap.size(); i++) {//遍历每个点
		tempno = (ap[i].anc.y)*width + ap[i].anc.x;
		//int radiubefore = radius[i];
		//if (ap[i].anc.y > 0 && ap[i].anc.y < height && ap[i].anc.x>0 && ap[i].anc.x < width) {
		bool contin = true;
		int stradiu = 0;
		double adweiwid = 0;//进行方差加权计算的宽度
		double noadwid = 0;//方差小不参与加权的宽度
		double tv = 0;
		double dw = 0;
		int rg = 31;
		double outm = 0;//大外圈均值
		double times = 0;//大外圈参与计算的个数
		for (int j = 0; j < rg *rg; j++) {
			int te = tempno - width*stradiu - stradiu + j / rg  * width + j % rg;
			if (te < 0 || te>width*height - 1)//防止越界
				continue;
			outm = po[te];
			times++;
		}
		outm /= times;//把outm作为一个基准，换掉255
		do {
			int mindex = tempno;
			//radiubefore = radius[i];
			int range = 2 * stradiu + 1;
			vector<double> vas;
			double sum = 0;
			for (int j = 0; j < range*range; j++) {
				int te = tempno - width*stradiu - stradiu + j / range  * width + j % range;
				if (te < 0 || te>width*height - 1)//防止越界
					continue;
				int tey = te / width;
				int tex; //= tey==0?te % tey:te;
				if (tey == 0) {
					tex = te;
				}
				else {
					tex = te%width;
				}
				float deltax = tex - ap[i].anc.x;
				float deltay = tey - ap[i].anc.y;
				float dis = sqrt(pow(deltax, 2) + pow(deltay, 2));
				if (stradiu >= dis/*&&dis >= stradiu - 1*/) {//矩形内接圆，圆内的像素一定在矩形内
					if (stradiu == 0) {
						if (pp[te] > 200)
							dw = double(255 - po[te]) / 255.0;
					}
					else {
						sum += (double)po[te];
						vas.push_back((double)po[te]);
						if (pp[te]<10) {
							contin = false;
						}
					}
				}
			}
			/*for (int k = 0; k < vas.size(); k++) {
			sum += vas[k];
			}*/
			if (stradiu > 0)
			{
				double mean = sum / vas.size();
				//double mean = po[mindex];
				double sqsum = 0;
				for (int k = 0; k < vas.size(); k++) {
					sqsum += (vas[k] - mean)*(vas[k] - mean);
				}
				double var = sqsum / vas.size();
				//如果方差很小不参与加权
				if (var <= 50) {
					noadwid += ((255 - mean) / 255);
				}
				else if (var > 2000) {
					double vi = 1.0 / var;
					adweiwid += vi*stradiu;
					tv += vi;
					contin = false;
				}
				else {
					double vi = 1.0 / var;
					adweiwid += vi*stradiu;
					tv += vi;
				}
			}
			if (contin == true)
				stradiu++;
		} while (contin);
		radius[i] = tv>0 ? noadwid + (adweiwid / tv) : noadwid;
		ap[i].width = radius[i] * 2 + dw;
		//}
	}
	delete[]radius;
}
//
void CalculataWidth_gray(cv::Mat &spur, cv::Mat preimg, cv::Mat orimg, vector<Ancpoint>& ap) {
	//5.30 根据灰度值的不同，裂缝的宽窄有所区别
	//起始半径设置为从0开始增长
	unsigned char *ps = spur.data;
	Ancpoint temp;
	int n = 0;
	int height = spur.rows;
	int width = spur.cols;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (ps[i*width + j] > 128) {
				ps[i*width + j] = 255;
			}
			else {
				ps[i*width + j] = 0;
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (ps[i*width + j] == 255) {
				temp.anc.x = j;
				temp.anc.y = i;
				temp.width = 1;
				ap.push_back(temp);
			}
		}
	}

	if (preimg.channels()>1)
		cvtColor(preimg, preimg, CV_RGB2GRAY);
	if (orimg.channels()>1)
		cvtColor(orimg, orimg, CV_RGB2GRAY);
	unsigned char * pp = preimg.data;
	unsigned char * po = orimg.data;

	//double *radius = new double[ap.size()];//设置扩散半径
	//for (int i = 0; i < ap.size(); i++) {
	//	radius[i] = 0;//初始化为0
	//}
	int tempno;
	for (int i = 0; i < ap.size(); i++) {//遍历每个点
		tempno = (ap[i].anc.y)*width + ap[i].anc.x;
		bool contin = true;
		int stradiu = 0;
		double dw = 0;//基准宽度，裂缝中心线上原图的灰度值，越黑裂缝越宽，宽度与（255-gray）成正比
		double sumwid = 0;
		do {
			int range = 2 * stradiu + 1;
			int num = 0;
			double sum = 0;
			for (int j = 0; j < range*range; j++) {
				int te = tempno - width*stradiu - stradiu + j / range  * width + j % range;
				if (te < 0 || te>width*height - 1)//防止越界
					continue;
				int tey = te / width;
				int tex;
				if (tey == 0) {
					tex = te;
				}
				else {
					tex = te%width;
				}
				float deltax = tex - ap[i].anc.x;
				float deltay = tey - ap[i].anc.y;
				float dis = sqrt(pow(deltax, 2) + pow(deltay, 2));
				if (stradiu >= dis&&dis >= stradiu - 1) {//矩形内接圆，圆内的像素一定在矩形内
					if (stradiu == 0) {
						if (pp[te] > 200)
							dw = double(255 - po[te]) / 255.0;
					}
					else {
						if (po[te] < 150) {
							sum += (double)(255 - po[te]) / 255.0;

						}
						num++;
						if (pp[te] < 10 || po[te] >= 150) {
							contin = false;
						}
					}
				}
			}
			if (stradiu>0 && num != 0 && sum != 0)
				sumwid += sum / double(num);
			if (contin == true)
				stradiu++;
		} while (contin);
		//radius[i] = wid / tv;
		ap[i].width = dw + 2 * sumwid;
		//}
	}
	//delete[]radius;
}
/*
void CalculataWidth_Bycur(cv::Mat &spur, const cv::Mat preimg, const cv::Mat curimg, vector<Ancpoint>& ap) {
//利用曲率真值计算宽度，将裂缝待计算的中心点位置的曲率值作为基准
unsigned char *ps = spur.data;
Ancpoint temp;
int n = 0;
int height = spur.rows;
int width = spur.cols;
for (int i = 0; i < height; i++) {
for (int j = 0; j < width; j++) {
if (ps[i*width + j] > 128) {
ps[i*width + j] = 255;
}
else {
ps[i*width + j] = 0;
}
}
}
for (int i = 0; i < height; i++) {
for (int j = 0; j < width; j++) {
if (ps[i*width + j] == 255) {
//a.push_back(pb[i*height + j]);
temp.anc.x = j;
temp.anc.y = i;
temp.width = 1;
ap.push_back(temp);
}
}
}
//cv::Mat preimg = imread(src2);
if (preimg.channels()>1)
cvtColor(preimg, preimg, CV_RGB2GRAY);
unsigned char * pp = preimg.data;
unsigned char * pc = curimg.data;

double *radius = new double[ap.size()];//设置扩散半径
for (int i = 0; i < ap.size(); i++) {
radius[i] = 1;//初始化为1
}
int tempno;
for (int i = 0; i < ap.size(); i++) {//遍历每个点
tempno = (ap[i].anc.y)*width + ap[i].anc.x;
double stancurval = pc[tempno];
//int radiubefore = radius[i];
//if (ap[i].anc.y > 0 && ap[i].anc.y < height && ap[i].anc.x>0 && ap[i].anc.x < width) {
bool contin = true;
int stradiu = 1;
double wid = 0;
double tv = 0;
cv::Mat logpos=cv::Mat::zeros(height, width, CV_8UC1);//被访问过的标志位
unsigned char * pl = logpos.data;
do {
int range = 2 * stradiu + 1;
vector<double> vas;
double sum = 0;
int round = 0;
for (int j = 0; j < range*range; j++) {
int te = tempno - width*stradiu - stradiu + j / range  * width + j % range;
if (te < 0 || te>width*height - 1)//防止越界
continue;
int tey = te / width;
int tex; //= tey==0?te % tey:te;
if (tey == 0) {
tex = te;
}
else {
tex = te%width;
}
float deltax = tex - ap[i].anc.x;
float deltay = tey - ap[i].anc.y;
float dis = sqrt(pow(deltax, 2) + pow(deltay, 2));
if (stradiu >= dis&&pl[te]==0) {//矩形内接圆，圆内的像素一定在矩形内
//lastinno = te;
sum += (double)pc[te];
vas.push_back((double)pc[te]);
if (pp[te]<10) {
contin = false;
}
}
}
//for (int k = 0; k < vas.size(); k++) {
//sum += vas[k];
//}
double mean = sum / vas.size();
double sqsum = 0;
for (int k = 0; k < vas.size(); k++) {
sqsum += (vas[k] - mean)*(vas[k] - mean);
}
double var = sqsum / vas.size();
double vi = 1.0 / var;
wid += vi*stradiu;
tv += vi;
if (contin == true)
stradiu++;
} while (contin);
radius[i] = wid / tv;
ap[i].width = radius[i];
//}
}
delete[]radius;
}
*/
//---------------