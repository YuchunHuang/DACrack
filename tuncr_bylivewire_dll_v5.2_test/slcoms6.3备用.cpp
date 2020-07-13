#include "stdafx.h"
#include"strus.h"
//#include <opencv2/core/utility.hpp>
//
//#include <opencv2/tracking.hpp>
//
//#include <opencv2/videoio.hpp>


#include <cstring>
using namespace std;
using namespace cv;
typedef unsigned int UINT;

vector<double> kseedsl(0);
vector<double> kseedsa(0);
vector<double> kseedsb(0);
vector<double> kseedsx(0);
vector<double> kseedsy(0);
vector<int> kseedsnums(0);
int superpixernum = 2000;
const int superpixel_size = 800;
float scale = 1.0;
Point2f cen(0, 0);
Point2f fp(0, 0);
Point2f sp(0, 0);
int t = 0;
vector<int> border(4);
vector<int >zoomsps(0);
Mat srcImg;
Mat dstImg;
//string orgimgpath;
string windowName;
bool moving = false;
bool Isdetect = true;
const double pi = 3.14159265358979323846;
bool verticalmeanfilter(const cv::Mat& src,cv::Mat &dst ,double sens) {
	/*clock_t start, end;
	start = clock();*/
	//setUseOpenCL(true);
	cv::Mat img;
	src.copyTo(img); // = ; imread(src);
	UMat imgcopy, img2;
	cv::Mat imgFilter;
	img.copyTo(imgcopy);
	cv::bilateralFilter(imgcopy, img2, 0, sens, 0.5*sens);//
												  /*double sigmaColor : 颜色空间过滤器的sigma值，这个参数的值越大，表明该像素邻域内有越宽广的颜色会被混合到一起，产生较大的半相等颜色区域。
														  double sigmaSpace : 坐标空间中滤波器的sigma值，如果该值较大，则意味着颜色相近的较远的像素将相互影响，从而使更大的区域中足够相似的颜色获取相同的颜色。当d>0时，d指定了邻域大小且与sigmaSpace无关，否则d正比于sigmaSpace.
														  */
	if(img2.channels()==3)
		cv::cvtColor(img2, img2, CV_RGB2GRAY);
	if (img.channels() == 3)
		cv::cvtColor(img, img, CV_RGB2GRAY);
	//cv::Mat imgFilter;
	img2.copyTo(imgFilter);
	
	cv::Mat imgDiff(imgFilter.rows, imgFilter.cols, imgFilter.type());

	for (int j = 0; j < img.cols; ++j)
	{
		int aveCol = 0;
		for (int i = 0; i < img.rows; ++i)
		{
			aveCol += imgFilter.at<uchar>(i, j);
		}
		aveCol = aveCol / img.rows;

		for (int i = 0; i < img.rows; ++i)
		{
			int diff = aveCol - imgFilter.at<uchar>(i, j);
			imgDiff.at<uchar>(i, j) = diff > 0 ? diff : 0;
		}
	}
	//imwrite(dst, imgDiff);
	//end = clock();
	imgDiff.copyTo(dst);
	imgDiff.release();
	img.release();
	img2.release();
	imgFilter.release();
	
	return true;
}
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y) {
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);
	cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
	cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}
cv::Mat padarray(const cv::Mat &Image, int RowPad, int ColPad) {
	//以前没有对colPad>m的解决方法
	//2019.3.1 对方法进行改进
	int n = Image.rows;
	int m = Image.cols;
	int len = ColPad / m;
	Mat temp1 = Mat::zeros(n, m + ColPad * 2, Image.type());
	Mat temp2 = Mat::zeros(n + RowPad * 2, m + ColPad * 2, Image.type());
	//int line = 0;
	//while (line <= len) {
	for (int i = 0; i < ColPad; i++)
	{
		int line = 0;
		if (i / m == 0) {
			Image.col(i).copyTo(temp1.col(ColPad - 1 - i));
			Image.col(m - 1 - i).copyTo(temp1.col(m + ColPad + i));
		}
		if (i / m > 0) {
			line = i / m;
			temp1.col(ColPad - (line*m) + (i%m)).copyTo(temp1.col(ColPad - (line*m) - (i%m) - 1));
			temp1.col(ColPad + (m - 1) + (line*m) - (i%m)).copyTo(temp1.col(ColPad + m + (line*m) + (i%m)));
		}
		/*if (i >= m) {
		temp1.col(ColPad - 1 - m).copyto
		}*/
	}
	//}
	Image.copyTo(temp1.colRange(ColPad, m + ColPad));
	for (int j = 0; j < RowPad; j++)
	{
		int line = 0;
		if (j / n == 0) {
			temp1.row(j).copyTo(temp2.row(RowPad - 1 - j));
			temp1.row(n - 1 - j).copyTo(temp2.row(n + RowPad + j));
		}
		if (j / n > 0) {
			line = j / n;
			temp2.row(RowPad - (line*n) + (j%n)).copyTo(temp2.row(RowPad - (line * n) - (j % n) - 1));
			temp2.row(RowPad + (n - 1) + (line*n) - (j%n)).copyTo(temp2.row(RowPad + n + (line * n) + (j % n)));
		}
	}
	temp1.copyTo(temp2.rowRange(RowPad, n + RowPad));
	temp1.release();
	return temp2;
}
cv::Mat addzero(const cv::Mat &input, int addrows, int addcols) {
	cv::Mat output = cv::Mat::zeros(addrows, addcols, input.type());
	//cv::Mat roiA(output, RECT(0, 0, input.rows, input.cols));

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			output.ptr<double>(i)[j] = input.ptr<double>(i)[j];
		}
	}
	return output;
}
cv::Mat expmartix(const cv::Mat &input) {
	cv::Mat output(input.rows, input.cols, input.type());
	output.convertTo(output, CV_64FC1, 1);
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			output.ptr<double>(i)[j] = exp(input.ptr<double>(i)[j]);
			//output.at<double>(i, j) = exp(input.at<double>(i, j));
		}
	}
	return output;
}
cv::Mat cosmartix(const cv::Mat &input) {
	cv::Mat output(input.rows, input.cols, input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			output.ptr<double>(i)[j] = cos(input.ptr<double>(i)[j]);
		}
	}
	return output;
}
cv::Mat sinmartix(const cv::Mat &input) {
	cv::Mat output(input.rows, input.cols, input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			output.ptr<double>(i)[j] = sin(input.ptr<double>(i)[j]);
		}
	}
	return output;
}
cv::Mat absmartix(const cv::Mat &input) {
	cv::Mat output(input.rows, input.cols, input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			output.ptr<double>(i)[j] = abs(input.ptr<double>(i)[j]);
		}
	}
	return output;
}
cv::Mat sqrtmartix(const cv::Mat &input) {
	cv::Mat output(input.rows, input.cols, input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			output.ptr<double>(i)[j] = sqrt(input.ptr<double>(i)[j]);
		}
	}
	return output;
}
cv::Mat sign(const cv::Mat &input) {
	cv::Mat output(input.rows, input.cols, input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.ptr<double>(i)[j] > 0) {
				output.ptr<double>(i)[j] = 1;
			}
			else {
				output.ptr<double>(i)[j] = -1;
			}
		}
	}
	return output;
}
cv::Mat bigger(const cv::Mat &input, const cv::Mat &input2) {
	cv::Mat output(input.rows, input.cols, input.type());
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.ptr<double>(i)[j] >input2.ptr<double>(i)[j]) {
				output.ptr<double>(i)[j] = 1;
			}
			else {
				output.ptr<double>(i)[j] = 0;
			}
		}
	}
	return output;
}
cv::Mat repl(const cv::Mat &old, const cv::Mat &newi,const  cv::Mat & model) {
	cv::Mat output(old.rows, old.cols, old.type());
	output.convertTo(output, CV_64FC1, 1);
	for (int i = 0; i < old.rows; i++) {
		for (int j = 0; j < old.cols; j++) {
			if (model.ptr<double>(i)[j] == 1) {
				output.ptr<double>(i)[j] = newi.ptr<double>(i)[j];
			}
			else {
				output.ptr<double>(i)[j] = old.ptr<double>(i)[j];
			}
		}
	}
	return output;
}
cv::Mat orepl(const cv::Mat &old, double newo, const cv::Mat &model) {
	cv::Mat output(old.rows, old.cols, old.type());
	output.convertTo(output, CV_64FC1, 1);
	for (int i = 0; i < old.rows; i++) {
		for (int j = 0; j < old.cols; j++) {
			if (model.ptr<double>(i)[j] == 1) {
				output.ptr<double>(i)[j] = newo;
			}
			else {
				output.ptr<double>(i)[j] = old.ptr<double>(i)[j];
			}
		}
	}
	return output;
}
cv::Mat mat2gray(const cv::Mat &input) {
	cv::Mat output(input.rows, input.cols, input.type());
	double max = 0, min = 0;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.ptr<double>(i)[j] > max) {
				max = input.ptr<double>(i)[j];
			}
			if (input.ptr<double>(i)[j] < min) {
				min = input.ptr<double>(i)[j];
			}
		}
	}
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			output.ptr<double>(i)[j] = (input.ptr<double>(i)[j] - min) / (max - min);
		}
	}
	return output;
}
cv::Mat im2int(const cv::Mat &input) {
	cv::Mat output(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			output.at<uchar>(i, j) = round(input.ptr<double>(i)[j]);
		}
	}
	return output;
}
double summar(const cv::Mat &input) {
	double sum = 0;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			sum = sum + input.ptr<double>(i)[j];
		}
	}
	return sum;
}
void selectpart(const cv::Mat &input,cv::Mat &output, int up, int down, int left, int right) {
	cv::Mat o = cv::Mat::zeros(down - up + 1, right - left + 1, input.type());
	for (int i = up; i < down; i++) {
		for (int j = left; j < right; j++) {
			o.ptr<double>(i - up)[j - left] = input.ptr<double>(i)[j];
		}
	}
	o.copyTo(output);
	o.release();
	//return output;
}
void Conv2d_fft_first(const cv::Mat & image,const  cv::Mat & h,const  cv::Mat  & h2,const cv::Mat & h3, cv::Mat &output, cv::Mat &output2, cv::Mat &output3) {
	//不考虑h做为参数进入到实际扩大空间的计算过程中
	/*clock_t start, scut[3];
	start = clock();*/
	cv::Mat ch, ch2, ch3;
	h.copyTo(ch);
	h2.copyTo(ch2);
	h3.copyTo(ch3);
	double size_h_r = h.size().height;
	double size_h_c = h.size().width;
	double size_image_r = image.size().height;
	double size_image_c = image.size().width;
	double padsize_r = (size_h_r - 1) / 2;
	double padsize_c = (size_h_c - 1) / 2;
	Mat img = padarray(image, (int)padsize_r, (int)padsize_c);
	int size_img_r = img.rows;
	int size_img_c = img.cols;
	cv::Mat o, o2, o3;
	o = cv::Mat::zeros((img.rows - h.rows) + 1, (img.cols - h.cols) + 1, img.type());
	o2 = cv::Mat::zeros((img.rows - h.rows) + 1, (img.cols - h.cols) + 1, img.type());
	o3 = cv::Mat::zeros((img.rows - h.rows) + 1, (img.cols - h.cols) + 1, img.type());
	//int wi = getOptimalDFTSize(size_h_c + size_img_c - 1);
	//int hi = getOptimalDFTSize(size_h_r + size_img_r - 1);
	img = addzero(img, size_h_r + size_img_r - 1, size_h_c + size_img_c - 1);
	ch = addzero(h, size_h_r + size_img_r - 1, size_h_c + size_img_c - 1);
	ch2 = addzero(h2, size_h_r + size_img_r - 1, size_h_c + size_img_c - 1);
	ch3 = addzero(h3, size_h_r + size_img_r - 1, size_h_c + size_img_c - 1);
	//o = conv2(img, h);//考虑到空间域卷积速度不符合实际要求，改成利用傅里叶变换实现卷积
	/*scut[0] = clock();
	cout << "    图像与卷积核大小处理用时" << (double)(scut[0] - start) / CLOCKS_PER_SEC << "s\n";
	*/
	dft(img, img, 0, size_img_r);//只做一次dft，对原图
	dft(h, h, 0, size_h_r);
	dft(h2, h2, 0, size_h_r);
	dft(h3, h3, 0, size_h_r);
	mulSpectrums(img, h, o, DFT_COMPLEX_OUTPUT);
	mulSpectrums(img, h2, o2, DFT_COMPLEX_OUTPUT);
	mulSpectrums(img, h3, o3, DFT_COMPLEX_OUTPUT);
	dft(o, o, DFT_INVERSE + DFT_SCALE, o.rows);
	dft(o2, o2, DFT_INVERSE + DFT_SCALE, o2.rows);
	dft(o3, o3, DFT_INVERSE + DFT_SCALE, o3.rows);
	/*scut[1] = clock();
	cout << "    fft卷积用时" << (double)(scut[1] - scut[0]) / CLOCKS_PER_SEC << "s\n";*/
	selectpart(o, o,floor(size_h_r / 2), (size_img_r + size_h_r - 1) - floor(size_h_r / 2)-1,
		floor(size_h_c / 2), (size_img_c + size_h_c - 1) - floor(size_h_c / 2)-1);
	selectpart(o2, o2, floor(size_h_r / 2), (size_img_r + size_h_r - 1) - floor(size_h_r / 2)-1,
		floor(size_h_c / 2), (size_img_c + size_h_c - 1) - floor(size_h_c / 2)-1);
	selectpart(o3, o3, floor(size_h_r / 2), (size_img_r + size_h_r - 1) - floor(size_h_r / 2)-1,
		floor(size_h_c / 2), (size_img_c + size_h_c - 1) - floor(size_h_c / 2)-1);
	selectpart(o, o, floor(size_h_r / 2), (size_image_r + size_h_r - 1) - floor(size_h_r / 2)-1,
		floor(size_h_c / 2), (size_image_c + size_h_c - 1) - floor(size_h_c / 2)-1);
	selectpart(o2, o2,  floor(size_h_r / 2), (size_image_r + size_h_r - 1) - floor(size_h_r / 2)-1,
		floor(size_h_c / 2), (size_image_c + size_h_c - 1) - floor(size_h_c / 2)-1);
	selectpart(o3, o3,  floor(size_h_r / 2), (size_image_r + size_h_r - 1) - floor(size_h_r / 2)-1,
		floor(size_h_c / 2), (size_image_c + size_h_c - 1) - floor(size_h_c / 2)-1);
	o.copyTo(output);
	o2.copyTo(output2);
	o3.copyTo(output3);
	o.release();
	o2.release();
	o3.release();

	/*scut[2] = clock();
	cout << "    输出到标准大小用时" << (double)(scut[2] - scut[1]) / CLOCKS_PER_SEC << "s\n";*/
	//output(Rect(0, 0, output.cols, output.rows)).copyTo(output);
}
void Conv2d_fft_second(const cv::Mat &  image, const cv::Mat & h,const  cv::Mat & h2, cv::Mat &output, cv::Mat &output2) {
	/*clock_t start, scut[3];
	start = clock();*/
	//h做参数进行计算不合适
	cv::Mat dh, dh2;
	h.copyTo(dh);
	h2.copyTo(dh2);
	double size_h_r = h.size().height;
	double size_h_c = h.size().width;
	double size_image_r = image.size().height;
	double size_image_c = image.size().width;
	double padsize_r = (size_h_r - 1) / 2;
	double padsize_c = (size_h_c - 1) / 2;
	Mat img = padarray(image, (int)padsize_r, (int)padsize_c);
	int size_img_r = img.rows;
	int size_img_c = img.cols;
	cv::Mat o, o2;
	o = cv::Mat::zeros((img.rows - h.rows) + 1, (img.cols - h.cols) + 1, img.type());
	o2 = cv::Mat::zeros((img.rows - h.rows) + 1, (img.cols - h.cols) + 1, img.type());
	//o3 = cv::Mat::zeros((img.rows - h.rows) + 1, (img.cols - h.cols) + 1, img.type());
	//int wi = getOptimalDFTSize(/*size_h_c +*/ size_img_c);
	// int hi = getOptimalDFTSize(/*size_h_r +*/ size_img_r);
	img = addzero(img, size_h_r + size_img_r - 1, size_h_c + size_img_c - 1);
	dh = addzero(dh, size_h_r + size_img_r - 1, size_h_c + size_img_c - 1);
	dh2 = addzero(dh2, size_h_r + size_img_r - 1, size_h_c + size_img_c - 1);
	//h3 = addzero(h3, size_h_r + size_img_r - 1, size_h_c + size_img_c - 1);
	//o = conv2(img, h);//考虑到空间域卷积速度不符合实际要求，改成利用傅里叶变换实现卷积
	/*scut[0] = clock();
	cout << "    图像与卷积核大小处理用时" << (double)(scut[0] - start) / CLOCKS_PER_SEC << "s\n";
	*/
	dft(img, img, 0, size_img_r);//只做一次dft，对原图
	dft(dh,dh, 0, size_h_r);
	dft(dh2, dh2, 0, size_h_r);
	//dft(h3, h3, 0, size_h_r);
	mulSpectrums(img, dh, o, DFT_COMPLEX_OUTPUT);
	mulSpectrums(img, dh2, o2, DFT_COMPLEX_OUTPUT);
	//mulSpectrums(img, h3, o3, DFT_COMPLEX_OUTPUT);
	dft(o, o, DFT_INVERSE + DFT_SCALE, o.rows);
	dft(o2, o2, DFT_INVERSE + DFT_SCALE, o2.rows);
	//dft(o3, o3, DFT_INVERSE + DFT_SCALE, o3.rows);
	/*scut[1] = clock();
	cout << "    fft卷积用时" << (double)(scut[1] - scut[0]) / CLOCKS_PER_SEC << "s\n";*/
	selectpart(o,o, floor(size_h_r / 2), (size_img_r + size_h_r - 1) - floor(size_h_r / 2)-1,
		floor(size_h_c / 2), (size_img_c + size_h_c - 1) - floor(size_h_c / 2)-1);
	selectpart(o2,o2, floor(size_h_r / 2), (size_img_r + size_h_r - 1) - floor(size_h_r / 2) - 1,
		floor(size_h_c / 2), (size_img_c + size_h_c - 1) - floor(size_h_c / 2) - 1);
	//output3 = selectpart(output3, 1 + floor(size_h_r / 2), (size_img_r + size_h_r - 1) - floor(size_h_r / 2),
	//    1 + floor(size_h_c / 2), (size_img_c + size_h_c - 1) - floor(size_h_c / 2));
	selectpart(o, o, floor(size_h_r / 2), (size_image_r + size_h_r - 1) - floor(size_h_r / 2)-1,
		 floor(size_h_c / 2), (size_image_c + size_h_c - 1) - floor(size_h_c / 2)-1);
	selectpart(o2, o2, floor(size_h_r / 2), (size_image_r + size_h_r - 1) - floor(size_h_r / 2) - 1,
		floor(size_h_c / 2), (size_image_c + size_h_c - 1) - floor(size_h_c / 2) - 1);
	o.copyTo(output);
	o2.copyTo(output2);
	o.release();
	o2.release();
	dh.release();
	dh2.release();
	//output3 = selectpart(output3, 1 + floor(size_h_r / 2), (size_image_r + size_h_r - 1) - floor(size_h_r / 2),
	//    1 + floor(size_h_c / 2), (size_image_c + size_h_c - 1) - floor(size_h_c / 2));
	/*scut[2] = clock();
	cout << "    输出到标准大小用时" << (double)(scut[2] - scut[1]) / CLOCKS_PER_SEC << "s\n";*/
	//output(Rect(0, 0, output.cols, output.rows)).copyTo(output);
}
char * CurvatureIndex2D(const cv::Mat &  I1, cv::Mat & dst, int nrScales, int  preservePolarity, double min_scale) {
	//point1 = clock();
	//cout << "共计用时: " << (double)(end - start) / CLOCKS_PER_SEC << "s\n";
	//clock_t start, point[15];
	//start = clock();
	//cv::Mat imgI1;
	//imgI1 = I1;//copyto
	//imgI1.convertTo(imgI1, CV_64FC1, 1);
	
	int siz = 0;
	cv::Mat imgI;
	I1.copyTo(imgI);// = imgI1;
	imgI.convertTo(imgI, CV_64FC1, 1);
	imgI = -imgI;//裂缝取负数
	double t = min_scale;
	int R, C;
	R = imgI.size().height;
	C = imgI.size().width;
	double sigma;
	cv::Mat Enhanceimg;
	cv::Mat psi, orient, scaleMap, scaleMapIdx, idx, polarity, polarity_scale;//重复使用且用到上一次迭代结果的变量在循环外赋值
	//point[0] = clock();//计时点1
					   //cout << "cindex2d循环前用时" << (double)(point[0] - start) / CLOCKS_PER_SEC << "s\n";
	for (int scale = nrScales; scale >1; scale--) {//开始N个尺度的迭代
		//point[1] = clock();//计时点2
		sigma = t*(pow(sqrt(2), scale - 1));
		siz = round(sigma * 4) + 1;
		//siz = round(sigma * 2) + 1;
		if (siz % 2 == 0) {
			siz = siz + 1;
		}

		cv::Mat X, Y;
		meshgrid(cv::Range(-siz, siz), cv::Range(-siz, siz), X, Y);
		//cout << X;
		X.convertTo(X, CV_64FC1, 1);
		Y.convertTo(Y, CV_64FC1, 1);
		double theta1 = 0;
		double theta2 = pi / 3;
		double theta3 = 2 * pi / 3;//方向
		cv::Mat u, v;
		u.convertTo(u, CV_64FC1, 1);
		v.convertTo(v, CV_64FC1, 1);
		u = X*cos(theta1) - Y*sin(theta1);
		v = X*sin(theta1) + Y*cos(theta1);
		Mat temp1 = -((u.mul(u) + v.mul(v)) / (2 * (sigma*sigma)));
		Mat temp2 = (1 / (2 * pi*(sigma*sigma)))*expmartix(temp1);//生成高斯核
		Mat G0;
		G0.convertTo(G0, CV_64FC1, 1);
		G0 = temp2;

		//point[2] = clock();
		//cout << "生成高斯核用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[2] - point[1]) / CLOCKS_PER_SEC << "s\n";

		Mat G20;// = (((u.mul(u). / sigma)))
		G20.convertTo(G20, CV_64FC1, 1);
		Mat uu;
		uu.convertTo(uu, CV_64FC1, 1);
		uu = u.mul(u);
		G20 = (uu / pow(sigma, 4) - (1 / (sigma*sigma))).mul(G0);//二阶导数
																 //cout << G20;
		u = X*cos(theta2) - Y*sin(theta2);
		uu = u.mul(u);
		Mat G260;
		G260 = (uu / pow(sigma, 4) - (1 / (sigma*sigma))).mul(G0);
		u = X*cos(theta3) - Y*sin(theta3);
		uu = u.mul(u);
		Mat G2120;
		G2120 = (uu / pow(sigma, 4) - (1 / (sigma*sigma))).mul(G0);
		G20 = G20 - mean(G20);
		G260 = G260 - mean(G260);
		G2120 = G2120 - mean(G2120);
		//point[3] = clock();//计时点3
						   //cout << "二阶导模板用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[3] - point[2]) / CLOCKS_PER_SEC << "s\n";

						   //cv::Mat J20 = Conv2d_fft(imgI, G20);
						   //point[4] = clock();//计时点3
						   //cout << "J20用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[4] - point[3]) / CLOCKS_PER_SEC << "s\n";
						   //cv::Mat J260 = Conv2d_fft(imgI, G260);
						   //point[5] = clock();//计时点3
						   //cout << "J260用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[5] - point[4]) / CLOCKS_PER_SEC << "s\n";
						   //cv::Mat J2120 = Conv2d_fft(imgI, G2120);
						   //point[6] = clock();//计时点3
						   //cout << "J2120用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[6] - point[5]) / CLOCKS_PER_SEC << "s\n";
		cv::Mat J20, J260, J2120;
		Conv2d_fft_first(imgI, G20, G260, G2120, J20, J260, J2120);
		//point[4] = clock();//计时点3
						   //cout << "一阶导卷积" << nrScales - scale + 1 << "次" << (double)(point[4] - point[3]) / CLOCKS_PER_SEC << "s\n";

		cv::Mat Nr, Dr;
		Nr = ((2 * sqrt(3)) / 9)*((J260.mul(J260)) - (J2120.mul(J2120)) + (J20.mul(J260)) - (J20.mul(J2120)));//C3
		Dr = (2.0 / 9.0) *((2 * (J20.mul(J20))) - (J260.mul(J260)) - (J2120.mul(J2120)) + (J20.mul(J260)) - (2 * (J260.mul(J2120))) + (J20.mul(J2120)));//C2
																																						//                                                                                                                                                //2019.3.11--16:43:angels数据不正确
																																						//                                                                                                                                                //2.单线程多处理

																																						//2019.3.12--9:33: (2/9)按int类型计算为0，改为（2.0/9.0）
		cv::Mat angles = cv::Mat::zeros(imgI.rows, imgI.cols, imgI.type());
		for (int i = 0; i < imgI.rows; i++) {
			for (int j = 0; j < imgI.cols; j++) {
				if (Dr.ptr<double>(i)[j] > 0) {
					angles.ptr<double>(i)[j] = atan(Nr.ptr<double>(i)[j] / (Dr.ptr<double>(i)[j]));
				}
				if (Nr.ptr<double>(i)[j] >= 0 && Dr.ptr<double>(i)[j] < 0) {
					angles.ptr<double>(i)[j] = atan(Nr.ptr<double>(i)[j] / (Dr.ptr<double>(i)[j])) + pi;
				}
				if (Nr.ptr<double>(i)[j] < 0 && Dr.ptr<double>(i)[j] < 0) {
					angles.ptr<double>(i)[j] = atan(Nr.ptr<double>(i)[j] / (Dr.ptr<double>(i)[j])) - pi;
				}
				if (Nr.ptr<double>(i)[j] > 0 && Dr.ptr<double>(i)[j] == 0) {
					angles.ptr<double>(i)[j] = pi / 2;//正无穷
				}
				if (Nr.ptr<double>(i)[j] < 0 && Dr.ptr<double>(i)[j] == 0) {
					angles.ptr<double>(i)[j] = -pi / 2;//负无穷
				}
				//cout << angles.ptr<double>(i)[j]<<",";
			}
		}
		angles = 0.5*angles;//区间【-pi/2,pi/2】。
		//point[7] = clock();//计时点3
						   //cout << "计算曲率方向用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[7] - point[6]) / CLOCKS_PER_SEC << "s\n";
						   //        cout << angles;//2019.3.11---15：20：angles为全0矩阵，说明赋值过程或者Nr，Dr数据有问题
						   //double sigmal = 5 * sigma;//???????为什么要生成5倍大的高斯核
		double sigmal = 5 * sigma;
		siz = round(sigmal * 4) + 1;
		if (siz % 2 == 0) {
			siz = siz + 1;
		}
		meshgrid(cv::Range(-siz, siz), cv::Range(-siz, siz), X, Y);
		theta1 = 0;
		//cout << X;

		u = X*cos(theta1) - Y*sin(theta1);
		v = X*sin(theta1) + Y*cos(theta1);
		u.convertTo(u, CV_64FC1, 1);
		v.convertTo(v, CV_64FC1, 1);
		//    cout << u;
		double G0p1, G0p2;

		//uuvv.convertTo(uuvv, CV_64FC1,1);
		G0p1 = 1.0 / (2.0 * pi*(sigmal*sigmal));
		G0p2 = 2.0 * (sigmal*sigmal);
		cv::Mat uuvv = -(u.mul(u) + v.mul(v));
		uuvv.convertTo(uuvv, CV_64FC1, 1);
		uuvv = (uuvv / (2.0 * sigmal*sigmal));
		cv::Mat G0_half;
		G0_half.convertTo(G0_half, CV_64FC1, 1);
		G0_half = expmartix(uuvv);
		//cout << G0_half;
		G0_half = G0p1*G0_half;//生成高斯核
							   //cout << G0_half;
							   //(1 / (2 * pi*(sigmal*sigmal)))*expmartix(-((u.mul(u) + v.mul(v)) / (2 * (sigmal*sigmal))));
		cv::Mat G0u(G0_half.rows, G0_half.cols, G0_half.type());// = G0_half.mul(u);// u.mul(G0_half);
		for (int i = 0; i < G0u.rows; i++) {
			for (int j = 0; j < G0u.cols; j++) {
				G0u.ptr<double>(i)[j] = G0_half.ptr<double>(i)[j] * u.ptr<double>(i)[j];
			}
		}
		//G0u = G0_half.mul(u);
		//        cout << G0u;
		G0u.convertTo(G0u, CV_64FC1, 1);
		G0u = (-1)*((1 / sigmal)*(1 / sigmal))*G0u;
		//        cout << G0u;
		//(-1)*((1 / sigmal)*(1 / sigmal))*u.mul(G0_half);
		theta1 = pi / 2;
		u = X*cos(theta1) - Y*sin(theta1);
		u.convertTo(u, CV_64FC1, 1);
		v.convertTo(v, CV_64FC1, 1);
		cv::Mat G90u(G0_half.rows, G0_half.cols, G0_half.type());
		for (int i = 0; i < G0u.rows; i++) {
			for (int j = 0; j < G0u.cols; j++) {
				G90u.ptr<double>(i)[j] = G0_half.ptr<double>(i)[j] * u.ptr<double>(i)[j];
			}
		}
		G90u = (-1)*((1 / sigmal)*(1 / sigmal))*G90u;
		//        cout << G90u;
		//= (-1)*((1 / sigmal)*(1 / sigmal))*u.mul(G0_half);
		G0u = G0u - mean(G0u);
		G90u = G90u - mean(G90u);
		//point[8] = clock();//计时点9
						   //cout << "一阶导模板用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[8] - point[7]) / CLOCKS_PER_SEC << "s\n";
						   //cv::Mat J0u = Conv2d_fft(imgI, G0u);
						   //cv::Mat J0u = D2_Filter(imgI, G0u);
						   //    cout << J0u;
						   //J0u= J0u*(1/1000000.0);
						   //point[9] = clock();//计时点
						   //cout << "J0u用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[9] - point[8]) / CLOCKS_PER_SEC << "s\n";
						   //cv::Mat J90u = Conv2d_fft(imgI, G90u);
						   //point[10] = clock();//计时点
						   //cout << "J90u用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[10] - point[9]) / CLOCKS_PER_SEC << "s\n";
						   //cv::Mat J90u = D2_Filter(imgI, G90u);
						   //J90u= J90u*(1 / 1000000.0);
						   //cout << J0u;

						   //2
		cv::Mat J0u, J90u;
		Conv2d_fft_second(imgI, G0u, G90u, J0u, J90u);
		//point[10] = clock();//计时点
							//cout << "二阶导卷积" << nrScales - scale + 1 << "次" << (double)(point[10] - point[8]) / CLOCKS_PER_SEC << "s\n";
							//-------------
		J0u.convertTo(J0u, CV_64FC1, 1);
		J90u.convertTo(J90u, CV_64FC1, 1);
		cv::Mat J2, J1;
		cv::Mat J2t1, J2t2, J2t3;
		J2t1 = (1.0 + (2.0 * cosmartix(2.0 * angles))).mul(J20);
		J2t2 = (1.0 - cosmartix(2.0 * angles) + (sqrt(3)*sinmartix(2.0 * angles))).mul(J260);
		J2t3 = (1.0 - cosmartix(2.0 * angles) - (sqrt(3)*sinmartix(2.0 * angles))).mul(J2120);
		J2t1.convertTo(J2t1, CV_64FC1, 1);
		J2t2.convertTo(J2t2, CV_64FC1, 1);
		J2t3.convertTo(J2t3, CV_64FC1, 1);
		//cout << J2t3;
		J2 = (1.0 / 3.0)*(J2t1 + J2t2 + J2t3);
		J1 = (J0u.mul(cosmartix(angles))) + (J90u.mul(sinmartix(angles)));
		J1.convertTo(J1, CV_64FC1, 1);
		J2.convertTo(J2, CV_64FC1, 1);
		//cout << J2;//-------2019.3.13--9:55：J2大了两个数量级，需要找到其中的问题
		cv::Mat psi_scale;

		cv::Mat psi1, psi2, psi3, psi4;
		psi1 = (sigma*sigma)*(absmartix(J2));
		//cout << psi1;
		psi1.convertTo(psi1, CV_64FC1, 1);
		psi2 = 1.0 + absmartix(J1).mul(absmartix(J1));
		//        cout << psi2;
		psi2.convertTo(psi2, CV_64FC1, 1);
		psi3 = (psi2.mul(psi2)).mul(psi2);
		//    cout << psi3;
		psi3.convertTo(psi3, CV_64FC1, 1);
		psi4 = 1.0 / sqrtmartix(psi3);
		psi4.convertTo(psi4, CV_64FC1, 1);
		//    cout << psi4;
		//point[11] = clock();//计时点9
							//cout << "计算曲率用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[11] - point[10]) / CLOCKS_PER_SEC << "s\n";
		if (preservePolarity == 0) {
			psi_scale = psi1.mul(psi4);

		}
		else {
			psi_scale = (J2.mul(psi4));
			//cout << psi_scale;
			psi_scale = (sigma*sigma)*psi_scale;
		}
		//cout << psi_scale;
		psi_scale.convertTo(psi_scale, CV_64FC1, 1);
		if (scale == nrScales) {
			psi = psi_scale;
			psi.convertTo(psi, CV_64FC1, 1);
			orient = angles;
			cv::Mat one = Mat::ones(Size(C, R), CV_64FC1);
			scaleMap = sigma*one;
			scaleMapIdx = scale*one;
			scaleMap.convertTo(scaleMap, CV_64FC1, 1);
			scaleMapIdx.convertTo(scaleMapIdx, CV_64FC1, 1);
			if (preservePolarity == 1) {
				polarity = sign(psi);
				polarity.convertTo(polarity, CV_64FC1, 1);
			}
		}
		else {
			if (preservePolarity == 0) {
				idx = bigger(psi_scale, psi);
				psi = repl(psi, psi_scale, idx);
				orient = repl(orient, angles, idx);
				scaleMap = orepl(scaleMap, sigma, idx);
				scaleMapIdx = orepl(scaleMapIdx, scale, idx);
			}
			else {
				polarity_scale = sign(psi_scale);
				polarity_scale.convertTo(polarity_scale, CV_64FC1, 1);
				idx = bigger(polarity_scale.mul(psi_scale), polarity.mul(psi));
				idx.convertTo(idx, CV_64FC1, 1);
				psi = repl(psi, psi_scale, idx);
				orient = repl(orient, angles, idx);
				orient.convertTo(orient, CV_64FC1, 1);
				scaleMap = orepl(scaleMap, sigma, idx);
				scaleMapIdx = orepl(scaleMapIdx, scale, idx);
				scaleMap.convertTo(scaleMap, CV_64FC1, 1);
				scaleMapIdx.convertTo(scaleMapIdx, CV_64FC1, 1);
				polarity = repl(polarity, polarity_scale, idx);
				polarity.convertTo(polarity, CV_64FC1, 1);
				//cv::imwrite("par.jpg", im2int( polarity));
			}
		}
		angles.release();
		//point[12] = clock();//计时点9
							//cout << "曲率更新后处理用时，循环第" << nrScales - scale + 1 << "次" << (double)(point[12] - point[11]) / CLOCKS_PER_SEC << "s\n";
	}
	//----------------------------------循环结束
	//point[13] = clock();//计时点9
						//cout << "循环用时" << (double)(point[13] - point[0]) / CLOCKS_PER_SEC << "s\n";
	cv::Mat pos_psi = cv::Mat::zeros(Size(C, R), CV_64FC1);
	cv::Mat neg_psi = cv::Mat::zeros(Size(C, R), CV_64FC1);
	idx = cv::Mat::zeros(Size(C, R), CV_64FC1);
	if (preservePolarity == 1) {
		idx.convertTo(idx, CV_64FC1, 1);
		for (int i = 0; i < polarity.rows; i++) {
			for (int j = 0; j < polarity.cols; j++) {
				if (polarity.ptr<double>(i)[j] != -1) {
					idx.ptr<double>(i)[j] = 1;
				}//可能是初始化或者矩阵大小的问题
				else {
					idx.ptr<double>(i)[j] = 0;
				}
			}
		}
		//cout << idx;
		neg_psi = repl(neg_psi, psi, idx);
		neg_psi.convertTo(neg_psi, CV_64FC1, 1);
		//        cout << neg_psi;
		cv::Mat negimg = mat2gray(neg_psi);
		negimg.convertTo(negimg, CV_64FC1, 255);
		//cout << negimg;

		Enhanceimg = im2int(negimg);
		//cout << Enhanceimg;
	}
	else
	{
		for (int i = 0; i < polarity.rows; i++) {
			for (int j = 0; j < polarity.cols; j++) {
				if (polarity.ptr<double>(i)[j] != -1) {
					idx.ptr<double>(i)[j] = 1;
				}
				else {
					idx.ptr<double>(i)[j] = 0;
				}
			}
		}
		pos_psi = repl(pos_psi, -psi, idx);
		cv::Mat posimg = mat2gray(pos_psi);
		Enhanceimg = im2int(posimg);
	}
	//char * outimg = "enhanceN=5_422.jpg";
	//cv::imwrite(dst, Enhanceimg);
	Enhanceimg.copyTo(dst);
	/*cv::imshow("en", Enhanceimg);
	cvWaitKey(0);*/
	//point[14] = clock();//计时点9
	imgI.release();
	psi.release();
	orient.release();
	Enhanceimg.release();					//cout << "循环后处理用时" << (double)(point[14] - point[13]) / CLOCKS_PER_SEC << "s\n";
	pos_psi.release();
	neg_psi.release();
	idx.release();
	return 0;
}

bool Curvature(const cv::Mat& inputimg,cv::Mat& retimg, int n) {
	/*clock_t start, end;
	start = clock();*/
	//cv::Mat imgI1s;
	//const std::string imgroad = "D:/bishe/Exam1/Branch/LiuHuan_SXX_2546.jpg";
	//imgI1s = imread(inputimg);
	cv::Mat imgI1;
	inputimg.copyTo(imgI1);
	//std::vector<std::vector<double> > I1s (imgI1s.rows,std::vector<double>(imgI1s.cols));
	//cvtColor(imgI1s, imgI1, CV_BGR2GRAY);
	imgI1.convertTo(imgI1, CV_64FC1, 1);
	//resize(imgI1, imgI1, Size(), 1, 1);
	//cv::Mat imgI1{ I1s };
	//end = clock();
	//cout << "curvature用时" << (double)(end - start) / CLOCKS_PER_SEC << "s\n";
	//string dst = "aa.jpg";
	cv::Mat outimg;
	CurvatureIndex2D(imgI1, outimg, n, 1, 1.5);
	//cv::Mat outimg = imread(dst);
	/*cv::imshow("outimg", outimg);
	cvWaitKey(0);*/
	if(outimg.channels()>1)
		cvtColor(outimg, outimg, CV_BGR2GRAY);
	outimg.copyTo(retimg);
	outimg.release();
	return true;
}
bool calCurCost(const cv::Mat & src,cv::Mat &dst) {
	cv::Mat temp;
	verticalmeanfilter(src,temp, 50);
	////-----------------part1
	//cv::Mat ds;
	/*cv::imshow("tmp", temp);
	cvWaitKey(0);*/
	Curvature(temp, dst, 3);//.copyTo(dst);
	/*cv::imshow("ds", dst);
	cvWaitKey(0);*/
	temp.release();
	return true;
}
void BinaryImageAdvance( cv::Mat  bmp_org, Mat &bmp_binaried, int theshold) {

	unsigned char * pBmp = bmp_org.data;
	int width = bmp_org.cols;
	int height = bmp_org.rows;
	int channels = bmp_org.channels();
	bmp_binaried.create(height, width, CV_8UC1);
	unsigned char * pBmp_binaried = bmp_binaried.data;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (pBmp[i*width + j] <= theshold)
				pBmp_binaried[i*width + j] = 0;
			else
				pBmp_binaried[i*width + j] = 255;
		}
	}
}

void FeatureCaputure(cv::Mat img, cv::Mat & dst, vector<CArea> &ConArea)
{
	clock_t start, end;
	start = clock();

	double perlength =sqrt(double(img.rows*img.cols))/1000*350 ;
	/*cv::Mat img = imread(src);
	cvtColor(img, img, CV_RGB2GRAY);*/
	BinaryImageAdvance(img, img, 127);
	int cols = img.cols;
	int rows = img.rows;

	Point seed, neighbor;//定义种子，和领域
	stack<Point> point;//定义种子堆栈
	int area = 0;//计算连通域的面积
	int leftpoint = 0;
	int rightpoint = 0;
	int uppoint = 0;
	int downpoint = 0;;//定义矩形框的边界
	Rect a;
	CArea carea;
	int **mark;
	mark = new int *[rows];
	int lable = 0;
	for (int i = 0; i<rows; i++) {
		mark[i] = new int[cols];
	}
	//初始化
	for (int i = 0; i<rows; i++)
		for (int j = 0; j<cols; j++)
			mark[i][j] = 0;
	Point weight;
	weight.x = 0;
	weight.y = 0;
	for (int i = 0; i<img.rows; i++)
	{
		uchar *pData = img.ptr<uchar>(i);
		for (int j = 0; j<img.cols; j++)
		{
			if (pData[j] == 255 && mark[i][j] == 0)
			{
				//mark[i][j]=1;
				seed = Point(j, i);//注意这里是j,i

				point.push(seed);//种子入栈
				leftpoint = seed.x;
				rightpoint = seed.x;
				uppoint = seed.y;
				downpoint = seed.y;
				area = 0;//置零
				weight.x = 0;
				weight.y = 0;
				area++;//
				weight.x += j;
				weight.y += i;
				lable++;
				mark[i][j] = lable;
				while (!point.empty()) {
					if (seed.x + 1 < cols) {
						neighbor = Point(seed.x + 1, seed.y);
						//右

						if (seed.x != (img.cols - 1) && img.at<uchar>(neighbor) == 255 && mark[neighbor.y][neighbor.x] == 0)
						{
							mark[neighbor.y][neighbor.x] = lable;
							point.push(neighbor);
							area++;
							weight.x += neighbor.x;
							weight.y += neighbor.y;
							if (rightpoint < neighbor.x)
								rightpoint = neighbor.x;
						}
					}
					//下
					if (seed.y + 1 < rows) {
						neighbor = Point(seed.x, seed.y + 1);

						if ((seed.y != (rows - 1)) && (img.at<uchar>(neighbor) == 255) && mark[neighbor.y][neighbor.x] == 0)
						{
							mark[neighbor.y][neighbor.x] = lable;
							point.push(neighbor);
							area++;
							weight.x += neighbor.x;
							weight.y += neighbor.y;
							if (downpoint < neighbor.y) downpoint = neighbor.y;

						}
					}
					//左
					if (seed.x - 1 >= 0) {
						neighbor = Point(seed.x - 1, seed.y);

						if ((seed.x != 0) && (img.at<uchar>(neighbor) == 255) && mark[neighbor.y][neighbor.x] == 0)
						{
							mark[neighbor.y][neighbor.x] = lable;
							point.push(neighbor);
							area++;
							weight.x += neighbor.x;
							weight.y += neighbor.y;
							if (leftpoint > neighbor.x) leftpoint = neighbor.x;
							//mark[i][j - 1] = 1;
						}
					}
					//上
					if (seed.y - 1 >= 0) {
						neighbor = Point(seed.x, seed.y - 1);
						if ((seed.y != 0) && (img.at<uchar>(neighbor) == 255) && mark[neighbor.y][neighbor.x] == 0)
						{
							mark[neighbor.y][neighbor.x] = lable;
							point.push(neighbor);
							area++;
							weight.x += neighbor.x;
							weight.y += neighbor.y;
							if (uppoint > neighbor.y)
								uppoint = neighbor.y;
						}
					}
					seed = point.top();
					//mark[seed.y][seed.x]=1;
					point.pop();//取出栈顶元素
				}
				//求周长
				double perimeter = 0;
				for (int m = uppoint; m <= downpoint; m++)
				{
					for (int n = leftpoint; n <= rightpoint; n++)
					{
						if ((m - 1) >= 0 && (m + 1)<rows && (n - 1) >= 0 && (n + 1)<cols)
						{
							if (mark[m][n] == lable)
							{
								int a1 = 0, a2 = 0;
								if (mark[m + 1][n] == 0 || mark[m - 1][n] == 0)
								{
									perimeter++;
									a1 = 1;
								}
								if (mark[m][n + 1] == 0 || mark[m][n - 1] == 0)
								{
									perimeter++;
									a2 = 1;
								}
								if (a1 == 1 && a2 == 1)
									perimeter = perimeter + 1.41421356 - 2;

							}
						}
					}
				}
				/*double hd = (rightpoint - leftpoint);
				double vd = (downpoint - uppoint);
				double hv = hd / vd;
				*/
				//--------------------------------2019-4.23 删除小连通域
				if (perimeter < perlength/*&&hv>0.2&&hv<5*/) {
					if (uppoint - 1 >= 0 && downpoint + 1 <= img.rows&&leftpoint - 1 >= 0 && rightpoint + 1 <= img.cols) {
						for (int i = uppoint - 1; i < downpoint + 1; i++) {
							for (int j = leftpoint - 1; j < rightpoint + 1; j++) {
								mark[i][j] = 9000;
							}
						}
					}
					else {
						for (int i = uppoint; i < downpoint; i++) {
							for (int j = leftpoint; j < rightpoint; j++) {
								mark[i][j] = 9000;
							}
						}
					}

				}
				else {
					a = Rect(leftpoint, uppoint, rightpoint - leftpoint, downpoint - uppoint);
					rectangle(img, a, (255, 0, 0));
					carea.area = area;
					carea.rectangle = a;
					carea.lable = lable;
					carea.perimeter = perimeter;
					ConArea.push_back(carea);
				}
				//-----------------------------------
			}
		}
	}
	cv::Mat newimg(rows, cols, CV_8UC1);
	unsigned char *pnewimg = newimg.data;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			pnewimg[(i *cols + j)] = 0;
		}
	}
	//上色
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (mark[i][j] == 9000)
				mark[i][j] = 0;
			pnewimg[(i *cols + j)] = 200 * mark[i][j];
		}
	}
	BinaryImageAdvance(newimg, newimg, 10);
	newimg.copyTo(dst);

	//cv::imwrite(dst, newimg);
	//resize(newimg, newimg, (0.3, 0.3));
	//imshow("123", newimg);
	//cout << "标号" << "\t" << "面积" << "\t" << "周长" << "\t" << endl;
	//for (vector<CArea>::iterator i = ConArea.begin(); i<ConArea.end(); i++)
	//{
	//	cout << i->lable << "\t" << i->area << "\t" << i->perimeter << "\t" << endl;
	//	//rectangle(newimg, i->rectangle, cvScalar(255, 255, 255, 255));
	//}
	for (int i = 0; i<rows; i++) {
		delete[] mark[i];// = new int[cols];
	}
	delete[] mark;
	end = clock();
	//std::cout << "联通分析用时: " << (double)(end - start) / CLOCKS_PER_SEC << "s\n";
}

bool maskcur(const cv::Mat &thesholdimg, cv::Mat & curimg) {
	unsigned char * pt = thesholdimg.data;//二值图
	unsigned char * pc = curimg.data;//曲率图
	int ro = curimg.rows;
	int co = curimg.cols;
	for (int i = 0; i<ro; i++) {
		for (int j = 0; j < co; j++) {
			if (pt[i*co + j] < 10) {
				pc[i*co + j] = 0;//如果不属于裂缝像素，就把曲率图的对应值也设为0
			}
		}
	}
	return true;
}
int ifacceptcorpoint(const vector<CArea> &ConArea, const cv::Point& p) {
	//if (ConArea.size() == 1)return true;
	int siz = ConArea.size();
	int innum = 0;
	for (int i = 0; i < siz; i++) {
		int l=ConArea[i].rectangle.x;
		int r = ConArea[i].rectangle.width + l;
		int u = ConArea[i].rectangle.y;
		int d = ConArea[i].rectangle.height + u;
		if (p.x <= r&&p.x >= l&&p.y <= d&&p.y >= u)
			innum++;
		else if (p.x <= r&&p.x >= l && (p.y > d || p.y < u)) {
			int dis = min(abs(p.y - d), abs(p.y - u));
			if (dis < 10)
				innum++;
		}
		else if ((p.x > r||p.x < l)&&p.y <= d&&p.y >= u) {
			int dis = min(abs(p.x - r), abs(p.x - l));
			if (dis < 10)
				innum++;
		}
		else {
			float disy = min(abs(p.y - d), abs(p.y - u));
			float disx = min(abs(p.x - r), abs(p.x - l));
			float dis = sqrt((disx*disx) + (disy*disy));
			if (dis < 10.0)
				innum++;
		}

	}
	if (innum == 0) return 0;
	if (innum == 1)return 1;
	else return 2;
}
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
//void calcrackareaby()