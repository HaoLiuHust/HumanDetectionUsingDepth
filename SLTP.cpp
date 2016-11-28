#include "SLTP.h"
#include <iterator>
#include "Utils.h"
using namespace cv;

void SLTP::compute(const Mat& img, vector<float>& features) const
{
	CV_Assert(img.size() == winSize);

	Mat dx, dy;
	compute_dxdy(img, dx, dy);

	Mat signdx, signdy;
	compute_sign(dx, signdx);
	compute_sign(dy, signdy);

	compute_histwin(signdx, signdy, features);
}

void SLTP::setDefaultParams()
{
	winSize = Size(64, 128);
	cellSize = Size(8, 8);
	signThreshold = 30;	
	//nlevels = 64;
	//scale0 = 1.05;
}

void SLTP::cal_params()
{
	numCellC = winSize.height / cellSize.height;
	numCellR = winSize.width / cellSize.width;

	numCellPerWin = numCellR*numCellC;

	//maskdx = Mat::zeros(3, 3, CV_8SC1);
	//maskdy = Mat::zeros(3, 3, CV_8SC1);

	//maskdx= (Mat_<char>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
	//maskdy = (Mat_<char>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);

	maskdx = (Mat_<char>(3, 3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
	maskdy = (Mat_<char>(3, 3) << 0, -1, 0, 0, 0, 0, 0, 1, 0);

	//compute feature len
	featurelen = 9 * numCellPerWin;

	//set sign array
	for (int i = 0,p=0; i < 3;++i)
	{
		for (int j = 0; j < 3; ++j, ++p)
		{
			signarry[i][j] = p;
		}
	}
}

void SLTP::compute_dxdy(const Mat& img, Mat& dx, Mat& dy) const
{
	filter2D(img, dx, CV_32FC1, maskdx);
	filter2D(img, dy, CV_32FC1, maskdy);
}

void SLTP::compute_sign(const Mat& dimg, Mat& signimg) const
{
	signimg=Mat::zeros(dimg.size(), CV_8SC1);

	signimg.setTo(-1, dimg <= -signThreshold);
	signimg.setTo(1, dimg >= signThreshold);
}

void SLTP::compute_histcell(const Mat& signimgx, const Mat& signimgy, vector<float>& hist) const
{
	CV_Assert(signimgx.rows == signimgy.rows&&signimgx.cols==signimgy.cols);
	hist.clear();
	hist.resize(9,0);

	for (int i = 0;i<signimgx.rows;++i)
	{
		for (int j = 0; j < signimgx.cols;++j)
		{
			int x = signimgx.at<char>(i, j);
			int y = signimgy.at<char>(i, j);
			int mode = signarry[x + 1][y + 1];

			++hist[mode];
		}
	}
}

void SLTP::compute_histcell(const Mat& signimgx, const Mat& signimgy, float* hist) const
{
	CV_Assert(signimgx.rows == signimgy.rows&&signimgx.cols == signimgy.cols);

	const char* ptrx, *ptry;

	for (int i = 0; i < signimgx.rows; ++i)
	{
		ptrx = signimgx.ptr<char>(i);
		ptry = signimgy.ptr<char>(i);
		for (int j = 0; j < signimgx.cols; ++j)
		{
			int x = ptrx[j];
			int y = ptry[j];
			int mode = signarry[x + 1][y + 1];

			++hist[mode];
		}
	}

	normalizeBlockHistogram(hist);
}

void SLTP::compute_histwin(const Mat& signimgx, const Mat& signimgy, vector<float>& hist) const
{
	CV_Assert(signimgx.rows == signimgy.rows&&signimgx.cols == signimgy.cols);
	hist.clear();

	int numCellcols = signimgx.cols / cellSize.width;
	int numCellrows = signimgx.rows / cellSize.height;

	hist.resize(9* numCellcols*numCellrows, 0);

	//scan all cells
//#pragma omp parallel
	{
		//Mat cellx, celly;
		for (int i = 0; i < numCellrows; ++i)
		{
			for (int j = 0; j < numCellcols; ++j)
			{
				Rect cellrt(j*cellSize.width, i*cellSize.height, cellSize.width, cellSize.height);

				Mat cellx = signimgx(cellrt);
				Mat celly = signimgy(cellrt);

				float* histptr = &hist[9 * (numCellcols*i + j)];
				compute_histcell(cellx, celly, histptr);
			}
		}
	}
}

void SLTP::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
{
	if (groupThreshold <= 0 || rectList.empty()||rectList.size()==1)
	//if (groupThreshold <= 0 || rectList.empty())

	{
		return;
	}

	CV_Assert(rectList.size() == weights.size());

	vector<int> labels;
	int nclasses = partition(rectList, labels, SimilarRects(eps));

	vector<cv::Rect_<double> > rrects(nclasses);
	vector<int> numInClass(nclasses, 0);
	vector<double> foundWeights(nclasses, -std::numeric_limits<double>::max());
	int i, j, nlabels = (int)labels.size();

	for (i = 0; i < nlabels; i++)
	{
		int cls = labels[i];
		rrects[cls].x += rectList[i].x;
		rrects[cls].y += rectList[i].y;
		rrects[cls].width += rectList[i].width;
		rrects[cls].height += rectList[i].height;
		foundWeights[cls] = max(foundWeights[cls], weights[i]);
		numInClass[cls]++;
	}

	for (i = 0; i < nclasses; i++)
	{
		// find the average of all ROI in the cluster
		cv::Rect_<double> r = rrects[i];
		double s = 1.0 / numInClass[i];
		rrects[i] = cv::Rect_<double>(cv::saturate_cast<double>(r.x*s),
			cv::saturate_cast<double>(r.y*s),
			cv::saturate_cast<double>(r.width*s),
			cv::saturate_cast<double>(r.height*s));
	}

	rectList.clear();
	weights.clear();

	for (i = 0; i < nclasses; i++)
	{
		cv::Rect r1 = rrects[i];
		int n1 = numInClass[i];
		double w1 = foundWeights[i];
		if (n1 <= groupThreshold)
			continue;
		// filter out small rectangles inside large rectangles
		for (j = 0; j < nclasses; j++)
		{
			int n2 = numInClass[j];

			if (j == i || n2 <= groupThreshold)
				continue;

			cv::Rect r2 = rrects[j];

			int dx = cv::saturate_cast<int>(r2.width * eps);
			int dy = cv::saturate_cast<int>(r2.height * eps);

			if (r1.x >= r2.x - dx &&
				r1.y >= r2.y - dy &&
				r1.x + r1.width <= r2.x + r2.width + dx &&
				r1.y + r1.height <= r2.y + r2.height + dy &&
				(n2 > std::max(3, n1) || n1 < 3))
				break;
		}

		if (j == nclasses)
		{
			rectList.push_back(r1);
			weights.push_back(w1);
		}
	}
}

void SLTP::compute_img(const cv::Mat & img, vector<float>& features) const
{
	Mat dx, dy;
	compute_dxdy(img, dx, dy);

	Mat signdx, signdy;
	compute_sign(dx, signdx);
	compute_sign(dy, signdy);

	compute_histwin(signdx, signdy, features);
}

void SLTP::get_winfeature(vector<float>& featuresimg, vector<float>& featureswin, cv::Point& startpos, int cellimgcols) const
{
	featureswin.clear();
	featureswin.resize(featurelen,0);

	int cellfeaturelen = 9 * numCellR;
	for (int i = 0; i < numCellC;++i)
	{
		int sindex = cellimgcols*(startpos.y+i) + startpos.x;
		sindex *= 9;
		memcpy(&featureswin[i*cellfeaturelen], &featuresimg[sindex], sizeof(float)*cellfeaturelen);
	}
}

void SLTP::normalizeBlockHistogram(float* blockhist) const
{
	float* hist = &blockhist[0];
	size_t i, sz = 9;

	float sum = 0;
	for (i = 0; i < sz; i++)
		sum += hist[i] * hist[i];

	float scale = 1.f / (std::sqrt(sum) + sz*0.1f), thresh = 0.8;

	for (i = 0, sum = 0; i < sz; i++)
	{
		hist[i] = std::min(hist[i] * scale, thresh);
		sum += hist[i] * hist[i];
	}

	scale = 1.f / (std::sqrt(sum) + 1e-3f);

	for (i = 0; i < sz; i++)
		hist[i] *= scale;
}

void SLTP::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	sltpsvm = _svm;
}


void SLTP::loadSvmDetector(const string & xmlfile)
{
	sltpsvm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void SLTP::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold/*=0*/, cv::Size winStride/*=cv::Size()*/, const vector<cv::Point>& locations/*=vector<cv::Point>()*/) const
{
	foundlocations.clear();
	if (sltpsvm->empty())
	{
		cerr << "no svm" << endl;
		return;
	}

	if (winStride == Size())
	{
		winStride = cellSize;
	}

	//Í¼ÏñÌî³ä
	Size stride(gcd(winStride.width, cellSize.width), gcd(winStride.height, cellSize.height));

	size_t nwindows = locations.size();
	Size padding;
	padding.width = (int)alignSize(std::max(padding.width, 0), stride.width);
	padding.height = (int)alignSize(std::max(padding.height, 0), stride.height);

	Size paddedImgSize(img.cols + padding.width * 2, img.rows + padding.height * 2);

	if (!nwindows)
	{
		nwindows = Size((paddedImgSize.width - winSize.width) / winStride.width + 1,
			(paddedImgSize.height - winSize.height) / winStride.height + 1).area();
	}

	Mat paddedimg;
	if (img.size() != paddedimg.size())
	{
		copyMakeBorder(img, paddedimg, padding.height, padding.height, padding.width, padding.width, BORDER_REFLECT_101);
	}
	else
		paddedimg = img;

	int numwinR = (paddedImgSize.width - winSize.width) / winStride.width + 1;
	int numwinC = (paddedImgSize.height - winSize.height) / winStride.height + 1;

	vector<float> featuresimg;
	compute_img(paddedimg, featuresimg);
	int numCellcols = paddedimg.cols / cellSize.width;
	int numCellrows = paddedimg.rows / cellSize.height;

	//Mat winimg;
	if (locations.size())
	{
		Point pt0;
		for (int i = 0; i < nwindows; ++i)
		{
			pt0 = locations[i];
			pt0 = locations[i];
			if (pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
				pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height)
				continue;

			Point pt(pt0.x + padding.width, pt0.y + padding.height);

			Mat winimg = paddedimg(Rect(pt.x, pt.y, winSize.width, winSize.height));
			vector<float> feature;
			compute(winimg, feature);
			float response = sltpsvm->predict(feature);
			//cout << response << endl;
			if ((int)response == 1)
			{
				foundlocations.push_back(pt0);
				weights.push_back(response);
			}
		}
	}
	else
	{
		//scan over windows
//#pragma omp parallel
	{
		Mat winimg;
		for (int j = 0; j < numwinC; ++j)
		{
			for (int i = 0; i < numwinR; ++i)
			{
				Point pt0;
				winimg = paddedimg(Rect(i*winStride.width, j*winStride.height, winSize.width, winSize.height));

				int cellindexrow = j*winStride.height / cellSize.height;
				int cellindexcol = i*winStride.width / cellSize.width;

				pt0.x = i*winStride.width - padding.width;
				pt0.y = j*winStride.height - padding.height;;

				vector<float> feature;
				Point startpos(cellindexcol, cellindexrow);
				get_winfeature(featuresimg, feature, startpos, numCellcols);

				//compute(winimg, feature);

				Mat result;
				//sltpsvm->predict(winimg,cv::displayStatusBar::)
				float response = sltpsvm->predict(feature, result, ml::StatModel::RAW_OUTPUT);
				response = result.at<float>(0);
				
				if (response<=-hitThreshold)
				{
					foundlocations.push_back(pt0);
					weights.push_back(-response);
				}

			}
		}
	}
	}
}

void SLTP::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}

class Parallel_Detection_SLTP :public ParallelLoopBody
{
private:
	const SLTP* sltp;
	Mat img;
	double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection_SLTP(const SLTP* _sltp, const Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights = 0, vector<double>* _scales = 0)
	{
		sltp = _sltp;
		img = _img;
		hitThreshold = _hitThreshold;
		winStride = _winStride;
		levelScale = _levelScale;
		mtx = _mtx;
		vec = _vec;
		weights = _weights;
		scales = _scales;
	}

	void operator() (const Range& range) const
	{
		int i, i1 = range.start, i2 = range.end;

		double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1 + 1] : std::max(img.cols, img.rows);
		Size maxSz(cvCeil(img.cols / minScale), cvCeil(img.rows / minScale));
		Mat smallerImgBuf(maxSz, img.type());
		vector<Point> locations;
		vector<double> hitsWeights;
		//cout << "sltp" << endl;
		for (i = i1; i < i2; i++)
		{
			double scale = levelScale[i];
			Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
			Mat smallerImg(sz, img.type(), smallerImgBuf.data);
			if (sz == img.size())
				smallerImg = Mat(sz, img.type(), img.data, img.step);
			else
				resize(img, smallerImg, sz, 0.0, 0.0,INTER_NEAREST);
			sltp->detect(smallerImg, locations,hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(sltp->winSize.width*scale), cvRound(sltp->winSize.height*scale));

			mtx->lock();

			for (size_t j = 0; j < locations.size(); j++)
			{
				//cout << "scale: " << scale << endl;
				/*cout << "locations: " << cvRound(locations[j].x*scale) << " " << cvRound(locations[j].y*scale) << endl;
				cout << "scaled WinSize: " << scaledWinSize.width << " " << scaledWinSize.height << endl;*/
				vec->push_back(Rect(cvRound(locations[j].x*scale),
					cvRound(locations[j].y*scale),
					scaledWinSize.width, scaledWinSize.height));
				if (scales)
				{
					scales->push_back(scale);
				}
			}
			mtx->unlock();

			if (weights && (!hitsWeights.empty()))
			{
				mtx->lock();
				for (size_t j = 0; j < locations.size(); j++)
				{
					weights->push_back(hitsWeights[j]);
				}
				mtx->unlock();
			}
		}
	}
};

void SLTP::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels/*=64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	if (sltpsvm->empty())
	{
		cerr << "svm error" << endl;
		return;
	}
	double scale = 1.;
	scale0 = 1.1;
	int levels = 0;
	vector<double> levelScale;
	for (levels = 0; levels < nlevels; ++levels)
	{
		levelScale.push_back(scale);
		if (cvRound(img.cols / scale) < winSize.width || cvRound(img.rows / scale) < winSize.height
			|| scale0 <= 1)
		{
			break;
		}

		scale *= scale0;
	}

	levels = std::max(levels, 1);
	levelScale.resize(levels);

	/*vector<Rect> allCandidates;
	vector<double> tempScales;
	vector<double> tempWeights;*/
	vector<double> foundScales;
	foundlocations.clear();
	weights.clear();

	Mutex mtx;
	parallel_for_(Range(0, levelScale.size()),
		Parallel_Detection_SLTP(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));


	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		//Utils::NonMaximalSuppression(foundlocations, weights, 0.5, 0);
		//groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void SLTP::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride, nlevels, scale0, finalThreshold, usemeanshift);
}
