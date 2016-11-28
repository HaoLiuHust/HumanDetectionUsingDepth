#include "LTDP.h"
#include <iterator>
#include <fstream>
using namespace cv;

void LTDP::setDefaultParams()
{
	winSize = Size(64, 128);
	blockSize = Size(16, 16);
	pthreshold = 70;
}

void LTDP::cal_params()
{
	numBlockC = winSize.height / blockSize.height;
	numBlockR = winSize.width / blockSize.width;

	numBlockPerWin = numBlockR*numBlockC;
	featurelen = featurelenblock*numBlockPerWin;

	masks.resize(8);
	masks[0] = (Mat_<char>(3, 3) << -3, -3, 5,-3, 0, 5, -3, -3, 5);
	masks[1] = (Mat_<char>(3, 3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);
	masks[2] = (Mat_<char>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
	masks[3] = (Mat_<char>(3, 3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);
	masks[4] = (Mat_<char>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);
	masks[5] = (Mat_<char>(3, 3) << -3, -3, -3, 5, 0, -3, 5, 5, -3);
	masks[6] = (Mat_<char>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);
	masks[7] = (Mat_<char>(3, 3) << -3, -3, -3, -3, 0, 5, -3, 5, 5);
	
	const uchar table[256] = { 0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10,
		11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
		16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
		17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
		22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
		58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
		23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
		24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
		29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
		58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34,
		58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
		58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
		36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
		58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
		42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
		47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57 };

	lookUpTable.create(1, 256, CV_8U);
	lookUpTable2 = Mat::ones(1, 256, CV_8UC1);
	lookUpTable2.at<uchar>(0, 0) = 0;

	memcpy((uchar*)lookUpTable.data, table, sizeof(uchar) * 256);
}

void LTDP::compute_Ltpvalue(const cv::Mat& src, cv::Mat& ltpimgpos,cv::Mat& ltpimgneg)const
{
	ltpimgpos = Mat::zeros(src.size(), CV_8UC1);
	ltpimgneg = Mat::zeros(src.size(), CV_8UC1);

	Mat dismap;
	for (int i = 0; i < 8;++i)
	{
		filter2D(src, dismap,CV_32FC1, masks[i]);

		Mat mask1 = (dismap >= pthreshold);
		//LUT(mask1, lookUpTable2, mask1);
		mask1.setTo(1, mask1 != 0);
		ltpimgpos += ((1 << (7 - i))*mask1);

		//imshow("test", ltpimgpos);
		//waitKey();
		Mat mask2 = (dismap<= -pthreshold);
		//LUT(mask2, lookUpTable2, mask2);
		mask2.setTo(1, mask2 != 0);
		ltpimgneg += ((1 << (7 - i))*mask2);
	}
}

void LTDP::compute_histblock(const cv::Mat& ltppos,const cv::Mat& ltpneg, float* feature)const
{
	CV_Assert(ltppos.size() == ltpneg.size());
//#pragma omp parallel for
	for (int i = 0; i < ltppos.rows;++i)
	{
		const uchar *p = ltppos.ptr<uchar>(i);
		const uchar *n = ltpneg.ptr<uchar>(i);
		for (int j = 0; j < ltppos.cols;++j)
		{
			++feature[p[j]];
			++feature[n[j] + 59];
			//++feature[p[j] + 59];
		}
	}

	normalizeBlockHistogram(feature);
}

void LTDP::compute_histwin(const cv::Mat& ltppos, const cv::Mat& ltpneg, vector<float>& features)const
{
	CV_Assert(ltpneg.size() == ltppos.size());
	features.clear();
	features.resize(featurelen, 0);

	Mat posblock, negblock;
//#pragma omp parallel for
	for (int i = 0; i < numBlockC;++i)
	{
		for (int j = 0; j < numBlockR;++j)
		{
			int p = i*numBlockR + j;
			Rect blockroi(j*blockSize.width, i*blockSize.height, blockSize.width, blockSize.height);

			posblock = ltppos(blockroi);
			negblock = ltpneg(blockroi);

			compute_histblock(posblock, negblock, &features[p*featurelenblock]);
		}
	}
}

void LTDP::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
{
	if (groupThreshold <= 0 || rectList.empty()||rectList.size()==1)
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

void LTDP::compute_Ltpimg(const cv::Mat& src, cv::Mat& uniformp, cv::Mat& uniformn) const
{
	Mat ltppos, ltpneg;
	compute_Ltpvalue(src, ltppos, ltpneg);

	LUT(ltppos, lookUpTable, uniformp);
	LUT(ltpneg, lookUpTable, uniformn);
}

void LTDP::normalizeBlockHistogram(float * blockhist) const
{
	float* hist = &blockhist[0];
	size_t i, sz = 118;

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

void LTDP::compute(const cv::Mat & img, vector<float>& features) const
{
	Mat ltppos, ltpneg;
	compute_Ltpvalue(img, ltppos, ltpneg);
	
	Mat uniformltpos, uniformltpneg;
	LUT(ltppos, lookUpTable, uniformltpos);
	LUT(ltpneg, lookUpTable, uniformltpneg);
	//imshow("test", uniformltpos);
	//waitKey();
	compute_histwin(uniformltpos, uniformltpneg, features);
}

void LTDP::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	ltdpsvm = _svm;
}

void LTDP::loadSvmDetector(const string& xmlfile)
{
	ltdpsvm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void LTDP::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, 
	vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/,
	const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	foundlocations.clear();
	if (ltdpsvm->empty())
	{
		cerr << "no svm" << endl;
		return;
	}

	if (winStride == Size())
	{
		winStride = Size(8,8);
	}

	//图像填充
	Size stride(gcd(winStride.width, blockSize.width), gcd(winStride.height, blockSize.height));

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

	//计算整张图的LTDP模式，加速检测过程
	Mat uniformpos, uniformneg;
	compute_Ltpimg(paddedimg, uniformpos, uniformneg);
	int numBlockcols = paddedimg.cols / blockSize.width;
	int numBlockrows = paddedimg.rows / blockSize.height;

	Mat winimg;
	Mat uniformp, uniformn;

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

			winimg = paddedimg(Rect(pt.x, pt.y, winSize.width, winSize.height));

			vector<float> feature;
			compute(winimg, feature);

			float response = ltdpsvm->predict(feature);
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
		for (int j = 0; j < numwinC; ++j)
		{
			for (int i = 0; i < numwinR; ++i)
			{
				Point pt0;
				Rect rt = Rect(i*winStride.width, j*winStride.height, winSize.width, winSize.height);
				winimg = paddedimg(rt);

				pt0.x = i*winStride.width - padding.width;
				pt0.y = j*winStride.height - padding.height;

				vector<float> feature;
				uniformp = uniformpos(rt);
				uniformn = uniformneg(rt);
				compute_histwin(uniformp, uniformn, feature);

				//compute(winimg, feature);
				Mat result;
				float response = ltdpsvm->predict(feature, result, ml::StatModel::RAW_OUTPUT);
				response = result.at<float>(0, 0);
			
				if (response <= -hitThreshold)
				{
					foundlocations.push_back(pt0);
					weights.push_back(-response);
				}
			}
		}
	}
}

void LTDP::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}

class Parallel_Detection_LTDP :public ParallelLoopBody
{
private:
	const LTDP* ltdp;
	Mat img;
	double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection_LTDP(const LTDP* _ltdp, const Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights = 0, vector<double>* _scales = 0)
	{
		ltdp = _ltdp;
		img = _img;
		hitThreshold = _hitThreshold;
		winStride = _winStride;
		levelScale = _levelScale;
		mtx = _mtx;
		vec = _vec;
		weights = _weights;
		scales = _scales;
		//cout << "fk" << endl;
	}

	void operator() (const Range& range) const
	{
		int i, i1 = range.start, i2 = range.end;

		double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1 + 1] : std::max(img.cols, img.rows);
		Size maxSz(cvCeil(img.cols / minScale), cvCeil(img.rows / minScale));
		Mat smallerImgBuf(maxSz, img.type());
		vector<Point> locations;
		vector<double> hitsWeights;
		//cout << "LTDP" << endl;
		for (i = i1; i < i2; i++)
		{
			double scale = levelScale[i];
			Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
			Mat smallerImg(sz, img.type(), smallerImgBuf.data);
			if (sz == img.size())
				smallerImg = Mat(sz, img.type(), img.data, img.step);
			else
				resize(img, smallerImg, sz, 0.0, 0.0, INTER_NEAREST);
			ltdp->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(ltdp->winSize.width*scale), cvRound(ltdp->winSize.height*scale));
			//cout << scaledWinSize << endl;
			mtx->lock();

			for (size_t j = 0; j < locations.size(); j++)
			{
				vec->push_back(Rect(cvRound(locations[j].x*scale),
					cvRound(locations[j].y*scale),
					scaledWinSize.width, scaledWinSize.height));
				if (scales)
				{
					scales->push_back(scale);
				}
			}
			mtx->unlock();

			//vec->clear();
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

void LTDP::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	if (ltdpsvm->empty())
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

	vector<double> foundScales;

	weights.clear();
	foundlocations.clear();

	Mutex mtx;
	parallel_for_(Range(0, levelScale.size()),
		Parallel_Detection_LTDP(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));


	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		//groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void LTDP::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride, nlevels, scale0, finalThreshold, usemeanshift);
}
