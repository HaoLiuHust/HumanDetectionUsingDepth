#include "LDPK.h"
#include <bitset>
#include <algorithm>
#include <functional>
#include <numeric>
using namespace cv;
using namespace std;

void LDPK::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	ldpksvm = _svm;
}

void LDPK::loadSvmDetector(const string& xmlfile)
{
	ldpksvm = ml::StatModel::load<ml::SVM>(xmlfile);
}

void LDPK::compute(const cv::Mat& img, vector<float>& features) const
{
	Mat ltpimg;
	compute_Ltpvalue(img, ltpimg);
	compute_histwin(ltpimg, features);
}

void LDPK::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	foundlocations.clear();
	if (ldpksvm->empty())
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


	stride = winStride;

	if (winStride.width > blockSize.width || winStride.height > blockSize.height)
	{
		stride = blockSize;
	}
		

	int blockperrow = (paddedImgSize.width - blockSize.width) / stride.width + 1;
	int blockpercol = (paddedImgSize.height - blockSize.height) / stride.height + 1;
	vector<vector<float> > featuresimg(blockperrow*blockpercol);
	vector<bool> cptflags(blockpercol*blockperrow, false);

	Mat patternimg;
	compute_Ltpvalue(paddedimg, patternimg);
	//int numCellcols = paddedimg.cols / cellSize.width;
	//int numCellrows = paddedimg.rows / cellSize.height;
	vector<float> featurewin(featurelen);
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
			float response = ldpksvm->predict(feature);
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
			for (int j = 0; j < numwinC; ++j)
			{
				for (int i = 0; i < numwinR; ++i)
				{
					Point pt0;

					Rect rt = Rect(i*winStride.width, j*winStride.height, winSize.width, winSize.height);
					Mat winbin = patternimg(rt);

					int blockindexrow = (j*winStride.height - padding.height) / stride.height;
					int blockindexcol = (i*winStride.width - padding.width) / stride.width;

					pt0.x = i*winStride.width - padding.width;
					pt0.y = j*winStride.height - padding.height;;

					for (int h = 0, p = 0; h < numBlockC; ++h)
					{
						for (int k = 0; k < numBlockR; ++k, ++p)
						{
							Rect blockrect = Rect(k*blockSize.width, h*blockSize.height, blockSize.width, blockSize.height);
							Mat blockroi = winbin(blockrect);
							Point blockpt = Point(blockindexcol + k, blockindexrow + h);

							getblockhist(blockroi, blockpt, &featurewin[p * featurelenblock], featuresimg, cptflags, blockperrow);
						}
					}

					Mat result;
					//sltpsvm->predict(winimg,cv::displayStatusBar::)
					float response = ldpksvm->predict(featurewin, result, ml::StatModel::RAW_OUTPUT);
					response = result.at<float>(0);

					if (response <= -hitThreshold)
					{
						foundlocations.push_back(pt0);
						weights.push_back(-response);
					}

				}
			}
		}
	}
}

void LDPK::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}

class Parallel_Detection_LDPK :public ParallelLoopBody
{
private:
	const LDPK* ldpk;
	Mat img;
	double hitThreshold;
	Size winStride;
	const double* levelScale;
	Mutex* mtx;
	vector<Rect>* vec;
	vector<double>* weights;
	vector<double>* scales;

public:
	Parallel_Detection_LDPK(const LDPK* _ldpk, const Mat& _img, double _hitThreshold, Size _winStride, const double* _levelScale,
		vector<Rect>* _vec, Mutex* _mtx, vector<double>* _weights = 0, vector<double>* _scales = 0)
	{
		ldpk = _ldpk;
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
				resize(img, smallerImg, sz, 0.0, 0.0, INTER_NEAREST);
			ldpk->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(ldpk->winSize.width*scale), cvRound(ldpk->winSize.height*scale));

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

void LDPK::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	if (ldpksvm->empty())
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
		Parallel_Detection_LDPK(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));

	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		//groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void LDPK::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations,
	double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, 
	bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride, nlevels, scale0, finalThreshold, usemeanshift);
}

void LDPK::setDefaultParams()
{
	winSize = Size(64, 128);
	blockSize = Size(16, 16);
	K = 3;
}

void LDPK::cal_params()
{
	numBlockC = winSize.height / blockSize.height;
	numBlockR = winSize.width / blockSize.width;

	numBlockPerWin = numBlockR*numBlockC;
	featurelen = featurelenblock*numBlockPerWin;

	masks.resize(8);
	masks[0] = (Mat_<char>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
	masks[1] = (Mat_<char>(3, 3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);
	masks[2] = (Mat_<char>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
	masks[3] = (Mat_<char>(3, 3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);
	masks[4] = (Mat_<char>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);
	masks[5] = (Mat_<char>(3, 3) << -3, -3, -3, 5, 0, -3, 5, 5, -3);
	masks[6] = (Mat_<char>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);
	masks[7] = (Mat_<char>(3, 3) << -3, -3, -3, -3, 0, 5, -3, 5, 5);
	
	bitset<8> b;
	lookUpTable=Mat::zeros(1, 256, CV_8U);

	
	//待完善，K不为3时需要从新计算
	for (int i = 7,p=0; i >=2 ;--i)
	{
		for (int j = i - 1; j >= 1;--j)
		{
			for (int k = j - 1; k >= 0;--k,++p)
			{
				b.reset();
				b.set(i);
				b.set(j);
				b.set(k);

				uchar v = (uchar)b.to_ulong();
				lookUpTable.at<uchar>(v) = p;
			}
		}
	}
}

void LDPK::compute_Ltpvalue(const cv::Mat& src, cv::Mat& ltpimg) const
{
	ltpimg = Mat::zeros(src.size(), CV_8UC1);

	Mat dismap[8];
	//#pragma omp parallel for
	for (int i = 0; i < 8; ++i)
	{
		filter2D(src, dismap[i], CV_32FC1, masks[i]);
		//dismap[i] = abs(dismap[i]);
	}
	float* ptr[8];
	uchar* ltpptr;
	for (int i = 0; i < src.rows; ++i)
	{
		for (int k = 0; k < 8; ++k)
		{
			ptr[k] = dismap[k].ptr<float>(i);
		}

		ltpptr = ltpimg.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			float kv[8];
			//vector<float> kv(8);
			for (int k = 0; k < 8; ++k)
			{
				//kv[k] = dismap[k].at<float>(i, j);
				kv[k] = ptr[k][j];
			}

			int sortindex[8] = { 0,1,2,3,4,5,6,7 };

			//插入排序
			for (int k = 1; k < 8; ++k)
			{
				int h = k - 1;
				float temp = kv[k];
				int tempsort = sortindex[k];
				while (h >= 0 && kv[h] < temp)
				{
					kv[h + 1] = kv[h];
					sortindex[h + 1] = sortindex[h];
					--h;
				}
				kv[h + 1] = temp;
				sortindex[h + 1] = tempsort;
			}

			int v[8] = { 0 };
			//vector<int> v(8, 0);
			for (int k = 0; k < K; ++k)
			{
				v[sortindex[k]] = 1;
			}
			//sort(kv.begin(), kv.end(),greater<float>());

			uchar sumv = 0;
			for (int k = 0; k < 8; ++k)
			{
				if (v[k] != 0)
				{
					sumv += ((1 << (7 - k)) * v[k]);
				}
			}
			//ltpimg.at<uchar>(i, j) = sumv;
			ltpptr[j] = sumv;
		}
	}

	LUT(ltpimg, lookUpTable, ltpimg);
	//cout << ltpimg << endl;
}

void LDPK::compute_histblock(const cv::Mat& blockltpimg, float* feature) const
{
	memset(feature, 0, sizeof(float)*featurelenblock);
	for (int i = 0; i < blockltpimg.rows;++i)
	{
		for (int j = 0; j < blockltpimg.cols;++j)
		{
			//cout << (int)blockltpimg.at<uchar>(i, j) << endl;
			++feature[blockltpimg.at<uchar>(i, j)];
		}
	}

	normalizeBlockHistogram(feature);
}

void LDPK::compute_histwin(const cv::Mat& ltpimg, vector<float>& features) const
{
	features.clear();
	features.resize(featurelen, 0);

	Mat ltpblock;
	//#pragma omp parallel for
	for (int i = 0,p=0; i < numBlockC; ++i)
	{
		for (int j = 0; j < numBlockR; ++j,++p)
		{
			//int p = i*numBlockR + j;
			Rect blockroi(j*blockSize.width, i*blockSize.height, blockSize.width, blockSize.height);

			ltpblock = ltpimg(blockroi);
			compute_histblock(ltpblock, &features[p*featurelenblock]);
		}
	}
}

//no-implemention
void LDPK::compute_histimg(const cv::Mat& ltpimg, vector<float>& features,Size winStride) const
{
	Size stride;
	if (winStride.width > blockSize.width||winStride.height>blockSize.height)
	{
		stride = blockSize;
	}
	else
		stride = winStride;
	int nblocksperrow = (ltpimg.cols - blockSize.width) / stride.width + 1;
	int nblockspercol = (ltpimg.rows - blockSize.height) / stride.height + 1;

	features.clear();
	features.resize(nblockspercol*nblocksperrow*featurelenblock);

	for (int i = 0,p=0; i < nblockspercol;++i)
	{
		for (int j = 0; j < nblocksperrow;++j,++p)
		{
			Rect rt = Rect(i*stride.width, j*stride.height, blockSize.width, blockSize.height);
			Mat blockimg = ltpimg(rt);
			compute_histblock(blockimg, &features[p*featurelenblock]);
		}
	}
}

void LDPK::getblockhist(const cv::Mat& blockimg, cv::Point pt, float* blockhist, vector<vector<float> >& imagehist, vector<bool> flags, int blockperrow) const
{
	int index = pt.y*blockperrow + pt.x;
	if (flags[index])
	{
		memcpy(blockhist, &imagehist[index][0], sizeof(float) *featurelenblock);
		return;
	}

	compute_histblock(blockimg, blockhist);
	imagehist[index].resize(featurelenblock);
	memcpy(&imagehist[index][0], blockhist, sizeof(float) * featurelenblock);
	flags[index] = true;
}

void LDPK::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
{
	if (groupThreshold <= 0 || rectList.empty() || rectList.size() == 1)
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

void LDPK::normalizeBlockHistogram(float* blockhist) const
{
	float* hist = &blockhist[0];
	size_t i, sz = featurelenblock;

	//另一种归一化
	/*float sum = std::accumulate(blockhist, blockhist + featurelenblock, 0.0);
	for (int i = 0; i < featurelenblock;++i)
	{
		hist[i] /= sum;
	}*/

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

