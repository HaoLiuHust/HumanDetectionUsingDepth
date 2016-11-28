#include "HDD.h"
#include <iterator>
using namespace cv;



HDDCache::HDDCache(const HDD* _descriptor,
	const Mat& _img, Size _paddingTL, Size _paddingBR,
	bool _useCache, Size _cacheStride)
{
	init(_descriptor, _img, _paddingTL, _paddingBR, _useCache, _cacheStride);
}

HDDCache::HDDCache()
{
	useCache = false;
	blockHistogramSize = count1 = count2 = count4 = 0;
	descriptor = 0;
}

void HDDCache::init(const HDD* _descriptor,
	const Mat& _img, Size _paddingTL, Size _paddingBR,
	bool _useCache, Size _cacheStride)
{
	descriptor = _descriptor;
	cacheStride = _cacheStride;
	useCache = _useCache;

	descriptor->computeGradient(_img, grad, qangle, _paddingTL, _paddingBR);
	imgoffset = _paddingTL;

	winSize = descriptor->winSize;
	Size blockSize = descriptor->blockSize;
	Size blockStride = descriptor->blockStride;
	Size cellSize = descriptor->cellSize;
	int i, j, nbins = descriptor->nbins;
	int rawBlockSize = blockSize.width*blockSize.height;

	nblocks = Size((winSize.width - blockSize.width) / blockStride.width + 1,
		(winSize.height - blockSize.height) / blockStride.height + 1);
	ncells = Size(blockSize.width / cellSize.width, blockSize.height / cellSize.height);
	blockHistogramSize = ncells.width*ncells.height*nbins;

	if (useCache)
	{
		Size cacheSize((grad.cols - blockSize.width) / cacheStride.width + 1,
			(winSize.height / cacheStride.height) + 1);
		blockCache.create(cacheSize.height, cacheSize.width*blockHistogramSize);
		blockCacheFlags.create(cacheSize);
		size_t cacheRows = blockCache.rows;
		ymaxCached.resize(cacheRows);
		for (size_t ii = 0; ii < cacheRows; ii++)
			ymaxCached[ii] = -1;
	}

	Mat_<float> weights(blockSize);
	float sigma = (float)descriptor->getWinSigma();
	float scale = 1.f / (sigma*sigma * 2);

	for (i = 0; i < blockSize.height; i++)
		for (j = 0; j < blockSize.width; j++)
		{
			float di = i - blockSize.height*0.5f;
			float dj = j - blockSize.width*0.5f;
			weights(i, j) = std::exp(-(di*di + dj*dj)*scale);
		}

	blockData.resize(nblocks.width*nblocks.height);
	pixData.resize(rawBlockSize*3);

	// Initialize 2 lookup tables, pixData & blockData.
	// Here is why:
	//
	// The detection algorithm runs in 4 nested loops (at each pyramid layer):
	//  loop over the windows within the input image
	//    loop over the blocks within each window
	//      loop over the cells within each block
	//        loop over the pixels in each cell
	//
	// As each of the loops runs over a 2-dimensional array,
	// we could get 8(!) nested loops in total, which is very-very slow.
	//
	// To speed the things up, we do the following:
	//   1. loop over windows is unrolled in the hddDescriptor::{compute|detect} methods;
	//         inside we compute the current search window using getWindow() method.
	//         Yes, it involves some overhead (function call + couple of divisions),
	//         but it's tiny in fact.
	//   2. loop over the blocks is also unrolled. Inside we use pre-computed blockData[j]
	//         to set up gradient and histogram pointers.
	//   3. loops over cells and pixels in each cell are merged
	//       (since there is no overlap between cells, each pixel in the block is processed once)
	//      and also unrolled. Inside we use PixData[k] to access the gradient values and
	//      update the histogram
	//
	count1 = count2 = count4 = 0;
	for (j = 0; j < blockSize.width; j++)
		for (i = 0; i < blockSize.height; i++)
		{
			PixData* data = 0;
			float cellX = (j + 0.5f) / cellSize.width - 0.5f;
			float cellY = (i + 0.5f) / cellSize.height - 0.5f;
			int icellX0 = cvFloor(cellX);
			int icellY0 = cvFloor(cellY);
			int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
			cellX -= icellX0;
			cellY -= icellY0;

			/*data = &pixData[count1++];
			if ((unsigned)icellY0 < (unsigned)ncells.height)
			{
				icellY1 = icellY0;
				cellY = 1.f - cellY;
			}
			data->histOfs[0] = (icellX1*ncells.height + icellY1)*nbins;
			data->histWeights[0] = cellX*cellY;
			data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
			data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;*/

			if ((unsigned)icellX0 < (unsigned)ncells.width &&
				(unsigned)icellX1 < (unsigned)ncells.width)
			{
				if ((unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height)
				{
					data = &pixData[rawBlockSize * 2 + (count4++)];
					data->histOfs[0] = (icellX0*ncells.height + icellY0)*nbins;
					data->histWeights[0] = (1.f - cellX)*(1.f - cellY);
					data->histOfs[1] = (icellX1*ncells.height + icellY0)*nbins;
					data->histWeights[1] = cellX*(1.f - cellY);
					data->histOfs[2] = (icellX0*ncells.height + icellY1)*nbins;
					data->histWeights[2] = (1.f - cellX)*cellY;
					data->histOfs[3] = (icellX1*ncells.height + icellY1)*nbins;
					data->histWeights[3] = cellX*cellY;
				}
				else
				{
					data = &pixData[rawBlockSize + (count2++)];
					if ((unsigned)icellY0 < (unsigned)ncells.height)
					{
						icellY1 = icellY0;
						cellY = 1.f - cellY;
					}
					data->histOfs[0] = (icellX0*ncells.height + icellY1)*nbins;
					data->histWeights[0] = (1.f - cellX)*cellY;
					data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
					data->histWeights[1] = cellX*cellY;
					data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[2] = data->histWeights[3] = 0;
				}
			}
			else
			{
				if ((unsigned)icellX0 < (unsigned)ncells.width)
				{
					icellX1 = icellX0;
					cellX = 1.f - cellX;
				}

				if ((unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height)
				{
					data = &pixData[rawBlockSize + (count2++)];
					data->histOfs[0] = (icellX1*ncells.height + icellY0)*nbins;
					data->histWeights[0] = cellX*(1.f - cellY);
					data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
					data->histWeights[1] = cellX*cellY;
					data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[2] = data->histWeights[3] = 0;
				}
				else
				{
					data = &pixData[count1++];
					if ((unsigned)icellY0 < (unsigned)ncells.height)
					{
						icellY1 = icellY0;
						cellY = 1.f - cellY;
					}
					data->histOfs[0] = (icellX1*ncells.height + icellY1)*nbins;
					data->histWeights[0] = cellX*cellY;
					data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
				}
			}
			data->gradOfs = (grad.cols*i + j) * 2;
			data->qangleOfs = (qangle.cols*i + j) * 2;
			data->gradWeight = weights(i, j);
		}

	assert(count1 + count2 + count4 == rawBlockSize);
	// defragment pixData
	for (j = 0; j < count2; j++)
		pixData[j + count1] = pixData[j + rawBlockSize];
	for (j = 0; j < count4; j++)
		pixData[j + count1 + count2] = pixData[j + rawBlockSize * 2];
	count2 += count1;
	count4 += count2;

	// initialize blockData
	for (j = 0; j < nblocks.width; j++)
		for (i = 0; i < nblocks.height; i++)
		{
			BlockData& data = blockData[j*nblocks.height + i];
			data.histOfs = (j*nblocks.height + i)*blockHistogramSize;
			data.imgOffset = Point(j*blockStride.width, i*blockStride.height);
		}
}

const float* HDDCache::getBlock(Point pt, float* buf)
{
	float* blockHist = buf;
	assert(descriptor != 0);

	Size blockSize = descriptor->blockSize;
	pt += imgoffset;

	CV_Assert((unsigned)pt.x <= (unsigned)(grad.cols - blockSize.width) &&
		(unsigned)pt.y <= (unsigned)(grad.rows - blockSize.height));

	if (useCache)
	{
		CV_Assert(pt.x % cacheStride.width == 0 &&
			pt.y % cacheStride.height == 0);
		Point cacheIdx(pt.x / cacheStride.width,
			(pt.y / cacheStride.height) % blockCache.rows);
		if (pt.y != ymaxCached[cacheIdx.y])
		{
			Mat_<uchar> cacheRow = blockCacheFlags.row(cacheIdx.y);
			cacheRow = (uchar)0;
			ymaxCached[cacheIdx.y] = pt.y;
		}

		blockHist = &blockCache[cacheIdx.y][cacheIdx.x*blockHistogramSize];
		uchar& computedFlag = blockCacheFlags(cacheIdx.y, cacheIdx.x);
		if (computedFlag != 0)
			return blockHist;
		computedFlag = (uchar)1; // set it at once, before actual computing
	}

	int k, C1 = count1, C2 = count2, C4 = count4;
	const float* gradPtr = (const float*)(grad.data + grad.step*pt.y) + pt.x * 2;
	const uchar* qanglePtr = qangle.data + qangle.step*pt.y + pt.x * 2;

	CV_Assert(blockHist != 0);
	for (k = 0; k < blockHistogramSize; k++)
		blockHist[k] = 0.f;


	const PixData* _pixData = &pixData[0];

	for (k = 0; k < C1; k++)
	{
		const PixData& pk = _pixData[k];
		const float* a = gradPtr + pk.gradOfs;
		float w = pk.gradWeight*pk.histWeights[0];
		const uchar* h = qanglePtr + pk.qangleOfs;
		int h0 = h[0], h1 = h[1];
		float* hist = blockHist + pk.histOfs[0];
		float t0 = hist[h0] + a[0] * w;
		float t1 = hist[h1] + a[1] * w;
		hist[h0] = t0; hist[h1] = t1;
	}

	for (; k < C2; k++)
	{
		const PixData& pk = _pixData[k];
		const float* a = gradPtr + pk.gradOfs;
		float w, t0, t1, a0 = a[0], a1 = a[1];
		const uchar* h = qanglePtr + pk.qangleOfs;
		int h0 = h[0], h1 = h[1];

		float* hist = blockHist + pk.histOfs[0];
		w = pk.gradWeight*pk.histWeights[0];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;

		hist = blockHist + pk.histOfs[1];
		w = pk.gradWeight*pk.histWeights[1];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;
	}

	for (; k < C4; k++)
	{
		const PixData& pk = _pixData[k];
		const float* a = gradPtr + pk.gradOfs;
		float w, t0, t1, a0 = a[0], a1 = a[1];
		const uchar* h = qanglePtr + pk.qangleOfs;
		int h0 = h[0], h1 = h[1];

		float* hist = blockHist + pk.histOfs[0];
		w = pk.gradWeight*pk.histWeights[0];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;

		hist = blockHist + pk.histOfs[1];
		w = pk.gradWeight*pk.histWeights[1];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;

		hist = blockHist + pk.histOfs[2];
		w = pk.gradWeight*pk.histWeights[2];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;

		hist = blockHist + pk.histOfs[3];
		w = pk.gradWeight*pk.histWeights[3];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;
	}

	normalizeBlockHistogram(blockHist);

	return blockHist;
}


void HDDCache::normalizeBlockHistogram(float* _hist) const
{
	float* hist = &_hist[0];
	size_t i, sz = blockHistogramSize;

	float sum = 0;
	for (i = 0; i < sz; i++)
		sum += hist[i] * hist[i];

	float scale = 1.f / (std::sqrt(sum) + sz*0.1f), thresh = (float)descriptor->L2HysThreshold;

	for (i = 0, sum = 0; i < sz; i++)
	{
		hist[i] = std::min(hist[i] * scale, thresh);
		sum += hist[i] * hist[i];
	}

	scale = 1.f / (std::sqrt(sum) + 1e-3f);

	for (i = 0; i < sz; i++)
		hist[i] *= scale;
}


Size HDDCache::windowsInImage(Size imageSize, Size winStride) const
{
	return Size((imageSize.width - winSize.width) / winStride.width + 1,
		(imageSize.height - winSize.height) / winStride.height + 1);
}

Rect HDDCache::getWindow(Size imageSize, Size winStride, int idx) const
{
	int nwindowsX = (imageSize.width - winSize.width) / winStride.width + 1;
	int y = idx / nwindowsX;
	int x = idx - nwindowsX*y;
	return Rect(x*winStride.width, y*winStride.height, winSize.width, winSize.height);
}


void HDD::setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm)
{
	hddsvm = _svm;
}

void HDD::loadSvmDetector(const string& xmlfile)
{
	hddsvm = ml::StatModel::load <ml::SVM>(xmlfile);
}

void HDD::compute(const cv::Mat& img, vector<float>& features) const
{

	Size winStride = cellSize;
	Size cacheStride(gcd(winStride.width, blockStride.width),
		gcd(winStride.height, blockStride.height));
	Size padding;
	padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
	padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
	Size paddedImgSize(img.cols + padding.width * 2, img.rows + padding.height * 2);

	HDDCache cache(this, img, padding, padding, true, cacheStride);

	size_t nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

	const HDDCache::BlockData* blockData = &cache.blockData[0];

	int nblocks = cache.nblocks.area();
	int blockHistogramSize = cache.blockHistogramSize;
	size_t dsize = getFeatureLen();
	features.resize(dsize*nwindows);

	for (size_t i = 0; i < nwindows; i++)
	{
		float* descriptor = &features[i*dsize];

		Point pt0;

		pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
		CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
		for (int j = 0; j < nblocks; j++)
		{
			const HDDCache::BlockData& bj = blockData[j];
			Point pt = pt0 + bj.imgOffset;

			float* dst = descriptor + bj.histOfs;
			const float* src = cache.getBlock(pt, dst);
			if (src != dst)
				for (int k = 0; k < blockHistogramSize; k++)
					dst[k] = src[k];
		}
	}
}

void HDD::detect(const cv::Mat& img, vector<cv::Point>& foundlocations, 
	vector<double>& weights, double hitThreshold /*= 0*/, Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	foundlocations.clear();
	if (hddsvm->empty())
		return;

	if (winStride == Size())
		winStride = cellSize;
	Size cacheStride(gcd(winStride.width, blockStride.width),
		gcd(winStride.height, blockStride.height));
	size_t nwindows = locations.size();
	Size padding;
	padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
	padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
	Size paddedImgSize(img.cols + padding.width * 2, img.rows + padding.height * 2);

	HDDCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

	if (!nwindows)
		nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

	const HDDCache::BlockData* blockData = &cache.blockData[0];

	int nblocks = cache.nblocks.area();
	int blockHistogramSize = cache.blockHistogramSize;
	size_t dsize = getFeatureLen();

	vector<float> winHist(blockHistogramSize*nblocks);
	vector<float> blockHist(blockHistogramSize);
	for (size_t i = 0; i < nwindows; i++)
	{
		Point pt0;
		if (!locations.empty())
		{
			pt0 = locations[i];
			if (pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
				pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height)
				continue;
		}
		else
		{
			pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
			CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
		}
		
		int j, k;

		for (j = 0; j < nblocks; j++)
		{
			const HDDCache::BlockData& bj = blockData[j];
			Point pt = pt0 + bj.imgOffset;

			const float* vec=cache.getBlock(pt, &blockHist[0]);
			memcpy(&winHist[blockHistogramSize*j], vec, sizeof(float)*blockHistogramSize);
		}
		Mat resultmat;
		float response = hddsvm->predict(winHist,resultmat, ml::StatModel::RAW_OUTPUT);
		//cout << resultmat << endl;

		response = resultmat.at<float>(0, 0);

		//float test = hddsvm->predict(winHist);
		if (response <=-hitThreshold)
		{
			foundlocations.push_back(pt0);
			weights.push_back(-response);
		}		
	}
}

void HDD::detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, const vector<cv::Point>& locations /*= vector<cv::Point>()*/) const
{
	vector<double> weights;
	detect(img, foundLocations, weights, hitThreshold, winStride, locations);
}

class HDDInvoker : public ParallelLoopBody
{
public:
	HDDInvoker(const HDD* _hdd, const Mat& _img,
		double _hitThreshold, Size _winStride, 
		const double* _levelScale, std::vector<Rect> * _vec, Mutex* _mtx,
		std::vector<double>* _weights = 0, std::vector<double>* _scales = 0)
	{
		hdd = _hdd;
		img = _img;
		hitThreshold = _hitThreshold;
		winStride = _winStride;
		levelScale = _levelScale;
		vec = _vec;
		weights = _weights;
		scales = _scales;
		mtx = _mtx;
	}

	void operator()(const Range& range) const
	{
		int i, i1 = range.start, i2 = range.end;
		double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1 + 1] : std::max(img.cols, img.rows);
		Size maxSz(cvCeil(img.cols / minScale), cvCeil(img.rows / minScale));
		Mat smallerImgBuf(maxSz, img.type());
		vector<Point> locations;
		vector<double> hitsWeights;
		//cout << "hdd" << endl;
		for (i = i1; i < i2; i++)
		{
			double scale = levelScale[i];
			Size sz(cvRound(img.cols / scale), cvRound(img.rows / scale));
			Mat smallerImg(sz, img.type(), smallerImgBuf.data);
			if (sz == img.size())
				smallerImg = Mat(sz, img.type(), img.data, img.step);
			else
				resize(img, smallerImg, sz,INTER_NEAREST);
			hdd->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride);
			Size scaledWinSize = Size(cvRound(hdd->winSize.width*scale), cvRound(hdd->winSize.height*scale));

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

	const HDD* hdd;
	Mat img;
	double hitThreshold;
	Size winStride;
	Size padding;
	const double* levelScale;
	std::vector<Rect>* vec;
	std::vector<double>* weights;
	std::vector<double>* scales;
	Mutex* mtx;
};

void HDD::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.1*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	double scale = 1.;
	int levels = 0;
	scale0 = 1.1;
	vector<double> levelScale;
	for (levels = 0; levels < nlevels; levels++)
	{
		levelScale.push_back(scale);
		if (cvRound(img.cols / scale) < winSize.width ||
			cvRound(img.rows / scale) < winSize.height ||
			scale0 <= 1)
			break;
		scale *= scale0;
	}
	levels = std::max(levels, 1);
	levelScale.resize(levels);

	std::vector<double> foundScales;
	foundlocations.clear();
	weights.clear();
	Mutex mtx;

	parallel_for_(Range(0, (int)levelScale.size()),
		HDDInvoker(this, img, hitThreshold, winStride, &levelScale[0], &foundlocations, &mtx, &weights, &foundScales));

	if (usemeanshift)
	{
		groupRectangles_meanshift(foundlocations, weights, foundScales, finalThreshold, winSize);
	}
	else
	{
		groupRectangles(foundlocations, weights, (int)finalThreshold, 0.2);
	}
}

void HDD::detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold /*= 0*/, cv::Size winStride /*= cv::Size()*/, double nlevels /*= 64*/, double scale0 /*= 1.05*/, double finalThreshold /*= 2.0*/, bool usemeanshift /*= false*/) const
{
	vector<double> weights;
	detectMultiScale(img, foundlocations, weights, hitThreshold, winStride, nlevels, scale0, finalThreshold, usemeanshift);
}

double HDD::getWinSigma() const
{
	return winSigma >= 0 ? winSigma : (blockSize.width + blockSize.height) / 8.;
}

void HDD::copyTo(HDD& c) const
{
	c.winSize = winSize;
	c.blockSize = blockSize;
	c.blockStride = blockStride;
	c.cellSize = cellSize;
	c.nbins = nbins;
	c.derivAperture = derivAperture;
	c.winSigma = winSigma;
	c.histogramNormType = histogramNormType;
	c.L2HysThreshold = L2HysThreshold;
	c.hddsvm = hddsvm;
}

void HDD::cal_parms()
{
	CV_Assert(blockSize.width % cellSize.width == 0 &&
		blockSize.height % cellSize.height == 0);
	CV_Assert((winSize.width - blockSize.width) % blockStride.width == 0 &&
		(winSize.height - blockSize.height) % blockStride.height == 0);
	featurenlen=nbins*(blockSize.width / cellSize.width)*
		(blockSize.height / cellSize.height)*
		((winSize.width - blockSize.width) / blockStride.width + 1)*
		((winSize.height - blockSize.height) / blockStride.height + 1);

	maskx = (Mat_<float>(3, 3) << 0, 0, 0, -0.5, 0, 0.5, 0, 0, 0);
	masky = (Mat_<float>(3, 3) << 0, -0.5, 0, 0, 0, 0, 0, 0.5, 0);

}

void HDD::groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const
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

void HDD::computeGradient(const Mat& img, Mat& grad, Mat& angleOfs, Size paddingTL /*= Size()*/, Size paddingBR /*= Size()*/) const
{
	Size gradsize(img.cols + paddingTL.width + paddingBR.width,
		img.rows + paddingTL.height + paddingBR.height);
	grad.create(gradsize, CV_32FC2);
	angleOfs.create(gradsize, CV_8UC2);

	Mat imgPadded;
	if (gradsize != img.size())
	{
		copyMakeBorder(img,imgPadded,paddingTL.height,paddingBR.height,paddingTL.width,paddingBR.width,BORDER_REFLECT101);
	}
	else
		imgPadded = img;
	Mat Dx, Dy;
	filter2D(imgPadded, Dx, CV_32FC1, maskx);
	filter2D(imgPadded, Dy, CV_32FC1, masky);

	//Dx.setTo(0, imgPadded == 0);
	//Dy.setTo(0, imgPadded == 0);

	int _nbins = nbins;
	float angleScale = (float)(_nbins / (2 * CV_PI));
	Mat angle, mag;
	cartToPolar(Dx, Dy, mag, angle);
	/*cout << angle << endl;
	imshow("fx", angle);
	waitKey();*/
	for (int y = 0; y < gradsize.height;++y)
	{
		float* gradPtr = (float*)grad.ptr(y);
		uchar* qanglePtr = (uchar*)angleOfs.ptr(y);
		float* magPtr = mag.ptr<float>(y);
		float* anglePtr = angle.ptr<float>(y);

		for (int x = 0; x < gradsize.width;++x)
		{
			float mag = magPtr[x];
			float angle = anglePtr[x] * angleScale - 0.5f;
			int hidx = cvFloor(angle);
			angle -= hidx;
			gradPtr[2*x] = mag*(1.f - angle);
			gradPtr[2*x+1] = mag*angle;

			if (hidx < 0)
				hidx += _nbins;
			else if (hidx >= _nbins)
				hidx -= _nbins;

			CV_Assert((unsigned)hidx < (unsigned)_nbins);
			qanglePtr[x * 2] = (uchar)hidx;
			++hidx;
			hidx &= hidx < _nbins ? -1 : 0;
			qanglePtr[x * 2 + 1] = (uchar)hidx;
		}

	}

	//Size wholeSize;
	//Point roiofs;
	//img.locateROI(wholeSize, roiofs);

	//int i, x, y;
	//int cn = img.channels();
	//Mat_<float> _lut;
	//if (img.type() == CV_8U)
	//{
	//	_lut.create(1, 256);
	//	for (i = 0; i < 256; ++i)
	//	{
	//		_lut(0, i) = (float)i;
	//	}
	//}
	//else
	//{
	//	_lut.create(1, 10001);
	//	for (i = 0; i < 10001; ++i)
	//	{
	//		_lut(0, i) = (float)i;
	//	}
	//}

	//AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
	//int* xmap = (int*)mapbuf + 1;
	//int* ymap = xmap + gradsize.width + 2;

	//const int borderType = (int)BORDER_REFLECT101;
	//const float* lut = &_lut(0, 0);
	//for (x = -1; x < gradsize.width + 1; ++x)
	//{
	//	xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
	//		wholeSize.width, borderType) - roiofs.x;
	//}
	//for (y = -1; y < gradsize.width + 1; ++y)
	//{
	//	ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
	//		wholeSize.height, borderType) - roiofs.y;
	//}

	////// x- & y- derivatives for the whole row
	//int width = gradsize.width;
	//vector<float> _dbuf(width * 4,0);
	////AutoBuffer<float> _dbuf(width * 4);
	//float* dbuf =&_dbuf[0];
	//Mat Dx(1, width, CV_32F, dbuf);
	//Mat Dy(1, width, CV_32F, dbuf + width);
	//Mat Mag(1, width, CV_32F, dbuf + width * 2);
	//Mat Angle(1, width, CV_32F, dbuf + width * 3);

	//

	//for (y = 0; y < gradsize.height; ++y)
	//{
	//	float* gradPtr = (float*)grad.ptr(y);
	//	uchar* qanglePtr = (uchar*)angleOfs.ptr(y);

	//	if (img.type() == CV_8U)
	//	{
	//		const uchar* imgPtr = img.data + img.step*ymap[y];
	//		const uchar* prevPtr = img.data + img.step*ymap[y - 1];
	//		const uchar* nextPtr = img.data + img.step*ymap[y + 1];

	//		for (x = 0; x < width; ++x)
	//		{
	//			int x1 = xmap[x];
	//			dbuf[x] = (float)(lut[imgPtr[xmap[x + 1]]] - lut[imgPtr[xmap[x - 1]]]) / 2;
	//			dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]) / 2;
	//		}
	//	}
	//	else
	//	{
	//		const ushort* imgPtr = (ushort*)img.data + img.step / img.elemSize()*ymap[y];
	//		const ushort* prevPtr = (ushort*)img.data + img.step / img.elemSize()*ymap[y - 1];
	//		const ushort* nextPtr = (ushort*)img.data + img.step / img.elemSize()*ymap[y + 1];
	//		cout << img.step << endl;
	//		for (x = 0; x < width; ++x)
	//		{
	//			int x1 = xmap[x];
	//			dbuf[x] = (float)(lut[imgPtr[xmap[x + 1]]] - lut[imgPtr[xmap[x - 1]]]) / 2;
	//			dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]) / 2;
	//		}
	//	}

	//	//0~2pi
	//	cartToPolar(Dx, Dy, Mag, Angle, false);

	//	for (x = 0; x < width; ++x)
	//	{
	//		float mag = dbuf[x + width * 2];
	//		float angle = dbuf[x + width * 3] * angleScale - 0.5f;
	//		int hidx = cvFloor(angle);
	//		angle -= hidx;
	//		gradPtr[x * 2] = mag*(1.f - angle);
	//		gradPtr[x * 2 + 1] = mag*angle;

	//		if (hidx < 0)
	//			hidx += _nbins;
	//		else if (hidx >= _nbins)
	//			hidx -= _nbins;

	//		CV_Assert((unsigned)hidx < (unsigned)_nbins);
	//		qanglePtr[x * 2] = (uchar)hidx;
	//		++hidx;
	//		hidx &= hidx < _nbins ? -1 : 0;
	//		qanglePtr[x * 2 + 1] = (uchar)hidx;
	//	}
	//}
	//_dbuf.clear();
}

