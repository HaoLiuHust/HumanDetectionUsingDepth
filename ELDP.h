#ifndef ELDP_H
#define ELDP_H
#include "opencvHeader.h"
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/ml.hpp>
#include "Algorithm.hpp"
using namespace std;

class ELDP :public DetectionAlgorithm
{
public:
	ELDP() :featurelenblock(56) { setDefaultParams(); cal_params(); }
	ELDP(cv::Size _winSize, cv::Size _blockSize) :featurelenblock(56), winSize(_winSize), blockSize(_blockSize)
	{
		cal_params();
	}
	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm);
	virtual void loadSvmDetector(const string& xmlfile);
	virtual int getFeatureLen()const { return featurelen; }

	virtual void compute(const cv::Mat& img, vector<float>& features)const;//compute a windows feature;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundlocations, vector<double>& weights, double hitThreshold = 0,
		cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>())const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights,
		double hitThreshold = 0, cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.1, double finalThreshold = 2.0, bool usemeanshift = false)const;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>()) const;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold = 0, cv::Size winStride = cv::Size(),
		double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const;
public:
	cv::Size blockSize;
	cv::Size winSize;
private:
	int numBlockR;
	int numBlockC;
	int featurelen;
	int numBlockPerWin;
	int featurelenblock;
	vector<cv::Mat> masks;

	cv::Mat lookUpTable;
private:
	cv::Ptr<cv::ml::SVM> ldpksvm;
private:
	void setDefaultParams();
	void cal_params();

	void compute_Ltpvalue(const cv::Mat& src, cv::Mat& ltpimg)const;
	void compute_histblock(const cv::Mat& ltpimg, float* feature)const;
	void compute_histwin(const cv::Mat& ltpimg, vector<float>& features)const;
	void compute_histimg(const cv::Mat& ltpimg, vector<float>& features, cv::Size winStride)const;
	void getblockhist(const cv::Mat& blockimg, cv::Point pt, float* blockhist, vector<vector<float> >& imagehist,
		vector<bool> flags, int blockperrow)const;

	void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
	void normalizeBlockHistogram(float* blockhist)const;

};
#endif
