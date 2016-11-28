#ifndef ALGORITHM_H
#define ALGORITHM_H
#include "opencvHeader.h"
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <string>
using namespace std;
class DetectionAlgorithm
{
public:
	virtual void compute(const cv::Mat& img, vector<float>& feature)const=0;
	virtual int getFeatureLen()const = 0;
	virtual void setSvmDetector(const cv::Ptr<cv::ml::SVM>& _svm) = 0;
	//virtual void setSvmDetector(const string& xmlfile)= 0;
	virtual void loadSvmDetector(const string& xmlfile) = 0;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, vector<double>& weights, double hitThreshold = 0,
		cv::Size winStride = cv::Size(), const vector<cv::Point>& locations = vector<cv::Point>()) const=0;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, vector<double>& weights, double hitThreshold = 0,
		cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const=0;
	virtual void detect(const cv::Mat& img, vector<cv::Point>& foundLocations, double hitThreshold = 0, cv::Size winStride = cv::Size(), 
		const vector<cv::Point>& locations = vector<cv::Point>()) const = 0;
	virtual void detectMultiScale(const cv::Mat& img, vector<cv::Rect>& foundlocations, double hitThreshold = 0, 
		cv::Size winStride = cv::Size(), double nlevels = 64, double scale0 = 1.05, double finalThreshold = 2.0, bool usemeanshift = false)const = 0;
	//virtual void setHitThreshold(const double _hitThreshold)=0;
	virtual void set_signThreshold(const int _signThreshold){}

	virtual ~DetectionAlgorithm(){}
};
#endif
