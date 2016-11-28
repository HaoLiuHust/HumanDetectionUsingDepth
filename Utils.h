#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <vector>
#include "opencvHeader.h"
using namespace std;


namespace Utils
{
	inline cv::Scalar randomColor(cv::RNG rng);
	void   writeTocsv(const string& filename, const vector<cv::Point2f>& points);
	bool  findallfiles(const string& folderpath, vector<string>& files, string filetype);
	void NonMaximalSuppression(vector<cv::Rect>& rects, vector<double>& weights, float overlap, int etype = 1);

	void NonMaximalSuppression2(vector<cv::Rect>& rects, vector<double>& weights, float overlap, int etype = 1);
	struct Detection {
		float score;
		cv::Rect bounds;
	};
	class NonMaximumSuppression
	{
	public:

		/** Type of the maximum computation per cluster. */
		static enum MaximumType {
			MAX_SCORE, ///< Take the detection with the highest score per cluster.
			AVERAGE, ///< Compute the average of the cluster.
			WEIGHTED_AVERAGE ///< Compute the average of the cluster, weighted by the score.
		};
		static float overlapThreshold;
		static int maximumType;
		static void eliminateRedundantDetections(vector<cv::Rect>& rects, vector<double>& weights, float overlap, int type = MaximumType::MAX_SCORE);
		static void sortByscores(vector<Detection>& candidates);
		static void cluster(vector<Detection>& candidates, vector<vector<Detection> >&clusters);
		static vector<Utils::Detection> extractOverlappingDetections(Detection& detection, vector<Detection>& candidates);
		static double computeOverlap(cv::Rect a, cv::Rect b);
		static void getMaxima(const vector<vector<Detection> >& clusters, vector<Detection>& finalDetections);
		static Detection getMaximum(const vector<Detection>& cluster);
	};
};
#endif
