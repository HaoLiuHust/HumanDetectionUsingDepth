#include "Utils.h"
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <io.h>

using namespace cv;

cv::Scalar Utils::randomColor(RNG rng)
{
	int icolor = (unsigned)rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void Utils::writeTocsv(const string& filename, const vector<Point2f>& points)
{
	ofstream outfile(filename,ios::binary);
	for (size_t i = 0; i < points.size();++i)
	{
		outfile << points[i].x << ',' << points[i].y << "\n";
	}
	outfile.close();
}

bool Utils::findallfiles(const string& folderpath, vector<string>& files,string filetype)
{
		_finddata_t fileInfo;
		string strfind = folderpath + "\\*."+filetype;
		//replacepath(strfind);
		intptr_t handle = _findfirst(strfind.c_str(), &fileInfo);

		if (handle == -1)
		{
			cerr << "failded to open folder" << folderpath << endl;
			return false;
		}
		
		files.push_back(fileInfo.name);

		while (_findnext(handle, &fileInfo) == 0)
		{
			if (fileInfo.attrib & _A_SUBDIR)
				continue;

			files.push_back(fileInfo.name);

		}
		_findclose(handle);
		return true;
}

void Utils::NonMaximalSuppression(vector<cv::Rect>& rects, vector<double>& weights, float overlap, int etype)
{

	Utils::NonMaximumSuppression::eliminateRedundantDetections(rects, weights, overlap, etype);
}

void Utils::NonMaximalSuppression2(vector<cv::Rect>& rects, vector<double>& weights, float overlap, int etype /*= 1*/)
{
	Utils::NonMaximumSuppression::eliminateRedundantDetections(rects, weights, overlap, etype);
	vector<Rect> found_filterd;
	vector<double> weight_filterd;
	int i, j;
	for (int i = 0; i < rects.size(); i++)
	{
		Rect r = rects[i];

		for (j = 0; j < rects.size(); j++)
			if (j != i && (r&rects[j]) == r)
				break;
		if (j == rects.size())
		{
			found_filterd.push_back(r);
			weight_filterd.push_back(weights[i]);
		}
	}

	rects.swap(found_filterd);
	weights.swap(weight_filterd);
}

float Utils::NonMaximumSuppression::overlapThreshold = 1.0;
int Utils::NonMaximumSuppression::maximumType = 0;
void Utils::NonMaximumSuppression::eliminateRedundantDetections(vector<cv::Rect>& rects, vector<double>& weights, float overlap, int type)
{
	if (overlap >= 1)
	{
		return;
	}
	overlapThreshold = overlap;
	maximumType = type;
	vector<Detection> candidates(rects.size());
	for (int i = 0; i < rects.size(); ++i)
	{
		candidates[i].bounds = rects[i];
		candidates[i].score = weights[i];
	}
	sortByscores(candidates);
	vector<vector<Detection> > clusters;
	cluster(candidates, clusters);
	getMaxima(clusters, candidates);

	rects.clear();
	weights.clear();
	for (int i = 0; i < candidates.size(); ++i)
	{
		rects.push_back(candidates[i].bounds);
		weights.push_back(candidates[i].score);
	}

}

void Utils::NonMaximumSuppression::sortByscores(vector<Detection>& candidates)
{
	std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
		return a.score < b.score;
	});
}

void Utils::NonMaximumSuppression::cluster(vector<Detection>& candidates, vector<vector<Detection> >&clusters)
{
	clusters.clear();
	while (!candidates.empty())
		clusters.push_back(extractOverlappingDetections(candidates.back(), candidates));
}

vector<Utils::Detection> Utils::NonMaximumSuppression::extractOverlappingDetections(Detection& detection, vector<Detection>& candidates)
{
	vector<Detection> overlappingDetections;
	auto firstOverlapping = std::stable_partition(candidates.begin(), candidates.end(), [&](const Detection& candidate) {
		return computeOverlap(detection.bounds, candidate.bounds) <= overlapThreshold;
	});
	std::move(firstOverlapping, candidates.end(), std::back_inserter(overlappingDetections));
	std::reverse(overlappingDetections.begin(), overlappingDetections.end());
	candidates.erase(firstOverlapping, candidates.end());
	return overlappingDetections;
}

double Utils::NonMaximumSuppression::computeOverlap(cv::Rect a, cv::Rect b)
{
	double intersectionArea = (a & b).area();
	//double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / a.area();
}

void Utils::NonMaximumSuppression::getMaxima(const vector<vector<Detection> >& clusters, vector<Detection>& finalDetections)
{
	finalDetections.clear();
	finalDetections.reserve(clusters.size());
	for (const vector<Detection>& cluster : clusters)
		finalDetections.push_back(getMaximum(cluster));
}

Utils::Detection Utils::NonMaximumSuppression::getMaximum(const vector<Detection>& cluster)
{
	if (maximumType == MaximumType::MAX_SCORE) {
		return cluster.front();
	}
	else if (maximumType == MaximumType::AVERAGE) {
		double xSum = 0;
		double ySum = 0;
		double wSum = 0;
		double hSum = 0;
		for (const Detection& elem : cluster) {
			xSum += elem.bounds.x;
			ySum += elem.bounds.y;
			wSum += elem.bounds.width;
			hSum += elem.bounds.height;
		}
		int x = static_cast<int>(std::round(xSum / cluster.size()));
		int y = static_cast<int>(std::round(ySum / cluster.size()));
		int w = static_cast<int>(std::round(wSum / cluster.size()));
		int h = static_cast<int>(std::round(hSum / cluster.size()));
		float score = cluster.front().score;
		Rect averageBounds(x, y, w, h);
		return Detection{ score, averageBounds };
	}
	else if (maximumType == MaximumType::WEIGHTED_AVERAGE) {
		double weightSum = 0;
		double xSum = 0;
		double ySum = 0;
		double wSum = 0;
		double hSum = 0;
		for (const Detection& elem : cluster) {
			double weight = elem.score;
			weightSum += weight;
			xSum += weight * elem.bounds.x;
			ySum += weight * elem.bounds.y;
			wSum += weight * elem.bounds.width;
			hSum += weight * elem.bounds.height;
		}
		int x = static_cast<int>(std::round(xSum / weightSum));
		int y = static_cast<int>(std::round(ySum / weightSum));
		int w = static_cast<int>(std::round(wSum / weightSum));
		int h = static_cast<int>(std::round(hSum / weightSum));
		float score = cluster.front().score;
		Rect averageBounds(x, y, w, h);
		return Detection{ score, averageBounds };
	}
	else {
		throw std::runtime_error("NonMaximumSuppression: unsupported maximum type");
	}

}