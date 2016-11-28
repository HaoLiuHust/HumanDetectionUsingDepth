#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "opencvHeader.h"
#include <direct.h>
#include <omp.h>
#include "Algorithm.hpp"
#include "SLTP.h"
#include <algorithm>
#include "LTDP.h"
#include "HDD.h"
#include <fstream>
#include <sstream>
#include "LDPK.h"
#include "ELDP.h"
#include <windows.h>
#include "Utils.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

void testfrommultiDetectors(DetectionAlgorithm* detectors[], string detectornames[], const string modelpath,const string& testpath,
	const string& testnegpath, const string& resultpath, string testtype)
{
	_mkdir(resultpath.c_str());
	vector<string> testfiles;
	Utils::findallfiles(testpath, testfiles,"png");
	sort(testfiles.begin(), testfiles.end());
	cout << "正样本样本数 " << testfiles.size() << endl;

	vector<string> negfiles;
	Utils::findallfiles(testnegpath, negfiles, "png");
	cout << "负样本数 " << negfiles.size() << endl;
	cout << testtype << endl;

	for (int j =2 ; j < 3;++j)
	{
		cout << detectornames[j] << endl;

		for (int k = 0; k < 1;++k)
		{
			string svmpath = modelpath + detectornames[j] + "_" + testtype;
			svmpath += (k == 0 ? "svm1.xml" : "svm2.xml");
			detectors[j]->loadSvmDetector(svmpath);

			string outputpath = resultpath + detectornames[j] + "_" + testtype;
			outputpath += (k == 0 ? "1.txt" : "2.txt");

			//输出结果
			fstream fout(outputpath, ios::out);
			double sumt = 0;
			for (int i = 0; i <testfiles.size(); i++)
			{
				
				//cout << testfiles[i] << endl;
				string fullpath = testpath + testfiles[i];
				Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
				if (sample.empty())
					continue;

				vector<Rect> founds;
				vector<double> weights;

				//double t1 = (double)getTickCount();
				detectors[j]->detectMultiScale(sample, founds, weights, 0.5);
				Utils::NonMaximalSuppression2(founds, weights, 0.5, 0);
				//double t2 = (double)getTickCount();
				//sumt +=(t2-t1);
				Mat imrgb;
				sample.convertTo(imrgb, CV_8U, 255.0 / 8000);
				cvtColor(imrgb, imrgb, CV_GRAY2BGR);
				//imshow("ori", imrgb);
				//Mat imrgb2 = imrgb.clone();

				for (int h = 0; h < founds.size(); ++h)
				{
					Rect r = founds[h];

					if (r.x < 0)
						r.x = 0;
					if (r.y < 0)
						r.y = 0;
					if (r.x + r.width > sample.cols)
						r.width = sample.cols - r.x;
					if (r.y + r.height > sample.rows)
						r.height = sample.rows - r.y;

					fout << testfiles[i] << " " << weights[h] << " "
						<< r.x << " " << r.y
						<< " " << r.width << " " << r.height << endl;
					//if(weights[h]>=0.5)
						rectangle(imrgb, r, Scalar(0, 0, 255), 4);
				}
				imshow("test", imrgb);

				cvWaitKey(10);
			}
			//fout.close();
			//system("pause");
			//cout << sumt/getTickFrequency()<< endl;
			//负样本测试
			for (int i = 0; i < negfiles.size(); ++i)
			{
				//cout << testpath[i] << endl;
				string fullpath = testnegpath + negfiles[i];
				Mat sample = imread(fullpath, IMREAD_ANYDEPTH);
				if (sample.empty())
					continue;

				vector<Rect> founds;
				vector<double> weights;
				detectors[j]->detectMultiScale(sample, founds, weights, 0.);

				for (int h = 0; h < founds.size(); ++h)
				{
					Rect r = founds[h];

					if (r.x < 0)
						r.x = 0;
					if (r.y < 0)
						r.y = 0;
					if (r.x + r.width > sample.cols)
						r.width = sample.cols - r.x;
					if (r.y + r.height > sample.rows)
						r.height = sample.rows - r.y;

						fout << negfiles[i] << " " << weights[h] << " "
							<< r.x << " " << r.y
							<< " " << r.width << " " << r.height << endl;
						//fout << negfiles[i] << " " << weights[h] << endl;
				}
			}

			fout.close();
		}		
	}
}

int main()
{
	string testpath = "F:\\depth10\\";
	string resultspath = "F:\\liuhao\\testTrainSet\\result11\\";
	string modelpath = "F:\\liuhao\\testTrainSet\\models8\\";
	string negpath = "F:\\liuhao\\negtestfull\\";

	//testcca(modelpath, testpath, negpath, resultspath);
	DetectionAlgorithm* detectors[10];


	string testtypes[4] = {"ori","ori","ori","ori"};
	int k[4] = { 0,0,0,0 };
	//detect here

	for (int i = 0; i < 10;++i)
	{
		delete detectors[i];
	}

	std::system("pause");
}
