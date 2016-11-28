#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include "opencvHeader.h"
#include <direct.h>
#include <omp.h>
#include "Algorithm.hpp"
#include "SLTP.h"
#include "LTDP.h"
#include "HDD.h"
#include "LDPK.h"
#include "ELDP.h"
#include <vector>
#include <sstream>
#include "Utils.h"
using namespace std;
using namespace cv;
using namespace cv::ml;
string hardpath;
string modelpath;
bool getSamples(const string& path, vector<string>& filenames)
{
	ifstream sap(path.c_str());
	if (!sap.is_open())
	{
		cerr << "sample open failed" << endl;
		return false;
	}

	string filename;
	while (getline(sap, filename) && !filename.empty())
	{
		filenames.push_back(filename);
	}

	return true;
}

void Train(DetectionAlgorithm* detectors[], string detectornames[], const string& pospath, const string& negpath, const string& negoripath,
	const string& hardfilepath, const string& modelpaths)
{
	_mkdir(hardfilepath.c_str());
	_mkdir(modelpaths.c_str());
	vector<string> posfiles;
	vector<string> negfiles;
	vector<string> negorifiles;

	Utils::findallfiles(pospath, posfiles,"png");
	Utils::findallfiles(negpath, negfiles,"png");
	Utils::findallfiles(negoripath, negorifiles, "png");

	//string detectornames[4] = { "SLTP","HDD","LTDP","HONV" };
//#pragma omp parallel for 
	for (int j = 2; j < 3; ++j)
	{
		Ptr<SVM> mysvm = SVM::create();
		mysvm->setKernel(SVM::LINEAR);
		mysvm->setType(SVM::C_SVC);
		mysvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

		Mat FeatureMat, LabelMat;
		int featurelen = detectors[j]->getFeatureLen();

		FeatureMat = Mat::zeros(posfiles.size() + negfiles.size(), featurelen, CV_32FC1);
		LabelMat = Mat::zeros(posfiles.size() + negfiles.size(), 1, CV_32S);

		cout << "读取正样本" << endl;
		cout << "正样本数量：" << posfiles.size() << endl;
#pragma omp parallel for
		for (int i = 0; i < posfiles.size(); ++i)
		{
			//cout << i << endl;
			string path = pospath + posfiles[i];
			//cout << posfiles[i] << endl;
			vector<float> description;
			description.reserve(featurelen);
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			if (sample.rows != 128 || sample.cols != 64)
			{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
			}

			//filterimg=sample;
			detectors[j]->compute(sample, description);

			float* ptr = FeatureMat.ptr<float>(i);
			memcpy(ptr, &description[0], sizeof(float)*featurelen);

			LabelMat.at<int>(i, 0) = 1;
		}
		cout << "正样本计算完毕" << endl;
		cout << "读取负样本" << endl;
		cout << "负样本数量：" << negfiles.size() << endl;

#pragma omp parallel for
		for (int i = 0; i < negfiles.size(); ++i)
		{
			//cout << i << endl;
			//cout << negfiles[i] << endl;
			string path = negpath + negfiles[i];
			vector<float> description;
			description.reserve(featurelen);
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			if (sample.rows != 128 || sample.cols != 64)
			{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
			}

			detectors[j]->compute(sample, description);

			float* ptr = FeatureMat.ptr<float>(i + posfiles.size());
			memcpy(ptr, &description[0], sizeof(float)*featurelen);

			LabelMat.at<int>(i + posfiles.size(), 0) = -1;
		}

		//第一轮训练
		cout << "开始训练" << endl;
		Ptr<TrainData> tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
		//mysvm->trainAuto(tData);
		mysvm->train(tData);
		mysvm->save(modelpaths + detectornames[j] + "_"+traintype+"svm1.xml");

		cout << "第一轮训练完毕，boosstrap" << endl;

		//计算platt_scaling参数
		Mat preLabel(LabelMat.size(), CV_32SC1);
		preLabel.setTo(-1);
		cout << "platt_scaling" << endl;
		for (int i = 0; i < posfiles.size();++i)
		{
			string path = pospath + posfiles[i];
			//cout << posfiles[i] << endl;
			vector<float> description;
			description.reserve(featurelen);
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			if (sample.rows != 128 || sample.cols != 64)
			{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
			}

			//filterimg=sample;
			vector<Rect> founds;
			vector<double> weights;
			detectors[j]->detectMultiScale(sample, founds, weights);
			if (!founds.empty())
			{
				preLabel.at<int>(i, 0) = 1;
			}
		}

		for (int i = 0; i < negfiles.size(); ++i)
		{
			string path = negpath + negfiles[i];
			//cout << posfiles[i] << endl;
			vector<float> description;
			description.reserve(featurelen);
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			if (sample.rows != 128 || sample.cols != 64)
			{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
			}

			Mat sample;
			if (pfunc == NULL)
			{
				sample = sample;
			}
			else
				pfunc(sample, sample);
			//filterimg=sample;
			vector<Rect> founds;
			vector<double> weights;
			detectors[j]->detectMultiScale(sample, founds, weights);
			if (!founds.empty())
			{
				preLabel.at<int>(i+posfiles.size(), 0) = 1;
			}
		}

		//bootstrap
		for (int h = 0; h < 1; ++h)
		{
			char hardName[256];

			sprintf(hardName, "%s%d\\", (hardfilepath+detectornames[j]+"\\"+traintype+"\\").c_str(), h);
			_mkdir((hardfilepath + detectornames[j]).c_str());
			_mkdir((hardfilepath + detectornames[j] + "\\" + traintype + "\\").c_str());
			_mkdir(hardName);
			string hardtemppath(hardName);
			int cursamplesize = FeatureMat.rows;
			detectors[j]->setSvmDetector(mysvm);		
			for (int i = 0; i < negorifiles.size(); ++i)
			{
				//cout << i << endl;
				string path = negoripath + negorifiles[i];
				Mat sample = imread(path, IMREAD_ANYDEPTH);

				Mat sample;
				if (pfunc == NULL)
				{
					sample = sample;
				}
				else
					pfunc(sample, sample);

				vector<Rect> found;
				vector<double> weights;
				detectors[j]->detectMultiScale(sample, found, weights);

				for (int j = 0; j < found.size(); ++j)
				{
					//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部  
					Rect r = found[j];
					if (r.x < 0)
						r.x = 0;
					if (r.y < 0)
						r.y = 0;
					if (r.x + r.width > sample.cols)
						r.width = sample.cols - r.x;
					if (r.y + r.height > sample.rows)
						r.height = sample.rows - r.y;

					//将矩形框保存为图片，就是Hard Example  
					Mat hardExampleImg = sample(r);//从原图上截取矩形框大小的图片
					char saveName[256];//裁剪出来的负样本图片文件名
					string hardsavepath = hardtemppath + negorifiles[i];
					hardsavepath.erase(hardsavepath.end() - 4, hardsavepath.end());
					resize(hardExampleImg, hardExampleImg, Size(64, 128), INTER_NEAREST);//将剪裁出来的图片缩放为64*128大小  
					sprintf(saveName, "%s-%02d.png", hardsavepath.c_str(), j);//生成hard example图片的文件名  
					imwrite(saveName, hardExampleImg);//保存文件  
				}
				found.clear();
				sample.release();
			}

			vector<string> hardfiles;
			Utils::findallfiles(hardtemppath, hardfiles, "png");
			cout << "错误分类数: " << hardfiles.size() << endl;
			if (hardfiles.size()<10)
			{
				break;
			}

			FeatureMat.resize(FeatureMat.rows + hardfiles.size());
			LabelMat.resize(LabelMat.rows + hardfiles.size());

#pragma omp parallel for
			for (int i = 0; i < hardfiles.size(); ++i)
			{
				string path = hardtemppath + hardfiles[i];
				//cout << hardfiles[i] << endl;
				vector<float> description;
				description.reserve(featurelen);
				Mat sample = imread(path, IMREAD_ANYDEPTH);
				if (sample.rows != 128 || sample.cols != 64)
				{
					resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
				}
				//filterimg = sample;
				Mat sample;
				if (pfunc == NULL)
				{
					sample = sample;
				}
				else
					pfunc(sample, sample);

				detectors[j]->compute(sample, description);

				float* ptr = FeatureMat.ptr<float>(i + cursamplesize);
				memcpy(ptr, &description[0], sizeof(float)*featurelen);

				LabelMat.at<int>(i + cursamplesize, 0) = -1;
			}

			//train again
			cout << "再次训练: " << h << endl;
			tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
			//mysvm->trainAuto(tData);
			mysvm->train(tData);
			cout << "训练完毕" << endl;
			char svmname[256];
			sprintf(svmname, (modelpaths + detectornames[j] + "_" + traintype+ "svm%d.xml").c_str(), h + 2);
			mysvm->save(svmname);
		}
	}
}

//训练单个分类器
void Train(DetectionAlgorithm* detector,const string& pospath, const vector<string>& posfiles, 
	const string& negpath, const vector<string>& negfiles, const string& negoripath, const vector<string>& negorifiles)
{
	Ptr<SVM> mysvm = SVM::create();
	mysvm->setKernel(SVM::LINEAR);
	mysvm->setType(SVM::C_SVC);
	mysvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	Mat FeatureMat;
	Mat LabelMat;

	int featurelen = detector->getFeatureLen();
	FeatureMat = Mat::zeros(posfiles.size() + negfiles.size(), featurelen, CV_32FC1);
	LabelMat = Mat::zeros(posfiles.size() + negfiles.size(), 1, CV_32S);

	//读取正样本
	cout << "读取正样本" << endl;
	cout << "正样本数量：" << posfiles.size() << endl;
#pragma omp parallel for default(none) shared(detector,FeatureMat,featurelen,LabelMat,pospath,posfiles)
	for (int i = 0; i < posfiles.size();++i)
	{
		string path = pospath + posfiles[i];
		//cout << posfiles[i] << endl;
		vector<float> description;
		description.reserve(featurelen);
		Mat sample = imread(path,IMREAD_ANYDEPTH);
		if (sample.rows!= 128 ||sample.cols!=64)
		{
			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}

		//filterimg=sample;
		detector->compute(sample, description);

		float* ptr = FeatureMat.ptr<float>(i);
		memcpy(ptr, &description[0], sizeof(float)*featurelen);

		LabelMat.at<int>(i, 0) = 1;
	}

	cout << "正样本计算完毕" << endl;
	cout << "读取负样本" << endl;
	cout << "负样本数量：" << negfiles.size() << endl;

#pragma omp parallel for default(none) shared(detector,FeatureMat,featurelen,LabelMat,negpath,negfiles,posfiles)
	for (int i = 0; i < negfiles.size(); ++i)
	{
		//cout << negfiles[i] << endl;
		string path = negpath + negfiles[i];
		vector<float> description;
		description.reserve(featurelen);
		Mat sample = imread(path, IMREAD_ANYDEPTH);
		if (sample.rows != 128 || sample.cols != 64)
		{
			resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
		}
	
		//filterimg=sample;
		detector->compute(sample, description);

		float* ptr = FeatureMat.ptr<float>(i+posfiles.size());
		memcpy(ptr, &description[0], sizeof(float)*featurelen);

		LabelMat.at<int>(i+posfiles.size(), 0) = -1;
	}

	//设置训练数据
	cout << "开始训练" << endl;
	Ptr<TrainData> tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
	//mysvm->trainAuto(tData);
	mysvm->train(tData);
	cout << "第一轮训练完毕" << endl;
	//在原始负样本上测试
	mysvm->save((modelpath + "sltpsvmfm1.xml").c_str());
	
	for (int h = 0; h < 2;++h)
	{
		char hardName[256];
		sprintf(hardName, "%s%d\\", hardpath.c_str(), h);
		_mkdir(hardName);
		string hardtemppath(hardName);

		detector->setSvmDetector(mysvm);
		int cursamplesize = FeatureMat.rows;
		//cout << cursamplesize << endl;

#pragma omp parallel for		
		for (int i = 0; i < negorifiles.size(); ++i)
		{
			//cout << i << endl;
			string path = negoripath + negorifiles[i];
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			Mat filterimg;
			preProcessing::pixelFilter(sample, filterimg);
			//filterimg = sample;
			vector<Rect> found;
			vector<double> weights;
			vector<Point> ts;
			//honv.detect(sample, ts, weights);
			detector->detectMultiScale(filterimg, found, weights);
			/*if (found.size() > 0)
				cout << found.size() << endl;*/
		    for (int j = 0; j < found.size(); ++j)
			{
				//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部  
				Rect r = found[j];
				if (r.x < 0)
					r.x = 0;
				if (r.y < 0)
					r.y = 0;
				if (r.x + r.width > sample.cols)
					r.width = sample.cols - r.x;
				if (r.y + r.height > sample.rows)
					r.height = sample.rows - r.y;

				//将矩形框保存为图片，就是Hard Example  
				Mat hardExampleImg = sample(r);//从原图上截取矩形框大小的图片
				char saveName[256];//裁剪出来的负样本图片文件名
				string hardsavepath = hardtemppath + negorifiles[i];
				hardsavepath.erase(hardsavepath.end() - 4, hardsavepath.end());
				resize(hardExampleImg, hardExampleImg, Size(64, 128), INTER_NEAREST);//将剪裁出来的图片缩放为64*128大小  
				sprintf(saveName, "%s-%02d.png", hardsavepath.c_str(), j);//生成hard example图片的文件名  
				imwrite(saveName, hardExampleImg);//保存文件  
			}
			found.clear();
			sample.release();
		}

		vector<string> hardfiles;
		Utils::findallfiles(hardtemppath, hardfiles, "png");
		cout << "错误分类数: " << hardfiles.size() << endl;
		if (hardfiles.size()<10)
		{
			break;
		}
		
		FeatureMat.resize(FeatureMat.rows + hardfiles.size());
		LabelMat.resize(LabelMat.rows + hardfiles.size());

#pragma omp parallel for
		for (int i = 0; i < hardfiles.size();++i)
		{
			string path = hardtemppath + hardfiles[i];
			//cout << hardfiles[i] << endl;
			vector<float> description;
			description.reserve(featurelen);
			Mat sample = imread(path, IMREAD_ANYDEPTH);
			if (sample.rows != 128 || sample.cols != 64)
			{
				resize(sample, sample, Size(64, 128), 0.0, 0.0, INTER_NEAREST);
			}
			detector->compute(sample, description);

			float* ptr = FeatureMat.ptr<float>(i+ cursamplesize);
			memcpy(ptr, &description[0], sizeof(float)*featurelen);

			LabelMat.at<int>(i+ cursamplesize, 0) = -1;
		}

		//train again
		cout << "再次训练: " <<h<< endl;
		tData = TrainData::create(FeatureMat, ROW_SAMPLE, LabelMat);
		//mysvm->trainAuto(tData);
		mysvm->train(tData);
		cout << "训练完毕" << endl;
		char svmname[256];
		sprintf(svmname, (modelpath+"sltpsvmfm1%d.xml").c_str(), h+2);
		mysvm->save(svmname);
	}
	//提取hardfiles的特征
	//mysvm->save("sltpsvm2.xml");
	cout << "训练完毕" << endl;
}

int main()
{


	string negpath = "";
	string pospath = "";

	string negoripath = "";
	
	hardpath = "";
	modelpath = "";
	string featurepath = "";

	DetectionAlgorithm* detectors[10];
    //Train here
	

	for (int i = 0; i < 10;++i)
	{
		delete detectors[i];
	}

	system("pause");
}
