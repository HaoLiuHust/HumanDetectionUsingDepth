#ifndef OPENCVHEADER_H
#define OPENCVHEADER_H
#include <cv.hpp>
#include <cxcore.hpp>
#include "opencv2/highgui.hpp"
template<typename _Tp> static inline _Tp gcd(_Tp a, _Tp b)
{
	if (a < b)
		std::swap(a, b);
	while (b > 0)
	{
		_Tp r = a % b;
		a = b;
		b = r;
	}
	return a;
}
//using namespace cv;
#endif
