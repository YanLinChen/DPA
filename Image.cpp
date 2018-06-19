#include "Image.h"


void Image::toDevice()
{
	ClearDevice();
	error = cudaMalloc((void**)&device_ptr, GetTexSize()*CHANNEL* sizeof(uchar));
	if (error != cudaSuccess){
		printf("\ncuda Memory alloc fail : %s", cudaGetErrorString(error));
	}

	error = cudaMemcpy(device_ptr, GetRaw(), GetTexSize()* CHANNEL*sizeof(uchar), cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printf("\ncuda Memory copy fail : %s", cudaGetErrorString(error));
	}
}

void Image::toHost(cv::Size texSize)
{
	uchar* data = new uchar[texSize.height*texSize.width*CHANNEL];
	error = cudaMemcpy(data, device_ptr, texSize.height*texSize.width*CHANNEL*sizeof(uchar), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("\ncuda Memory copy fail : %s", cudaGetErrorString(error));
	}
	src = cv::Mat(texSize, CV_8UC3, data);
}

void Image::toHost()
{
	toHost(GetSize());
}

Image::Image(std::string tex_path)
{
	src = cv::imread(tex_path);
	toDevice();
}


Image::Image(cv::Size texSize)
{
	src = cv::Mat(texSize, CV_8UC3, cvScalar(0));
	toDevice();
}

Image::Image(cv::Size texSize, uchar* data, bool isDevice/*=false*/)
{
	if (isDevice){
		ClearDevice();
		device_ptr = data;
		toHost(texSize);
	}
	else{
		src = cv::Mat(texSize, CV_8UC3, &data[0]);
		toDevice();
	}
}

Image::~Image()
{
	src.release();
}

uchar* Image::GetRaw()
{
	return src.data;
}

uchar* Image::GetDeviceRaw()
{
	return device_ptr;
}

float Image::ratio()
{
	return src.cols*1.0f / src.rows;
}


cv::Size Image::GetSize(int scale/*=1*/, int offset/*=0*/)
{
	return cv::Size(src.cols*scale + offset, src.rows*scale + offset);
}

uint Image::GetTexSize() const
{
	return src.rows*src.cols;
}

void Image::ClearDevice()
{
	if (device_ptr != NULL){
		cudaFree(device_ptr);
	}
}


void Image::Show(bool pause /*= false*/, int scale/*=10*/)
{
	cv::Mat temp;
	cv::resize(src, temp, cv::Size(scale*src.cols, scale*src.rows),0,0,0);
	imshow("tex", temp);
	if (pause)cv::waitKey(0);
}
