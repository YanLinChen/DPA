#pragma once
#include<opencv2\core.hpp>
#include<opencv2\imgproc.hpp>
#include<opencv2\highgui.hpp>
#include<cuda_runtime_api.h>
#include<cuda_device_runtime_api.h>
#include "device_launch_parameters.h"
#include<string>
#define CHANNEL 3

typedef struct{
	uchar* data;
	int width, height;
	__host__ __device__ const int InverseY(int y){
		return height - 1 - y;
	}
	__host__ __device__ const float InverseY(float y){
		return height - y;
	}
	__host__ __device__ const int GetIndex(int x, int y){
		return InverseY(y)*width + x;
	}
	__host__ __device__ const int GetPos(int x, int y){
		return GetIndex(x, y)*CHANNEL;
	}	
	__host__ __device__ const int GetIndex(int2 xy){
		return InverseY(xy.y)*width + xy.x;
	}
	__host__ __device__ const int GetPos(int2 xy){
		return GetIndex(xy.x, xy.y)*CHANNEL;
	}
	__host__ __device__ float2 GetUV(int2 xy){
		float2 uv;
		uv.x = xy.x*1.0f / width;
		uv.y = xy.y*1.0f / height;
		return uv;
	}
	__host__ __device__ float2 GetUV(int x, int y){
		float2 uv;
		uv.x = x*1.0f / width;
		uv.y = y*1.0f / height;
		return uv;
	}
	__host__ __device__ uchar* Texture2D(float2 uv){
		int x = floor(uv.x*width);
		int y = floor(uv.y*height);
		return Texture2D(x, y);
	}
	__host__ __device__ uchar* Texture2D(int x, int y){
		return &data[(width*InverseY(y) + x)*CHANNEL];
	}
	__host__ __device__ void SetPixel(int pos, uchar* data){
		for (int i = 0; i < CHANNEL; ++i){
			this->data[pos + i] = data[i];
		}
	}
	__host__ __device__ void SetPixel(int pos, uchar x, uchar y , uchar z){
		SetChannel(pos, x);
		SetChannel(pos + 1, y);
		SetChannel(pos + 2, z);
	}
	__host__ __device__ void SetChannel(int pos, uchar data){
		this->data[pos] = data;
	}
	__host__ __device__ uchar GetChannel(int pos){
		return this->data[pos];
	}
}texture_t;
class Image
{
	cudaError_t error;
	cv::Mat src;
	uchar* device_ptr=NULL;
	void toHost(cv::Size texSize);
public:
	texture_t GetTexture(){
		texture_t t;
		t.data = device_ptr;
		t.height = src.rows;
		t.width = src.cols;
		return t;
	}
	//load image
	Image(std::string tex_path);
	//create empty tex with size
	Image(cv::Size texSize);
	//create tex with data in host/device
	Image(cv::Size texSize,uchar* data,bool isDevice=false);
	void toDevice();
	void toHost();
	~Image();
	uchar* GetRaw();
	uchar* GetDeviceRaw();
	float ratio();// [width]/[height]
	cv::Size GetSize(int scale=1,int offset=0);
	uint GetTexSize()const;
	void ClearDevice();
	void Show(bool pause = false,int scale=20);
	void Write(std::string path){
		imwrite(path, src);
	}
};

