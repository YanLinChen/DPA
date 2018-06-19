#pragma once
#include"Image.h" 
#include<iostream>
#include<fstream>


#define LOG
typedef struct{
	float2 *pos;
	int4 *vneighbors;
	int *vflags;
	int4 *vnNcolors;
	int4 *vnEcolors;
	int4 *vnScolors;
	int4 *vnWcolors;
}CellGraphBudffer_t;
typedef struct{
	int4 nNcolors;
	int4 nEcolors;
	int4 nScolors;
	int4 nWcolors;
	float2 pos;
	int4 neighbors;
	int flags;
}CellGraphPixel_t;
static CellGraphBudffer_t* GenCGBuffer(int width,int height){
	CellGraphBudffer_t *cgBuffer=new CellGraphBudffer_t();
	cudaMalloc((void**)&cgBuffer->pos, sizeof(float2)*(width-1)*(height-1)*2);
	cudaMalloc((void**)&cgBuffer->vneighbors, sizeof(int4)*(width - 1)*(height - 1) * 2);
	cudaMalloc((void**)&cgBuffer->vflags, sizeof(int)*(width - 1)*(height - 1) * 2);
	cudaMalloc((void**)&cgBuffer->vnEcolors, sizeof(int4)*(width - 1)*(height - 1) * 2);
	cudaMalloc((void**)&cgBuffer->vnNcolors, sizeof(int4)*(width - 1)*(height - 1) * 2);
	cudaMalloc((void**)&cgBuffer->vnScolors, sizeof(int4)*(width - 1)*(height - 1) * 2);
	cudaMalloc((void**)&cgBuffer->vnWcolors, sizeof(int4)*(width - 1)*(height - 1) * 2);
	return cgBuffer;
}
//<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>
//const int yt = blockIdx.y * blockDim.y + threadIdx.y;
//const int xt = blockIdx.x * blockDim.x + threadIdx.x;
//const int curt = wt*yt+xt;
class GraphBuilder
{
#ifdef LOG
	std::ofstream logFile;
#endif //LOG
public:
	GraphBuilder();
	~GraphBuilder();
	Image* Bulid_DisSimilarGraph(Image *source);
	CellGraphBudffer_t* BulidCellGraph(Image *source, Image *simGraph);
	void OptimizeCurve(CellGraphBudffer_t* cgBuffer, cv::Size sourceSize);
	void Rasterizer(CellGraphBudffer_t *cgBuffer, Image *simGraph, Image *source,Image * result);
};

