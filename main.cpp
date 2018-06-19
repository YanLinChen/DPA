#include<stdio.h>
#include<stdlib.h>
#include"GraphBuilder.cuh"


uchar *canvas;
uchar *device_canvas;



void InitCanvas(size_t totalChanels, size_t totalSize){
	cudaError_t error;
	canvas = new uchar[totalChanels];
	error = cudaMalloc((void**)&device_canvas, totalSize);
	if (error != cudaSuccess){
		printf("\ncuda Memory alloc fail : %s", cudaGetErrorString(error));
	}
	cudaMemset(device_canvas, 0, totalSize);
}
//program texturePath displayWidth
int main(int argc, char **argv){
	if (argc < 3){
		printf("\nArguments count less than 3 is not allow");
	}
	printf("arguments:%d\n",argc);
	std::vector<Image*>texes = std::vector<Image*>(argc-2);
	for (int i = 0; i < argc - 2; ++i){
		printf("%s\n",argv[i+1]);
		texes[i] = new Image(argv[i+1]);
	}
	uint width = std::stoi(argv[argc-1]);
	uint height = width / texes[0]->ratio();
	size_t totalChanels = width*height*CHANNEL;
	size_t totalSize = sizeof(uchar)*totalChanels;
	GraphBuilder *GB = new GraphBuilder();
	for (int i = 0; i < texes.size(); ++i){
		Image *tex = texes[i];
		Image* simGraph = GB->Bulid_DisSimilarGraph(tex);
		CellGraphBudffer_t *cellBuffer = GB->BulidCellGraph(tex, simGraph);
		GB->OptimizeCurve(cellBuffer, tex->GetSize());
		Image *result = new Image(cv::Size(width, height));
		GB->Rasterizer(cellBuffer, simGraph, tex, result);
		result->toHost();
		char str[255];
		sprintf_s(str, "result_%04d.png", i);
		result->Write(str);
	}
	printf("Done\n");
	

}

