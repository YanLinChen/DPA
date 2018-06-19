#include "GraphBuilder.cuh"
#include "cutil_math.cuh"
__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }

#pragma region DEFINE
#define EDGE_HORVERT 16
#define EDGE_DIAGONAL_ULLR 32
#define EDGE_DIAGONAL_LLUR 64
#define EDGE_CROSSING 96
#define NORTH		128
#define NORTHEAST	64
#define EAST		32
#define SOUTHEAST	16
#define SOUTH		8
#define SOUTHWEST	4
#define WEST		2
#define NORTHWEST	1
#define HAS_NORTHERN_NEIGHBOR 1
#define HAS_EASTERN_NEIGHBOR 2
#define HAS_SOUTHERN_NEIGHBOR 4
#define HAS_WESTERN_NEIGHBOR 8
#define HAS_NORTHERN_SPLINE 16
#define HAS_EASTERN_SPLINE 32
#define HAS_SOUTHERN_SPLINE 64
#define HAS_WESTERN_SPLINE 128
#define HAS_CORRECTED_POSITION 256
#define DONT_OPTIMIZE_N 512
#define DONT_OPTIMIZE_E 1024
#define DONT_OPTIMIZE_S 2048
#define DONT_OPTIMIZE_W 4096
#define NORTH_G		1
#define EAST_G		2
#define SOUTH_G		4
#define WEST_G		8
#define CENTER		16
#define XOFFSET_CUL	-0.25
#define YOFFSET_CUL	0.25
#define XOFFSET_CUR	0.25
#define YOFFSET_CUR	0.25
#define XOFFSET_CLL	-0.25
#define YOFFSET_CLL	-0.25
#define XOFFSET_CLR	0.25
#define YOFFSET_CLR	-0.25
#define HAS_NORTHERN_NEIGHBOR 1
#define HAS_EASTERN_NEIGHBOR 2
#define HAS_SOUTHERN_NEIGHBOR 4
#define HAS_WESTERN_NEIGHBOR 8
#define HAS_NORTHERN_SPLINE 16
#define HAS_EASTERN_SPLINE 32
#define HAS_SOUTHERN_SPLINE 64
#define HAS_WESTERN_SPLINE 128
#define HAS_CORRECTED_POSITION 256
#define DONT_OPTIMIZE_N 512
#define DONT_OPTIMIZE_E 1024
#define DONT_OPTIMIZE_S 2048
#define DONT_OPTIMIZE_W 4096
#define POSITIONAL_ENERGY_SCALING 2.5
#define LIMIT_SEARCH_ITERATIONS 20.0
#define R  						0.61803399
#define C  						1 - R
#define TOL 					0.0001
#define BRACKET_SEARCH_A 		0.1
#define BRACKET_SEARCH_B  		-0.1
#define GOLD 					1.618034
#define GLIMIT 					10.0
#define TINY 					0.000000001 
#define STEP 0.2
#define GAUSS_MULTIPLIER 2.5
#define LINEAR_SEGMENT_LENGTH 1.0
#pragma endregion DEFINE

#pragma region BASIC
__device__ float distance(float3 A, float3 B){
	return sqrt((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y) + (A.z - B.z)*(A.z - B.z));
}

__device__ float distance(float2 A, float2 B){
	return sqrt((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y));
}

__device__ float3 vec3(float a, float b, float c){
	float3 ans;
	ans.x = a;
	ans.y = b;
	ans.z = c;
	return ans;
}

__device__ float3 vec3(float2 a, float b){
	float3 ans;
	ans.x = a.x;
	ans.y = a.y;
	ans.z = b;
	return ans;
}

__device__ float2 vec2(float a, float b){
	float2 ans;
	ans.x = a;
	ans.y = b;
	return ans;
}

__device__ int2 ivec2(int a, int b){
	int2 ans;
	ans.x = a;
	ans.y = b;
	return ans;
}

__device__ int3 ivec3(int a, int b, int c){
	int3 ans;
	ans.x = a;
	ans.y = b;
	ans.z = c;
	return ans;
}

__device__ int4 ivec4(int a, int b, int c, int d){
	int4 ans;
	ans.x = a;
	ans.y = b;
	ans.z = c;
	ans.w = d;
	return ans;
}

__device__ float4 vec4(float a, float b, float c, float d){
	float4 ans;
	ans.x = a;
	ans.y = b;
	ans.z = c;
	ans.w = d;
	return ans;
}

__device__ int4 ivec4(int2 a, int2 b){
	int4 ans;
	ans.x = a.x;
	ans.y = a.y;
	ans.z = b.x;
	ans.w = b.y;
	return ans;
}

__device__ bool isSimilar(uchar3 pixelA, uchar3 pixelB) {
	//Y = 0.299*R + 0.587*G + 0.114*B
	//U = (B-Y)*0.493
	//V = (R-Y)*0.877
	float yA = 0.299f*pixelA.z + 0.587f*pixelA.y + 0.114f*pixelA.x;
	float uA = 0.493f*(pixelA.x - yA);
	float vA = 0.877f*(pixelA.z - yA);
	float yB = 0.299f*pixelB.z + 0.587f*pixelB.y + 0.114f*pixelB.x;
	float uB = 0.493f*(pixelB.x - yB);
	float vB = 0.877f*(pixelB.z - yB);

	bool similar = false;
	if (abs(yA - yB) <= 48.0) {
		if (abs(uA - uB) <= 7.0) {
			if (abs(vA - vB) <= 6.0) {
				similar = true;
			}
		}
	}
	return similar;
}

__device__ bool isSimilar(uchar* pixelA, uchar* pixelB) {
	//Y = 0.299*R + 0.587*G + 0.114*B
	//U = (B-Y)*0.493
	//V = (R-Y)*0.877
	float yA = 0.299f*pixelA[2] + 0.587f*pixelA[1] + 0.114f*pixelA[0];
	float uA = 0.493f*(pixelA[0] - yA);
	float vA = 0.877f*(pixelA[2] - yA);
	float yB = 0.299f*pixelB[2] + 0.587f*pixelB[1] + 0.114f*pixelB[0];
	float uB = 0.493f*(pixelB[0] - yB);
	float vB = 0.877f*(pixelB[2] - yB);

	bool similar = false;
	if (abs(yA - yB) <= 48.0) {
		if (abs(uA - uB) <= 7.0) {
			if (abs(vA - vB) <= 6.0) {
				similar = true;
			}
		}
	}
	return similar;
}

__device__ int Clamp(int val, int min, int max){
	return val<min ? min : val>max ? max : val;
}

__device__ float2 m_normalize(float2 data){
	float len = sqrt(data.x*data.x + data.y*data.y);
	return vec2(data.x / len, data.y / len);
}

__device__ float m_dot(float2 a, float2 b){
	return a.x*b.x + a.y*b.y;
}

__device__ float sign(float v){
	return v > 0 ? 1 : -1;
}
#pragma endregion BASIC

#pragma region SIMILARITY_GRAPH
__global__ void DisSimilarGraph_Kernel(texture_t input, texture_t output){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= output.width - 1 || y >= output.height - 1 || x <= 0 || y <= 0)return;
	const int cur = output.GetIndex(x, y);
	int modx = x % 2, mody = y % 2;
	//if the current fragment represents a node invalidate it by setting it to zero
	if (modx == 1 && mody == 1) {
		for (int i = 0; i < CHANNEL; ++i){
			//set output=input
			output.SetPixel(cur*CHANNEL, input.Texture2D((x - 1) / 2, (y - 1) / 2));
		}
	}
	else if (modx == 0 && mody == 0){
		//UL ... diagonal
		int diagonal = 0;
		//check UL-LR connection
		int pAx = Clamp((x - 1 - 1) / 2, 0, output.width), pAy = Clamp((y + 1 - 1) / 2, 0, output.height);
		uchar *pA = input.Texture2D(pAx, pAy);
		int pBx = Clamp((x + 1 - 1) / 2, 0, output.width), pBy = Clamp((y - 1 - 1) / 2, 0, output.height);
		uchar *pB = input.Texture2D(pBx, pBy);
		if (isSimilar(pA, pB)) {
			diagonal = EDGE_DIAGONAL_ULLR;
		}
		//check LL-UR connection
		pAx = Clamp((x - 1 - 1) / 2, 0, output.width); pAy = Clamp((y - 1 - 1) / 2, 0, output.height);
		pA = input.Texture2D(pAx, pAy);
		pBx = Clamp((x + 1 - 1) / 2, 0, output.width), pBy = Clamp((y + 1 - 1) / 2, 0, output.height);
		pB = input.Texture2D(pBx, pBy);
		if (isSimilar(pA, pB)) {
			diagonal = diagonal | EDGE_DIAGONAL_LLUR;
		}
		output.SetChannel(cur*CHANNEL + 2, diagonal);
	}
	else if (modx == 0 && mody == 1){
		//LL ... horizontal Edge
		//pA, pB ... pixels connected by edge
		int pAx = Clamp((x - 1 - 1) / 2, 0, output.width), pAy = Clamp((y - 1) / 2, 0, output.height);
		uchar *pA = input.Texture2D(pAx, pAy);
		int pBx = Clamp((x + 1 - 1) / 2, 0, output.width), pBy = Clamp((y - 1) / 2, 0, output.height);
		uchar *pB = input.Texture2D(pBx, pBy);
		if (isSimilar(pA, pB)) {
			output.SetChannel(cur*CHANNEL + 2, 16);
		}
	}
	else if (modx == 1 && mody == 0){
		//UR ... vertical edge
		//pA, pB ... pixels connected by edge
		int pAx = Clamp((x - 1) / 2, 0, output.width), pAy = Clamp((y + 1 - 1) / 2, 0, output.height);
		uchar *pA = input.Texture2D(pAx, pAy);
		int pBx = Clamp((x - 1) / 2, 0, output.width), pBy = Clamp((y - 1 - 1) / 2, 0, output.height);
		uchar *pB = input.Texture2D(pBx, pBy);
		if (isSimilar(pA, pB)) {
			output.SetChannel(cur*CHANNEL + 2, 16);
		}
	}
}

__global__ void ValenceUpdate_Kernel(texture_t input, texture_t output){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= output.width - 1 || y >= output.height - 1 || x <= 0 || y <= 0)return;
	const int cur = output.GetIndex(x, y);
	int modx = x % 2, mody = y % 2;
	if (modx == 1 && mody == 1) {
		//calculate node valence
		int valence = 0;
		int edges = 0;
		//browse neighborhood
		//NW
		int pAx = Clamp(x - 1, 0, output.width), pAy = Clamp(y + 1, 0, output.height);
		int posA = output.GetPos(pAx, pAy) + 2;
		uchar edgeValue = input.GetChannel(posA);
		if ((edgeValue & 32) == 32) {
			valence++;
			edges = edges | NORTHWEST;
		}
		//N
		pAx = Clamp(x, 0, output.width);
		pAy = Clamp(y + 1, 0, output.height);
		posA = output.GetPos(pAx, pAy) + 2;
		if (input.GetChannel(posA) > 0) {
			valence++;
			edges = edges | NORTH;
		}
		//NE
		pAx = Clamp(x + 1, 0, output.width);
		pAy = Clamp(y + 1, 0, output.height);
		posA = output.GetPos(pAx, pAy) + 2;
		edgeValue = input.GetChannel(posA);
		if ((edgeValue & 64) == 64) {
			valence++;
			edges = edges | NORTHEAST;
		}
		//E
		pAx = Clamp(x + 1, 0, output.width);
		pAy = Clamp(y, 0, output.height);
		posA = output.GetPos(pAx, pAy) + 2;
		if (input.GetChannel(posA) > 0) {
			valence++;
			edges = edges | EAST;
		}
		//SE
		pAx = Clamp(x + 1, 0, output.width);
		pAy = Clamp(y - 1, 0, output.height);
		posA = output.GetPos(pAx, pAy) + 2;
		edgeValue = input.GetChannel(posA);
		if ((edgeValue & 32) == 32) {
			valence++;
			edges = edges | SOUTHEAST;
		}
		//S
		pAx = Clamp(x, 0, output.width);
		pAy = Clamp(y - 1, 0, output.height);
		posA = output.GetPos(pAx, pAy) + 2;
		if (input.GetChannel(posA) > 0) {
			valence++;
			edges = edges | SOUTH;
		}
		//SW
		pAx = Clamp(x - 1, 0, output.width);
		pAy = Clamp(y - 1, 0, output.height);
		posA = output.GetPos(pAx, pAy) + 2;
		edgeValue = input.GetChannel(posA);
		if ((edgeValue & 64) == 64) {
			valence++;
			edges = edges | SOUTHWEST;
		}
		//W
		pAx = Clamp(x - 1, 0, output.width);
		pAy = Clamp(y, 0, output.height);
		posA = output.GetPos(pAx, pAy) + 2;
		if (input.GetChannel(posA) > 0) {
			valence++;
			edges = edges | WEST;
		}
		output.SetChannel(cur*CHANNEL + 2, valence);
		output.SetChannel(cur*CHANNEL + 1, edges);
	}
	else{
		output.SetPixel(cur*CHANNEL, input.Texture2D(x, y));
	}
}

__device__ void countForComponent(int c, int &componentSizeA, int &componentSizeB) {
	switch (c) {
	case 1:
		componentSizeA++;
		break;
	case 2:
		componentSizeB++;
		break;
	default:
		break;
	}
}

__device__ void voteSparsePixels(int x, int y, int &voteA, int &voteB, int &debugA, int &debugB, texture_t input, int &componentSizeA, int &componentSizeB) {
	//INFO on border treatment
	// border treatment currently relies on the similiaritygraph texture wrap setting beeing GL_CLAMP_TO_BORDER
	// addidtionally GL_TEXTURE_BORDER_COLOR needs to be set to black(0,0,0,0), which is the default value btw.

	//label-array 8x8
	//let component A be 1 and B be 2
	int lArray[64] = {
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 2, 0, 0, 0,
		0, 0, 0, 2, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0
	};
	// neigborhood indices
	int nNW = 0;
	int nW = 0;
	int nSW = 0;
	int nS = 0;
	int nSE = 0;
	int nE = 0;
	int nNE = 0;
	int nN = 0;
	//this loop iterates trough 3 levels starting from the middle of the 8x8 label array
	//*
	for (int level = 0; level < 2; level++) {

		int xOFFSET = -(1 + 2 * level);
		int yOFFSET = (1 + (2 * level));
		//NW corner-node
		//nhood ... stores this nodes neighboring information taken from the similarity graph
		int px, py, pos;
		px = Clamp(x + xOFFSET, 0, input.width);
		py = Clamp(y + yOFFSET, 0, input.height);
		pos = input.GetPos(px, py) + 1;
		int nhood = input.GetChannel(pos);
		int currentComponentIndex = 8 * (3 - level) + 3 - level;
		int currentComponent = lArray[currentComponentIndex];
		nS = 8 * (4 - level) + (3 - level);//TODO: OPtimieren
		nSW = 8 * (4 - level) + (2 - level);
		nW = 8 * (3 - level) + (2 - level);
		nNW = 8 * (2 - level) + (2 - level);
		nN = 8 * (2 - level) + (3 - level);
		nNE = 8 * (2 - level) + (4 - level);
		nE = 8 * (3 - level) + (4 - level);

		if (currentComponent == 0) {
			//this block scans neighborhood for connected components, in case this node has not yet been labeled
			if (((nhood & SOUTH) == SOUTH) && (lArray[nS] != 0))		{ currentComponent = lArray[nS]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & SOUTHWEST) == SOUTHWEST) && (lArray[nSW] != 0))		{ currentComponent = lArray[nSW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & WEST) == WEST) && (lArray[nW] != 0))				{ currentComponent = lArray[nW];  lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & NORTHWEST) == NORTHWEST) && (lArray[nNW] != 0))	{ currentComponent = lArray[nNW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & NORTH) == NORTH) && (lArray[nN] != 0))			{ currentComponent = lArray[nN];  lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & NORTHEAST) == NORTHEAST) && (lArray[nNE] != 0))	{ currentComponent = lArray[nNE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & EAST) == EAST) && (lArray[nE] != 0))	{ currentComponent = lArray[nE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
		}

		if (currentComponent != 0) {
			//check SW W NW N NE
			if ((nhood & SOUTHWEST) == SOUTHWEST) {
				//SW
				//check if node is already labeled
				if (lArray[nSW] == 0) {
					lArray[nSW] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & WEST) == WEST) {
				//W
				if (lArray[nW] == 0) {
					lArray[nW] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & NORTHWEST) == NORTHWEST) {
				if (lArray[nNW] == 0) {
					lArray[nNW] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & NORTH) == NORTH) {
				//N
				if (lArray[nN] == 0) {
					lArray[nN] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & NORTHEAST) == NORTHEAST) {
				//NE
				if (lArray[nNE] == 0) {
					lArray[nNE] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
		}

		//N nodes
		if (level>0) {
			for (int i = 0; i < level * 2; i++) {
				xOFFSET = -(2 * level - 1) + 2 * i;
				yOFFSET = (1 + 2 * level);
				px = Clamp(x + xOFFSET, 0, input.width);
				py = Clamp(y + yOFFSET, 0, input.height);
				pos = input.GetPos(px, py) + 1;
				nhood = input.GetChannel(pos);
				//nhood = similarityGraph[FragCoordX-(2*level-1)+2*i + (FragCoordY+1+2*level)*17];
				currentComponentIndex = 8 * (3 - level) + (i + 4 - level);
				currentComponent = lArray[currentComponentIndex];
				nW = 8 * (3 - level) + (i + 3 - level);
				nNW = 8 * (2 - level) + (i + 3 - level);
				nN = 8 * (2 - level) + (i + 4 - level);
				nNE = 8 * (2 - level) + (i + 5 - level);
				nE = 8 * (3 - level) + (i + 5 - level);
				if (currentComponent == 0) {
					//this block scans neighborhood for connected components, in case this node has not yet been labeled
					if (((nhood & WEST) == WEST) && (lArray[nW] != 0))		{ currentComponent = lArray[nW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & NORTHWEST) == NORTHWEST) && (lArray[nNW] != 0))		{ currentComponent = lArray[nNW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & NORTH) == NORTH) && (lArray[nN] != 0))			{ currentComponent = lArray[nN];	 lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & NORTHEAST) == NORTHEAST) && (lArray[nNE] != 0))	{ currentComponent = lArray[nNE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & EAST) == EAST) && (lArray[nE] != 0))	{ currentComponent = lArray[nE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
				}
				//check NW,N,NE neighbors
				if (currentComponent != 0) {
					if ((nhood & NORTHWEST) == NORTHWEST) {
						if (lArray[nNW] == 0) {
							// label the NW neighbor in the label-array
							lArray[nNW] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
					if ((nhood & NORTH) == NORTH) {
						if (lArray[nN] == 0) {
							// label the N neighbor in the label-array
							lArray[nN] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
					if ((nhood & NORTHEAST) == NORTHEAST) {
						if (lArray[nNE] == 0) {
							// label the NE neighbor in the label-array
							lArray[nNE] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
				}
			}
		}

		//NE corner-node
		xOFFSET = (1 + 2 * level);
		yOFFSET = (1 + (2 * level));
		px = Clamp(x + xOFFSET, 0, input.width);
		py = Clamp(y + yOFFSET, 0, input.height);
		pos = input.GetPos(px, py) + 1;
		nhood = input.GetChannel(pos);

		//nhood = similarityGraph[FragCoordX + xOFFSET + (FragCoordY + yOFFSET)*17];
		//current value in the label-array
		currentComponentIndex = 8 * (3 - level) + (4 + level);
		currentComponent = lArray[currentComponentIndex];
		nW = 8 * (3 - level) + (3 + level);
		nNW = 8 * (2 - level) + (3 + level);
		nN = 8 * (2 - level) + (4 + level);
		nNE = 8 * (2 - level) + (5 + level);
		nE = 8 * (3 - level) + (5 + level);
		nSE = 8 * (4 - level) + (5 + level);
		nS = 8 * (4 - level) + (4 + level);
		if (currentComponent == 0) {
			//this block scans neighborhood for connected components, in case this node has not yet been labeled
			if (((nhood & WEST) == WEST) && (lArray[nNW] != 0)){ currentComponent = lArray[nW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & NORTHWEST) == NORTHWEST) && (lArray[nNW] != 0)){ currentComponent = lArray[nNW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & NORTH) == NORTH) && (lArray[nN] != 0)){ currentComponent = lArray[nN];  lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & NORTHEAST) == NORTHEAST) && (lArray[nNE] != 0)){ currentComponent = lArray[nNE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & EAST) == EAST) && (lArray[nE] != 0)){ currentComponent = lArray[nE];  lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & SOUTHEAST) == SOUTHEAST) && (lArray[nSE] != 0)){ currentComponent = lArray[nSE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & SOUTH) == SOUTH) && (lArray[nS] != 0)){ currentComponent = lArray[nS]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
		}
		if (currentComponent != 0) {
			//check NW N NE E SE
			if ((nhood & NORTHWEST) == NORTHWEST) {
				if (lArray[nNW] == 0) {
					lArray[nNW] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & NORTH) == NORTH) {
				if (lArray[nN] == 0) {
					lArray[nN] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & NORTHEAST) == NORTHEAST) {
				if (lArray[nNE] == 0) {
					lArray[nNE] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & EAST) == EAST) {
				if (lArray[nE] == 0) {
					lArray[nE] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & SOUTHEAST) == SOUTHEAST) {
				if (lArray[nSE] == 0) {
					lArray[nSE] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
		}
		//E nodes
		if (level>0) {
			for (int i = 0; i < level * 2; i++) {
				xOFFSET = 1 + 2 * level;
				yOFFSET = (2 * level - 1 - 2 * i);
				px = Clamp(x + xOFFSET, 0, input.width);
				py = Clamp(y + yOFFSET, 0, input.height);
				pos = input.GetPos(px, py) + 1;
				nhood = input.GetChannel(pos);
				//nhood = similarityGraph[FragCoordX + xOFFSET + (FragCoordY + yOFFSET)*17];
				currentComponentIndex = 8 * (i + 4 - level) + (4 + level);
				currentComponent = lArray[currentComponentIndex];
				nN = 8 * (i + 3 - level) + (4 + level);
				nNE = 8 * (i + 3 - level) + (5 + level);
				nE = 8 * (i + 4 - level) + (5 + level);
				nSE = 8 * (i + 5 - level) + (5 + level);
				nS = 8 * (i + 5 - level) + (4 + level);
				if (currentComponent == 0) {
					//this block scans neighborhood for connected components, in case this node has not yet been labeled
					if (((nhood & NORTH) == NORTH) && (lArray[nN] != 0)){ currentComponent = lArray[nN]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & NORTHEAST) == NORTHEAST) && (lArray[nNE] != 0)){ currentComponent = lArray[nNE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & EAST) == EAST) && (lArray[nE] != 0))			{ currentComponent = lArray[nE];	 lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & SOUTHEAST) == SOUTHEAST) && (lArray[nSE] != 0)){ currentComponent = lArray[nSE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & SOUTH) == SOUTH) && (lArray[nS] != 0)){ currentComponent = lArray[nS]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
				}
				//check NE,E,SE neighbors
				if (currentComponent != 0) {
					if ((nhood & NORTHEAST) == NORTHEAST) {
						if (lArray[nNE] == 0) {
							// label the NW neighbor in the label-array
							lArray[nNE] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
					if ((nhood & EAST) == EAST) {
						if (lArray[nE] == 0) {
							// label the N neighbor in the label-array
							lArray[nE] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
					if ((nhood & SOUTHEAST) == SOUTHEAST) {
						if (lArray[nSE] == 0) {
							// label the NE neighbor in the label-array
							lArray[nSE] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
				}

			}
		}
		//SE corner-node
		xOFFSET = (1 + 2 * level);
		yOFFSET = -(1 + 2 * level);
		px = Clamp(x + xOFFSET, 0, input.width);
		py = Clamp(y + yOFFSET, 0, input.height);
		pos = input.GetPos(px, py) + 1;
		nhood = input.GetChannel(pos);
		//nhood = similarityGraph[FragCoordX + xOFFSET + (FragCoordY + yOFFSET)*17];
		currentComponentIndex = 8 * (4 + level) + (4 + level);
		currentComponent = lArray[currentComponentIndex];
		nN = 8 * (3 + level) + (4 + level);
		nNE = 8 * (3 + level) + (5 + level);
		nE = 8 * (4 + level) + (5 + level);
		nSE = 8 * (5 + level) + (5 + level);
		nS = 8 * (5 + level) + (4 + level);
		nSW = 8 * (5 + level) + (3 + level);
		nW = 8 * (4 + level) + (3 + level);
		if (currentComponent == 0) {
			//this block scans neighborhood for connected components, in case this node has not yet been labeled
			if (((nhood & NORTH) == NORTH) && (lArray[nN] != 0))	 { currentComponent = lArray[nN]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & NORTHEAST) == NORTHEAST) && (lArray[nNE] != 0))	 { currentComponent = lArray[nNE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & EAST) == EAST) && (lArray[nE] != 0))			 { currentComponent = lArray[nE];  lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & SOUTHEAST) == SOUTHEAST) && (lArray[nSE] != 0)){ currentComponent = lArray[nSE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & SOUTH) == SOUTH) && (lArray[nS] != 0))		 { currentComponent = lArray[nS];  lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & SOUTHWEST) == SOUTHWEST) && (lArray[nSW] != 0)){ currentComponent = lArray[nSW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & WEST) == WEST) && (lArray[nW] != 0)){ currentComponent = lArray[nW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
		}
		if (currentComponent != 0) {
			//check NE E SE S SW
			if ((nhood & NORTHEAST) == NORTHEAST) {
				if (lArray[nNE] == 0){
					lArray[nNE] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & EAST) == EAST) {
				if (lArray[nE] == 0){
					lArray[nE] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & SOUTHEAST) == SOUTHEAST) {
				if (lArray[nSE] == 0){
					lArray[nSE] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & SOUTH) == SOUTH) {
				if (lArray[nS] == 0){
					lArray[nS] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & SOUTHWEST) == SOUTHWEST) {
				if (lArray[nSW] == 0){
					lArray[nSW] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
		}
		//S nodes
		if (level>0) {
			for (int i = 0; i < level * 2; i++) {
				xOFFSET = -(2 * level - 1) + 2 * i;
				yOFFSET = -(1 + 2 * level);
				px = Clamp(x + xOFFSET, 0, input.width);
				py = Clamp(y + yOFFSET, 0, input.height);
				pos = input.GetPos(px, py) + 1;
				nhood = input.GetChannel(pos);
				//nhood = similarityGraph[FragCoordX + xOFFSET + (FragCoordY + yOFFSET)*17];
				currentComponentIndex = 8 * (4 + level) + (i + 4 - level);
				currentComponent = lArray[currentComponentIndex];
				nE = 8 * (4 + level) + (i + 5 - level);
				nSE = 8 * (5 + level) + (i + 5 - level);
				nS = 8 * (5 + level) + (i + 4 - level);
				nSW = 8 * (5 + level) + (i + 3 - level);
				nW = 8 * (4 + level) + (i + 3 - level);
				if (currentComponent == 0) {
					//this block scans neighborhood for connected components, in case this node has not yet been labeled
					if (((nhood & EAST) == EAST) && (lArray[nE] != 0)){ currentComponent = lArray[nE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & SOUTHEAST) == SOUTHEAST) && (lArray[nSE] != 0)){ currentComponent = lArray[nSE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & SOUTH) == SOUTH) && (lArray[nS] != 0))			{ currentComponent = lArray[nS];	 lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & SOUTHWEST) == SOUTHWEST) && (lArray[nSW] != 0)){ currentComponent = lArray[nSW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & WEST) == WEST) && (lArray[nW] != 0)){ currentComponent = lArray[nW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
				}
				//check SW,S,SE neighbors
				if (currentComponent != 0) {
					if ((nhood & SOUTHEAST) == SOUTHEAST) {
						if (lArray[nSE] == 0) {
							lArray[nSE] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
					if ((nhood & SOUTH) == SOUTH) {
						if (lArray[nS] == 0) {
							lArray[nS] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
					if ((nhood & SOUTHWEST) == SOUTHWEST) {
						if (lArray[nSW] == 0) {
							lArray[nSW] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
				}
			}
		}

		//SW corner-node
		xOFFSET = -(1 + 2 * level);
		yOFFSET = -(1 + 2 * level);
		px = Clamp(x + xOFFSET, 0, input.width);
		py = Clamp(y + yOFFSET, 0, input.height);
		pos = input.GetPos(px, py) + 1;
		nhood = input.GetChannel(pos);
		//nhood = similarityGraph[FragCoordX + xOFFSET + (FragCoordY + yOFFSET)*17];
		currentComponentIndex = 8 * (4 + level) + (3 - level);
		currentComponent = lArray[currentComponentIndex];
		nE = 8 * (4 + level) + (4 - level);
		nSE = 8 * (5 + level) + (4 - level);
		nS = 8 * (5 + level) + (3 - level);
		nSW = 8 * (5 + level) + (2 - level);
		nW = 8 * (4 + level) + (2 - level);
		nNW = 8 * (3 + level) + (2 - level);
		nN = 8 * (3 + level) + (3 - level);
		if (currentComponent == 0) {
			//this block scans neighborhood for connected components, in case this node has not yet been labeled
			if (((nhood & EAST) == EAST) && (lArray[nE] != 0)){ currentComponent = lArray[nE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & SOUTHEAST) == SOUTHEAST) && (lArray[nSE] != 0)){ currentComponent = lArray[nSE]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & SOUTH) == SOUTH) && (lArray[nS] != 0))			{ currentComponent = lArray[nS];  lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & SOUTHWEST) == SOUTHWEST) && (lArray[nSW] != 0)){ currentComponent = lArray[nSW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & WEST) == WEST) && (lArray[nW] != 0))			{ currentComponent = lArray[nW];  lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & NORTHWEST) == NORTHWEST) && (lArray[nNW] != 0)){ currentComponent = lArray[nNW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
			else if (((nhood & NORTH) == NORTH) && (lArray[nN] != 0)){ currentComponent = lArray[nN]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
		}
		if (currentComponent != 0) {
			//check SE S SW W NW
			if ((nhood & SOUTHEAST) == SOUTHEAST) {
				if (lArray[nSE] == 0){
					lArray[nSE] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & SOUTH) == SOUTH) {
				if (lArray[nS] == 0){
					lArray[nS] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & SOUTHWEST) == SOUTHWEST) {
				if (lArray[nSW] == 0){
					lArray[nSW] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & WEST) == WEST) {
				if (lArray[nW] == 0){
					lArray[nW] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
			if ((nhood & NORTHWEST) == NORTHWEST) {
				if (lArray[nNW] == 0){
					lArray[nNW] = currentComponent;
					countForComponent(currentComponent, componentSizeA, componentSizeB);
				}
			}
		}
		//W nodes
		if (level>0) {
			for (int i = 0; i < level * 2; i++) {
				xOFFSET = -(1 + 2 * level);
				yOFFSET = -(2 * level - 1) + 2 * i;
				px = Clamp(x + xOFFSET, 0, input.width);
				py = Clamp(y + yOFFSET, 0, input.height);
				pos = input.GetPos(px, py) + 1;
				nhood = input.GetChannel(pos);
				//nhood = similarityGraph[FragCoordX + xOFFSET + (FragCoordY + yOFFSET)*17];
				currentComponentIndex = 8 * (3 + level - i) + (3 - level);
				currentComponent = lArray[currentComponentIndex];
				nN = 8 * (2 + level - i) + (3 - level);
				nNW = 8 * (2 + level - i) + (2 - level);
				nW = 8 * (3 + level - i) + (2 - level);
				nSW = 8 * (4 + level - i) + (2 - level);
				nS = 8 * (4 + level - i) + (3 - level);
				if (currentComponent == 0) {
					//this block scans neighborhood for connected components, in case this node has not yet been labeled
					if (((nhood & SOUTH) == SOUTH) && (lArray[nS] != 0))	{ currentComponent = lArray[nS]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & SOUTHWEST) == SOUTHWEST) && (lArray[nSW] != 0))	{ currentComponent = lArray[nSW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & WEST) == WEST) && (lArray[nW] != 0))			{ currentComponent = lArray[nW];	 lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & NORTHWEST) == NORTHWEST) && (lArray[nNW] != 0))	{ currentComponent = lArray[nNW]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
					else if (((nhood & NORTH) == NORTH) && (lArray[nN] != 0))	{ currentComponent = lArray[nN]; lArray[currentComponentIndex] = currentComponent; countForComponent(currentComponent, componentSizeA, componentSizeB); }
				}
				//check SW W NW neighbors
				if (currentComponent != 0) {
					if ((nhood & SOUTHWEST) == SOUTHWEST) {
						if (lArray[nSW] == 0) {
							lArray[nSW] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
					if ((nhood & WEST) == WEST) {
						if (lArray[nW] == 0) {
							lArray[nW] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
					if ((nhood & NORTHWEST) == NORTHWEST) {
						if (lArray[nNW] == 0) {
							lArray[nNW] = currentComponent;
							countForComponent(currentComponent, componentSizeA, componentSizeB);
						}
					}
				}
			}
		}
	}
	//Level 4 treatment
	//northen border
	/*
	for (int i = 0; i < 8; i++) {
	int px = Clamp(x - 7 + 2 * i, 0, input.width);
	int py = Clamp(y + 7, 0, input.height);
	int pos = input.GetPos(px, py) + 1;
	int nhood = input.GetChannel(pos);
	}
	*/
	debugA = componentSizeA;
	debugB = componentSizeB;
	//now that the connected component sizes are computed we vote for the smaller component
	if (componentSizeA < componentSizeB) {
		//vote for A ... weight is difference between the sizes of the components
		voteA = voteA + (componentSizeB - componentSizeA);
	}
	else if (componentSizeA > componentSizeB) {
		//vote for B ... weight is difference between the sizes of the components
		voteB = voteB + (componentSizeA - componentSizeB);
	}
}

__device__ void voteIslands(int x, int y, int &voteA, int &voteB, texture_t input) {
	int px, py, pos;
	px = Clamp(x - 1, 0, input.width);
	py = Clamp(y + 1, 0, input.height);
	pos = input.GetPos(px, py) + 2;
	if (input.GetChannel(pos) == 1) {
		voteA = voteA + 5;
		return;
	}
	px = Clamp(x + 1, 0, input.width);
	py = Clamp(y - 1, 0, input.height);
	pos = input.GetPos(px, py) + 2;
	if (input.GetChannel(pos) == 1) {
		voteA = voteA + 5;
		return;
	}
	px = Clamp(x - 1, 0, input.width);
	py = Clamp(y - 1, 0, input.height);
	pos = input.GetPos(px, py) + 2;
	if (input.GetChannel(pos) == 1) {
		voteB = voteB + 5;
		return;
	}
	px = Clamp(x + 1, 0, input.width);
	py = Clamp(y + 1, 0, input.height);
	pos = input.GetPos(px, py) + 2;
	if (input.GetChannel(pos) == 1) {
		voteB = voteB + 5;
		return;
	}
}

__device__ int traceNodes(int2 nodeCoords, int predecessorNodeDirection, texture_t input) {
	//codes the edge directions
	// N  ... 0x10000000 = 128
	// NE ... 0x01000000 = 64
	// E  ... 0x00100000 = 32
	// SE ... 0x00010000 = 16
	// S  ... 0x00001000 = 8
	// SW ... 0x00000100 = 4
	// W  ... 0x00000010 = 2
	// NW ... 0x00000001 = 1 
	int totalLength = 0; // initial total length

	int2 currentNodeCoords = nodeCoords;
	//check node valence
	uchar* currentNodeValue = input.Texture2D(input.GetUV(currentNodeCoords));


	int2 nextNodeCoords;
	nextNodeCoords.x = nextNodeCoords.y = 0;
	int directionToCurrentNode = 0;

	while (currentNodeValue[2] == 2) {
		//get next neighbor
		int nextNodeDirection = currentNodeValue[1] ^ predecessorNodeDirection;

		switch (nextNodeDirection) {
		case 1:
			//NW
			nextNodeCoords.x = currentNodeCoords.x - 1;
			nextNodeCoords.y = currentNodeCoords.y + 1;
			directionToCurrentNode = 16;
			break;
		case 2:
			//W
			nextNodeCoords.x = currentNodeCoords.x - 1;
			nextNodeCoords.y = currentNodeCoords.y;
			directionToCurrentNode = 32;
			break;
		case 4:
			//SW
			nextNodeCoords.x = currentNodeCoords.x - 1;
			nextNodeCoords.y = currentNodeCoords.y - 1;
			directionToCurrentNode = 64;
			break;
		case 8:
			//S
			nextNodeCoords.x = currentNodeCoords.x;
			nextNodeCoords.y = currentNodeCoords.y - 1;
			directionToCurrentNode = 128;
			break;
		case 16:
			//SE
			nextNodeCoords.x = currentNodeCoords.x + 1;
			nextNodeCoords.y = currentNodeCoords.y - 1;
			directionToCurrentNode = 1;
			break;
		case 32:
			//E
			nextNodeCoords.x = currentNodeCoords.x + 1;
			nextNodeCoords.y = currentNodeCoords.y;
			directionToCurrentNode = 2;
			break;
		case 64:
			//NE
			nextNodeCoords.x = currentNodeCoords.x + 1;
			nextNodeCoords.y = currentNodeCoords.y + 1;
			directionToCurrentNode = 4;
			break;
		case 128:
			//N
			nextNodeCoords.x = currentNodeCoords.x;
			nextNodeCoords.y = currentNodeCoords.y + 1;
			directionToCurrentNode = 8;
			break;
		default:
			//this should not be the case, but just in case ;)
			directionToCurrentNode = predecessorNodeDirection;
			nextNodeCoords = currentNodeCoords;
			return 0;
		}
		//get next node
		nextNodeCoords.x = Clamp(nextNodeCoords.x, 0, input.width);
		nextNodeCoords.y = Clamp(nextNodeCoords.y, 0, input.height);
		currentNodeCoords = nextNodeCoords;
		predecessorNodeDirection = directionToCurrentNode;
		currentNodeValue = input.Texture2D(input.GetUV(currentNodeCoords));
		totalLength++;
	}
	return totalLength;
}

__device__ void voteCurves(int x, int y, int &voteA, int &voteB, texture_t input) {
	int lengthA = 1;
	int lengthB = 1;
	//coordinates for nodes A1,A2,B1,B2
	//A1 B2
	//B1 A2
	int2 A1; A1.x = x - 1; A1.y = y + 1;
	int2 A2; A2.x = x + 1; A2.y = y - 1;
	int2 B1; B1.x = x - 1; B1.y = y - 1;
	int2 B2; B2.x = x + 1; B2.y = y + 1;
	lengthA = lengthA + traceNodes(A1, 16, input);
	lengthA = lengthA + traceNodes(A2, 1, input);
	lengthB = lengthB + traceNodes(B1, 64, input);
	lengthB = lengthB + traceNodes(B2, 4, input);
	//evaluate lengths and vote
	if (lengthA == lengthB) {
		//no one wins
		return;
	}
	else if (lengthA > lengthB) {
		//A wins
		voteA = voteA + lengthA - lengthB;
	}
	else {
		//B wins
		voteB = voteB + lengthB - lengthA;
	}
}

__device__ bool isFullyConnectedCD(texture_t input, int x, int y) {
	int px, py, pos;
	px = Clamp(x, 0, input.width);
	py = Clamp(y + 1, 0, input.height);
	pos = input.GetPos(px, py) + 2;
	//we only have to look at one neighboring edge to determine whether a 2x2 block is fully connected or not
	if (input.GetChannel(pos) == 0) {
		return false;
	}
	else return true;
}

__device__ bool isFullyConnectedD(texture_t input, int x, int y) {
	int px, py, pos;
	px = Clamp(x, 0, input.width);
	py = Clamp(y + 1, 0, input.height);
	pos = input.GetPos(px, py) + 2;
	//examine upper edge
	if (input.GetChannel(pos) == 16) {
		px = Clamp(x + 1, 0, input.width);
		py = Clamp(y, 0, input.height);
		pos = input.GetPos(px, py) + 2;
		//examine right edge
		if (input.GetChannel(pos) == 16) {
			px = Clamp(x, 0, input.width);
			py = Clamp(y - 1, 0, input.height);
			pos = input.GetPos(px, py) + 2;
			//examine lower edge
			if (input.GetChannel(pos) == 16) {
				px = Clamp(x - 1, 0, input.width);
				py = Clamp(y, 0, input.height);
				pos = input.GetPos(px, py) + 2;
				//we could skip the last edgecheck -> this one is fully connected for shure
				if (input.GetChannel(pos) == 16) {
					return true;
				}
				else return false;
			}
			else return false;
		}
		else return false;
	}
	else return false;
}

__global__ void eliminateCrossings_Kernel(texture_t input, texture_t output){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= output.width || y >= output.height)return;
	const int cur = output.GetIndex(x, y);
	uchar fragvalue = input.GetChannel(cur*CHANNEL + 2);
	int voteA, voteB, debugA, debugB, componentSizeA = 2, componentSizeB = 2;
	if (fragvalue == 96) {
		// we are looking at a crossing diagonal
		// 1. check if 2x2 block is fully connected
		if (isFullyConnectedCD(input, x, y)) {
			output.SetPixel(cur*CHANNEL, 0, 0, 0);
			return;
		}
		voteA = 0;
		voteB = 0;
		debugA = 0;
		debugB = 0;
		voteCurves(x, y, voteA, voteB, input);
		voteIslands(x, y, voteA, voteB, input);
		voteSparsePixels(x, y, voteA, voteB, debugA, debugB, input, componentSizeA, componentSizeB);
		//eliminate loser
		debugA = 0;
		debugB = 0;
		if (voteA == voteB) {
			output.SetPixel(cur*CHANNEL, debugB, debugA, 0);
		}
		else if (voteA > voteB) {
			output.SetPixel(cur*CHANNEL, debugB, debugA, 32);
		}
		else {
			output.SetPixel(cur*CHANNEL, debugB, debugA, 64);
		}

	}
	else if (fragvalue == 32 || fragvalue == 64) {
		//we just hit a Diagonal ... it might be fully connected anyways
		if (isFullyConnectedD(input, x, y)) {
			output.SetPixel(cur*CHANNEL, 0, 0, 0);
		}
		else {
			output.SetPixel(cur*CHANNEL, input.Texture2D(x, y));
		}
	}
	else {
		//copy texel
		output.SetPixel(cur*CHANNEL, input.Texture2D(x, y));
	}

}
#pragma endregion SIMILARITY_GRAPH

#pragma region CELL_GRAPH
__device__ bool isContour(texture_t source, int4 LRcolors) {
	uchar* pL = source.Texture2D(LRcolors.x, LRcolors.y);
	uchar* pR = source.Texture2D(LRcolors.z, LRcolors.w);
	float yA = 0.299*pL[2] + 0.587*pL[1] + 0.114*pL[0];
	float uA = 0.493*(pL[0] - yA);
	float vA = 0.877*(pL[2] - yA);
	float yB = 0.299*pR[2] + 0.587*pR[1] + 0.114*pR[0];
	float uB = 0.493*(pR[0] - yB);
	float vB = 0.877*(pR[2] - yB);

	bool isContour = false;
	if (distance(vec3(yA, uA, vA), vec3(yB, uB, vB)) > 100.0) {
		isContour = true;
	}
	return isContour;
}

__device__ CellGraphPixel_t emitPoint(float2 vpos, int4 vneighbors, int vflags, int4 vnNcolors, int4 vnEcolors, int4 vnScolors, int4 vnWcolors) {
	CellGraphPixel_t pixel;
	pixel.pos = vpos;
	pixel.neighbors = vneighbors;
	pixel.flags = vflags;
	pixel.nNcolors = vnNcolors;
	pixel.nEcolors = vnEcolors;
	pixel.nScolors = vnScolors;
	pixel.nWcolors = vnWcolors;
	return pixel;
}

__device__ int getNeighborIndex(int x, int y, int dir, int targetSector, texture_t source) {
	int index = -1;
	int2 centralPos = ivec2(x, y); //centralPos ... current 2x2 cell x,y position (from 0,0 to textureSize(pixelArt)-1).
	int dy = source.width - 1;

	if ((dir & NORTH_G) == NORTH_G) {
		index = ((centralPos.y + 1) * dy + centralPos.x) * 2 + targetSector;
	}
	else if ((dir & EAST_G) == EAST_G) {
		index = (centralPos.y * dy + centralPos.x + 1) * 2 + targetSector;
	}
	else if ((dir & SOUTH_G) == SOUTH_G) {
		index = ((centralPos.y - 1) * dy + centralPos.x) * 2 + targetSector;
	}
	else if ((dir & WEST_G) == WEST_G) {
		index = (centralPos.y * dy + centralPos.x - 1) * 2 + targetSector;
	}
	else if ((dir & CENTER) == CENTER) {
		index = (centralPos.y * dy + centralPos.x) * 2 + targetSector;
	}
	return index;
}
__device__ float2 calcAdjustedPoint(float2 p0, float2 p1, float2 p2) {
	return vec2(0.125*p0.x + 0.75*p1.x + 0.125*p2.x, 0.125*p0.y + 0.75*p1.y + 0.125*p2.y);
}

__device__ bool checkForCorner(float2 spline1, float2 spline2) {
	float dp = dot(normalize(spline1), normalize(spline2));
	if (dp > -0.7072 && dp < -0.7070){
		return true;
	}
	else if (dp > -0.3163 && dp < -0.3161) {
		return true;
	}
	else if (dp > -0.0001 && dp < 0.0001) {
		return true;
	}
	return false;
}

__global__ void FullCellGraphCOnstruction_Kernel(texture_t input, texture_t source, CellGraphBudffer_t cellBuffer){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= source.width - 1 || y >= source.height - 1)return;
	const int cur = y*(source.width - 1) + x;
	int2 sGDCoords = ivec2(x * 2 + 2, y * 2 + 2);

	int px = Clamp(sGDCoords.x, 0, input.width), py = Clamp(sGDCoords.y, 0, input.height);
	int eCenter = input.Texture2D(px, py)[2];
	px = Clamp(sGDCoords.x, 0, input.width);
	py = Clamp(sGDCoords.y + 1, 0, input.height);
	int eNorth = input.Texture2D(px, py)[2];
	px = Clamp(sGDCoords.x, 0, input.width);
	py = Clamp(sGDCoords.y + 2, 0, input.height);
	int eNorthCenter = input.Texture2D(px, py)[2];
	px = Clamp(sGDCoords.x + 1, 0, input.width);
	py = Clamp(sGDCoords.y, 0, input.height);
	int eEast = input.Texture2D(px, py)[2];
	px = Clamp(sGDCoords.x + 2, 0, input.width);
	py = Clamp(sGDCoords.y, 0, input.height);
	int eEastCenter = input.Texture2D(px, py)[2];
	px = Clamp(sGDCoords.x, 0, input.width);
	py = Clamp(sGDCoords.y - 1, 0, input.height);
	int eSouth = input.Texture2D(px, py)[2];
	px = Clamp(sGDCoords.x, 0, input.width);
	py = Clamp(sGDCoords.y - 2, 0, input.height);
	int eSouthCenter = input.Texture2D(px, py)[2];
	px = Clamp(sGDCoords.x - 1, 0, input.width);
	py = Clamp(sGDCoords.y, 0, input.height);
	int eWest = input.Texture2D(px, py)[2];
	px = Clamp(sGDCoords.x - 2, 0, input.width);
	py = Clamp(sGDCoords.y, 0, input.height);
	int eWestCenter = input.Texture2D(px, py)[2];

	float2 v0_pos = vec2(-1.0, -1.0);
	int4 v0_neighbors = ivec4(-1, -1, -1, -1);
	int v0_flags = 0;
	int4 v0_nNcolors = ivec4(-1, -1, -1, -1);
	int4 v0_nEcolors = ivec4(-1, -1, -1, -1);
	int4 v0_nScolors = ivec4(-1, -1, -1, -1);
	int4 v0_nWcolors = ivec4(-1, -1, -1, -1);
	int v1_valence = 0;
	float2 v1_pos = vec2(-1.0, -1.0);
	int4 v1_neighbors = ivec4(-1, -1, -1, -1);
	int v1_flags = 0;
	int4 v1_nNcolors = ivec4(-1, -1, -1, -1);
	int4 v1_nEcolors = ivec4(-1, -1, -1, -1);
	int4 v1_nScolors = ivec4(-1, -1, -1, -1);
	int4 v1_nWcolors = ivec4(-1, -1, -1, -1);
	bool ignoreN = false;
	bool ignoreE = false;
	bool ignoreS = false;
	bool ignoreW = false;
	if (y > source.height - 3) ignoreN = true;
	if (x > source.width - 3) ignoreE = true;
	if (y < 1.0) ignoreS = true;
	if (x < 1.0) ignoreW = true;
	bool neighborsFound = false;
	bool nNeighborsFound = false;
	bool wNeighborsFound = false;
	bool sNeighborsFound = false;
	bool eNeighborsFound = false;
	int neighborCount = 0;
	int nNeighborIndex = -1;
	int wNeighborIndex = -1;
	int sNeighborIndex = -1;
	int eNeighborIndex = -1;
	float2 nVector = vec2(0.0, 0.0);
	float2 eVector = vec2(0.0, 0.0);
	float2 sVector = vec2(0.0, 0.0);
	float2 wVector = vec2(0.0, 0.0);

	if (!ignoreN && eNorth == 0) {
		nNeighborsFound = true;
		neighborsFound = true;
		neighborCount++;
		if (eNorthCenter == EDGE_DIAGONAL_ULLR) {
			nNeighborIndex = getNeighborIndex(x, y, NORTH_G, 0, source);
			nVector = vec2(-0.25, 0.75);
		}
		else if (eNorthCenter == EDGE_DIAGONAL_LLUR) {
			nNeighborIndex = getNeighborIndex(x, y, NORTH_G, 1, source);
			nVector = vec2(0.25, 0.75);
		}
		else {
			nNeighborIndex = getNeighborIndex(x, y, NORTH_G, 0, source);
			nVector = vec2(0.0, 1.0);
		}
	}
	if (!ignoreW && eWest == 0) {
		wNeighborsFound = true;
		neighborsFound = true;
		neighborCount++;
		if (eWestCenter == EDGE_DIAGONAL_ULLR) {
			wNeighborIndex = getNeighborIndex(x, y, WEST_G, 1, source);
			wVector = vec2(-0.75, 0.25);
		}
		else if (eWestCenter == EDGE_DIAGONAL_LLUR) {
			wNeighborIndex = getNeighborIndex(x, y, WEST_G, 1, source);
			wVector = vec2(-0.75, -0.25);
		}
		else {
			wNeighborIndex = getNeighborIndex(x, y, WEST_G, 0, source);
			wVector = vec2(-1.0, 0.0);
		}
	}

	if (!ignoreS && eSouth == 0) {
		sNeighborsFound = true;
		neighborsFound = true;
		neighborCount++;
		if (eSouthCenter == EDGE_DIAGONAL_ULLR) {
			sNeighborIndex = getNeighborIndex(x, y, SOUTH_G, 1, source);
			sVector = vec2(0.25, -0.75);
		}
		else if (eSouthCenter == EDGE_DIAGONAL_LLUR) {
			sNeighborIndex = getNeighborIndex(x, y, SOUTH_G, 0, source);
			sVector = vec2(-0.25, -0.75);
		}
		else {
			sNeighborIndex = getNeighborIndex(x, y, SOUTH_G, 0, source);
			sVector = vec2(0.0, -1.0);
		}
	}

	if (!ignoreE && eEast == 0) {
		eNeighborsFound = true;
		neighborsFound = true;
		neighborCount++;

		if (eEastCenter == EDGE_DIAGONAL_ULLR) {
			eNeighborIndex = getNeighborIndex(x, y, EAST_G, 0, source);
			eVector = vec2(0.75, -0.25);
		}
		else if (eEastCenter == EDGE_DIAGONAL_LLUR) {
			eNeighborIndex = getNeighborIndex(x, y, EAST_G, 0, source);
			eVector = vec2(0.75, 0.25);
		}
		else {
			eNeighborIndex = getNeighborIndex(x, y, EAST_G, 0, source);
			eVector = vec2(1.0, 0.0);
		}
	}

	if (neighborsFound) {
		int2 LLPixelColorIndex = ivec2(x, y);
		int2 ULPixelColorIndex = ivec2(x, y + 1);
		int2 LRPixelColorIndex = ivec2(x + 1, y);
		int2 URPixelColorIndex = ivec2(x + 1, y + 1);
		float2 centerPos = vec2(x + 0.5f, y + 0.5f);
		if (eCenter == EDGE_DIAGONAL_ULLR) {
			bool twoNeighbors = true;
			v0_pos = vec2(XOFFSET_CLL + centerPos.x, centerPos.y + YOFFSET_CLL);
			int sIndex = sNeighborIndex;
			int wIndex = wNeighborIndex;
			v0_nScolors = ivec4(LRPixelColorIndex, LLPixelColorIndex);
			v0_nWcolors = ivec4(LLPixelColorIndex, ULPixelColorIndex);
			if (sNeighborsFound) {
				v0_flags = HAS_SOUTHERN_NEIGHBOR | HAS_SOUTHERN_SPLINE;
			}
			else twoNeighbors = false;
			if (wNeighborsFound) {
				v0_flags = v0_flags | HAS_WESTERN_NEIGHBOR | HAS_WESTERN_SPLINE;
			}
			else twoNeighbors = false;
			if (twoNeighbors) {
				if (checkForCorner(vec2(sVector.x - XOFFSET_CLL, sVector.y - YOFFSET_CLL), vec2(wVector.x - XOFFSET_CLL, wVector.y - YOFFSET_CLL))) {
					v0_flags = v0_flags | DONT_OPTIMIZE_S | DONT_OPTIMIZE_W;
				}
			}

			v0_neighbors = ivec4(-1, -1, sIndex, wIndex);
			twoNeighbors = true;
			v1_pos = vec2(centerPos.x + XOFFSET_CUR, centerPos.y + YOFFSET_CUR);
			int nIndex = nNeighborIndex;
			int eIndex = eNeighborIndex;
			v1_nNcolors = ivec4(ULPixelColorIndex, URPixelColorIndex);
			v1_nEcolors = ivec4(URPixelColorIndex, LRPixelColorIndex);
			if (nNeighborsFound) {
				v1_flags = HAS_NORTHERN_NEIGHBOR | HAS_NORTHERN_SPLINE;
			}
			else twoNeighbors = false;
			if (eNeighborsFound) {
				v1_flags = v1_flags | HAS_EASTERN_NEIGHBOR | HAS_EASTERN_SPLINE;
			}
			else twoNeighbors = false;

			if (twoNeighbors) {
				if (checkForCorner(vec2(nVector.x - XOFFSET_CUR, nVector.y - YOFFSET_CUR), vec2(eVector.x - XOFFSET_CUR, eVector.y - YOFFSET_CUR))) {
					v1_flags = v1_flags | DONT_OPTIMIZE_N | DONT_OPTIMIZE_E;
				}
			}

			v1_neighbors = ivec4(nIndex, eIndex, -1, -1);
		}
		else if (eCenter == EDGE_DIAGONAL_LLUR) {
			bool twoNeighbors = true;
			v0_pos = vec2(centerPos.x + XOFFSET_CUL, centerPos.y + YOFFSET_CUL);
			int nIndex = nNeighborIndex;
			int wIndex = wNeighborIndex;
			v0_nNcolors = ivec4(ULPixelColorIndex, URPixelColorIndex);
			v0_nWcolors = ivec4(LLPixelColorIndex, ULPixelColorIndex);
			if (nNeighborsFound) {
				v0_flags = HAS_NORTHERN_NEIGHBOR | HAS_NORTHERN_SPLINE;
			}
			else twoNeighbors = false;
			if (wNeighborsFound) {
				v0_flags = v0_flags | HAS_WESTERN_NEIGHBOR | HAS_WESTERN_SPLINE;
			}
			else twoNeighbors = false;
			if (twoNeighbors) {
				if (checkForCorner(vec2(nVector.x - XOFFSET_CUL, nVector.y - YOFFSET_CUL), vec2(wVector.x - XOFFSET_CUL, wVector.y - YOFFSET_CUL))) {
					v0_flags = v0_flags | DONT_OPTIMIZE_N | DONT_OPTIMIZE_W;
				}
			}
			v0_neighbors = ivec4(nIndex, -1, -1, wIndex);

			twoNeighbors = true;
			v1_pos = vec2(centerPos.x + XOFFSET_CLR, centerPos.y + YOFFSET_CLR);
			int sIndex = sNeighborIndex;
			int eIndex = eNeighborIndex;
			v1_nScolors = ivec4(LRPixelColorIndex, LLPixelColorIndex);
			v1_nEcolors = ivec4(URPixelColorIndex, LRPixelColorIndex);
			if (sNeighborsFound) {
				v1_flags = HAS_SOUTHERN_NEIGHBOR | HAS_SOUTHERN_SPLINE;
			}
			else twoNeighbors = false;
			if (eNeighborsFound) {
				v1_flags = v1_flags | HAS_EASTERN_NEIGHBOR | HAS_EASTERN_SPLINE;
			}
			else twoNeighbors = false;
			if (twoNeighbors) {
				if (checkForCorner(vec2(sVector.x - XOFFSET_CLR, sVector.y - YOFFSET_CLR), vec2(eVector.x - XOFFSET_CLR, eVector.y - YOFFSET_CLR))) {
					v1_flags = v1_flags | DONT_OPTIMIZE_S | DONT_OPTIMIZE_E;
				}
			}
			v1_neighbors = ivec4(-1, eIndex, sIndex, -1);

		}
		else {
			v0_pos = centerPos;
			int nIndex = nNeighborIndex;
			int eIndex = eNeighborIndex;
			int sIndex = sNeighborIndex;
			int wIndex = wNeighborIndex;
			v0_nNcolors = ivec4(ULPixelColorIndex, URPixelColorIndex);
			v0_nEcolors = ivec4(URPixelColorIndex, LRPixelColorIndex);
			v0_nScolors = ivec4(LRPixelColorIndex, LLPixelColorIndex);
			v0_nWcolors = ivec4(LLPixelColorIndex, ULPixelColorIndex);
			if (nNeighborsFound) {
				v0_flags = HAS_NORTHERN_NEIGHBOR;
			}
			if (eNeighborsFound) {
				v0_flags = v0_flags | HAS_EASTERN_NEIGHBOR;
			}
			if (sNeighborsFound) {
				v0_flags = v0_flags | HAS_SOUTHERN_NEIGHBOR;
			}
			if (wNeighborsFound) {
				v0_flags = v0_flags | HAS_WESTERN_NEIGHBOR;
			}

			if (neighborCount == 2) {
				if (nNeighborsFound) {
					v0_flags = v0_flags | HAS_NORTHERN_SPLINE;
				}
				if (eNeighborsFound) {
					v0_flags = v0_flags | HAS_EASTERN_SPLINE;
				}
				if (sNeighborsFound) {
					v0_flags = v0_flags | HAS_SOUTHERN_SPLINE;
				}
				if (wNeighborsFound) {
					v0_flags = v0_flags | HAS_WESTERN_SPLINE;
				}
			}
			else if (neighborCount == 3) {
				int contours = 0;
				int contourCount = 0;
				float2 p[3];
				if (nNeighborsFound) {
					if (isContour(source, v0_nNcolors)) {
						p[0] = nVector;
						contourCount++;
						contours = HAS_NORTHERN_SPLINE;
					}
				}
				if (eNeighborsFound) {
					if (isContour(source, v0_nEcolors)) {
						p[contourCount] = eVector;
						contourCount++;
						contours = contours | HAS_EASTERN_SPLINE;
					}
				}
				if (sNeighborsFound) {
					if (isContour(source, v0_nScolors)) {
						p[contourCount] = sVector;
						contourCount++;
						contours = contours | HAS_SOUTHERN_SPLINE;
					}
				}
				if (wNeighborsFound) {
					if (isContour(source, v0_nWcolors)) {
						p[contourCount] = wVector;
						contourCount++;
						contours = contours | HAS_WESTERN_SPLINE;
					}
				}
				if (contourCount == 2) {
					v0_flags = v0_flags | contours | HAS_CORRECTED_POSITION;
					v1_pos = calcAdjustedPoint(vec2(centerPos.x + p[0].x, centerPos.y + p[0].y), centerPos, vec2(centerPos.x + p[1].x, centerPos.y + p[1].y));
					v1_flags = -1;
				}
				else {
					if (nNeighborsFound && sNeighborsFound) {
						v0_flags = v0_flags | HAS_NORTHERN_SPLINE | HAS_SOUTHERN_SPLINE | HAS_CORRECTED_POSITION;
						v1_pos = calcAdjustedPoint(vec2(centerPos.x + nVector.x, centerPos.y + nVector.y), centerPos, vec2(centerPos.x + sVector.x, centerPos.y + sVector.y));
						v1_flags = -1;
					}
					else {
						v0_flags = v0_flags | HAS_EASTERN_SPLINE | HAS_WESTERN_SPLINE | HAS_CORRECTED_POSITION;
						v1_pos = calcAdjustedPoint(vec2(centerPos.x + eVector.x, centerPos.y + eVector.y), centerPos, vec2(centerPos.x + wVector.x, centerPos.y + wVector.y));
						v1_flags = -1;
					}
				}
			}
			v0_neighbors = ivec4(nIndex, eIndex, sIndex, wIndex);
		}

	}
	CellGraphPixel_t pixel1 = emitPoint(v0_pos, v0_neighbors, v0_flags, v0_nNcolors, v0_nEcolors, v0_nScolors, v0_nWcolors);
	CellGraphPixel_t pixel2 = emitPoint(v1_pos, v1_neighbors, v1_flags, v1_nNcolors, v1_nEcolors, v1_nScolors, v1_nWcolors);
	cellBuffer.pos[cur * 2] = pixel1.pos;
	cellBuffer.vneighbors[cur * 2] = pixel1.neighbors;
	cellBuffer.vflags[cur * 2] = pixel1.flags;
	cellBuffer.vnEcolors[cur * 2] = pixel1.nEcolors;
	cellBuffer.vnNcolors[cur * 2] = pixel1.nNcolors;
	cellBuffer.vnWcolors[cur * 2] = pixel1.nWcolors;
	cellBuffer.vnScolors[cur * 2] = pixel1.nScolors;
	cellBuffer.pos[cur * 2 + 1] = pixel2.pos;
	cellBuffer.vneighbors[cur * 2 + 1] = pixel2.neighbors;
	cellBuffer.vflags[cur * 2 + 1] = pixel2.flags;
	cellBuffer.vnEcolors[cur * 2 + 1] = pixel2.nEcolors;
	cellBuffer.vnNcolors[cur * 2 + 1] = pixel2.nNcolors;
	cellBuffer.vnWcolors[cur * 2 + 1] = pixel2.nWcolors;
	cellBuffer.vnScolors[cur * 2 + 1] = pixel2.nScolors;
}
#pragma endregion CELL_GRAPH

#pragma region OPTIMIZEENERGY
__device__ int getNeighborIndex(int4* knotNeighbors, int sourceIndex, int dir, int targetSector) {
	int index = -1;

	if ((dir & NORTH_G) == NORTH_G) {
		//index = texelFetch(knotNeighbors, sourceIndex).x;
		index = knotNeighbors[sourceIndex].x;
	}
	else if ((dir & EAST_G) == EAST_G) {
		index = knotNeighbors[sourceIndex].y;
	}
	else if ((dir & SOUTH_G) == SOUTH_G) {
		index = knotNeighbors[sourceIndex].z;
	}
	else if ((dir & WEST_G) == WEST_G) {
		index = knotNeighbors[sourceIndex].w;
	}
	return index;
}

__device__ float calcPositionalEnergy(float2 pNew, float2 pOld) {
	float dist = 2.5*distance(pNew, pOld);
	return dist*dist*dist*dist;
}

__device__ float2 calcGradient(float2 node1, float2 node2, float2 node3) {
	return 8 * node2 - 4 * node1 - 4 * node3;
}

__device__ float calcSegmentCurveEnergy(float2 node1, float2 node2, float2 node3) {
	float2 tmp = node1 - 2 * node2 + node3;
	return tmp.x*tmp.x + tmp.y*tmp.y;
}

__device__ float3 findBracket(float2 pos, float2 splineNeighbors[2], float2 gradient) {
	float ulim, u, r, q, fu, dum, qr;
	float ax = BRACKET_SEARCH_A;
	float bx = BRACKET_SEARCH_B;
	float2 pOpt = pos - gradient * ax;
	float fa = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
	pOpt = pos - gradient * bx;
	float fb = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
	if (fb > fa) {
		//switch roles of a and b so we can go downhill from a to b
		dum = ax;
		ax = bx;
		bx = dum;
		dum = fb;
		fb = fa;
		fa = dum;
	}
	//first guess for c
	float cx = bx + GOLD * (bx - ax);
	pOpt = pos - gradient * cx;
	float fc = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
	//find bracket
	while (fb > fc) {
		r = (bx - ax) * (fb - fc);
		q = (bx - cx) * (fb - fa);
		qr = q - r;
		u = bx - ((bx - cx) * q - (bx - ax) * r) / (2.0 * sign(qr) * max(abs(qr), TINY));
		ulim = bx + GLIMIT * (cx - bx);
		if ((bx - u) * (u - cx) > 0.0) {
			pOpt = pos - gradient * u;
			fu = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
			if (fu < fc) {
				return vec3(bx, u, cx);
			}
			else if (fu > fb) {
				return vec3(ax, bx, u);
			}
			u = cx + GOLD * (cx - bx);
			pOpt = pos - gradient * u;
			fu = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
		}
		else if ((cx - u) * (u - ulim) > 0.0) {
			pOpt = pos - gradient * u;
			fu = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
			if (fu < fc) {
				dum = cx + GOLD * (cx - bx);
				bx = cx;
				cx = u;
				u = dum;
				fb = fc;
				fc = fu;
				pOpt = pos - gradient * u;
				fu = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
			}
		}
		else if ((u - ulim) * (ulim - cx) >= 0.0) {
			u = ulim;
			pOpt = pos - gradient * u;
			fu = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
		}
		else {
			u = cx + GOLD * (cx - bx);
			pOpt = pos - gradient * u;
			fu = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
		}
		ax = bx;
		bx = cx;
		cx = u;
		fa = fb;
		fb = fc;
		fc = fu;
	}
	return vec3(ax, bx, cx);
}

__device__ float3 searchOffset(float2 pos, float2 splineNeighbors[2]) {
	float2 gradient = calcGradient(splineNeighbors[0], pos, splineNeighbors[1]);
	if (length(gradient) > 0.0) {
		gradient = normalize(gradient);
	}
	else return vec3(0.0, 0.0, 0.0);

	float3 bracket = findBracket(pos, splineNeighbors, gradient);

	//float R = 0.61803399;
	//float C = 1 - R;

	//float tol = 10^-4;

	float x0 = bracket.x;
	float x1 = 0;
	float x2 = 0;
	float x3 = bracket.z;


	if (abs(bracket.z - bracket.y) > abs(bracket.y - bracket.x)) {
		x1 = bracket.y;
		x2 = bracket.y + C * (bracket.z - bracket.y);
	}
	else {
		x1 = bracket.y - C * (bracket.y - bracket.x);
		x2 = bracket.y;
	}
	float2 pOpt = vec2(0.0, 0.0);
	pOpt = pos - gradient * x1;
	float f1 = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);

	pOpt = pos - gradient * x2;
	float f2 = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
	int counter = 0;
	float fx;
	while (abs(x3 - x0) > TOL * (abs(x1) + abs(x2)) && (counter < LIMIT_SEARCH_ITERATIONS)) {
		counter = counter + 1;
		if (f2 < f1) {
			x0 = x1;
			x1 = x2;
			x2 = R * x1 + C * x3;
			pOpt = pos - gradient * x2;
			fx = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
			f1 = f2;
			f2 = fx;
		}
		else {
			x3 = x2;
			x2 = x1;
			x1 = R * x2 + C * x0;
			pOpt = pos - gradient * x1;
			fx = calcSegmentCurveEnergy(splineNeighbors[0], pOpt, splineNeighbors[1]) + calcPositionalEnergy(pOpt, pos);
			f2 = f1;
			f1 = fx;
		}
	}
	float offset = 0;
	if (f1 < f2) {
		offset = x1;
	}
	else {
		offset = x2;
	}
	return vec3(gradient, offset);
}

__global__ void OptimizeEnergy(CellGraphBudffer_t cellbuffer, int width, int bufferSize){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int cur = y*width + x;
	if (cur >= bufferSize)return;
	int height = bufferSize / width / 2;

	float2 optimizedPos = cellbuffer.pos[cur];
	int4 neighbors = cellbuffer.vneighbors[cur];
	int flags = cellbuffer.vflags[cur];
	if (flags > 16) {
		if (flags < 512) {
			float2 splineNeighbors[2];
			int splineCount = 0;
			bool splineNoOpt = false;
			float2 splineNeighborsNeighbors[2];
			splineNeighborsNeighbors[0] = splineNeighborsNeighbors[1] = vec2(0.0, 0.0);
			bool neighborHasNeighbor[2] = { false, false };

			//spline neighbor indicaors
			bool hasNorthernSpline = (flags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE;
			bool hasEasternSpline = (flags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE;
			bool hasSouthernSpline = (flags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE;
			bool hasWesternSpline = (flags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE;

			if (hasNorthernSpline) {
				int neighborflags = cellbuffer.vflags[neighbors.x];
				if (((flags & DONT_OPTIMIZE_N) == DONT_OPTIMIZE_N)
					|| ((neighborflags & DONT_OPTIMIZE_S) == DONT_OPTIMIZE_S)) {
					splineNoOpt = true;
				}
				if (!splineNoOpt) {
					if (((neighborflags & HAS_CORRECTED_POSITION) == HAS_CORRECTED_POSITION)
						&& !((neighborflags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE)) {
						//our neighbour is an endpoint on a t-junction and needs to be adjusted
						splineNeighbors[splineCount] = cellbuffer.pos[neighbors.x + 1];
						splineCount++;
					}
					else {
						splineNeighbors[splineCount] = cellbuffer.pos[neighbors.x];
						//see if the northern point has a neighbor - this one would be influenced by the optimization
						if ((neighborflags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE) {
							//get neighbors neighbors flag
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.x, NORTH_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_S) == 0)) {
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						else if ((neighborflags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE) {
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.x, EAST_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_W) == 0))
							{
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						else if ((neighborflags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE) {
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.x, WEST_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_E) == 0))
							{
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						splineCount++;
					}
				}
			}
			if (hasEasternSpline) {
				int neighborflags = cellbuffer.vflags[neighbors.y];
				if (((flags & DONT_OPTIMIZE_E) == DONT_OPTIMIZE_E) || ((neighborflags & DONT_OPTIMIZE_W) == DONT_OPTIMIZE_W)) {
					splineNoOpt = true;
				}
				if (!splineNoOpt) {
					if (((neighborflags & HAS_CORRECTED_POSITION) == HAS_CORRECTED_POSITION)
						&& !((neighborflags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE)) {
						//our neighbour is an endpoint on a t-junction and needs to be adjusted
						splineNeighbors[splineCount] = cellbuffer.pos[neighbors.y + 1];
						splineCount++;
					}
					else {
						splineNeighbors[splineCount] = cellbuffer.pos[neighbors.y];

						if ((neighborflags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE) {
							//get neighbors neighbors flag
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.y, NORTH_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_S) == 0))
							{
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						else if ((neighborflags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE) {
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.y, EAST_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_W) == 0))
							{
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						else if ((neighborflags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE) {
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.y, SOUTH_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_N) == 0))
							{
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						splineCount++;
					}
				}
			}
			if (hasSouthernSpline) {
				int neighborflags = cellbuffer.vflags[neighbors.z];
				if (((flags & DONT_OPTIMIZE_S) == DONT_OPTIMIZE_S) || ((neighborflags & DONT_OPTIMIZE_N) == DONT_OPTIMIZE_N)) {
					splineNoOpt = true;
				}
				if (!splineNoOpt) {
					if (((neighborflags & HAS_CORRECTED_POSITION) == HAS_CORRECTED_POSITION)
						&& !((neighborflags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE)) {
						//our neighbour is an endpoint on a t-junction and needs to be adjusted
						splineNeighbors[splineCount] = cellbuffer.pos[neighbors.z + 1];
						splineCount++;
					}
					else {
						splineNeighbors[splineCount] = cellbuffer.pos[neighbors.z];

						if ((neighborflags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE) {
							//get neighbors neighbors flag
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.z, WEST_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_E) == 0))
							{
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						else if ((neighborflags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE) {
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.z, EAST_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_W) == 0)) {
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						else if ((neighborflags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE) {
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.z, SOUTH_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_N) == 0)) {
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}

						}
						splineCount++;
					}
				}
			}
			if (hasWesternSpline) {
				int neighborflags = cellbuffer.vflags[neighbors.w];
				if (((flags & DONT_OPTIMIZE_W) == DONT_OPTIMIZE_W) || ((neighborflags & DONT_OPTIMIZE_E) == DONT_OPTIMIZE_E)) {
					splineNoOpt = true;
				}
				if (!splineNoOpt) {
					if (((neighborflags & HAS_CORRECTED_POSITION) == HAS_CORRECTED_POSITION)
						&& !((neighborflags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE)) {
						//our neighbour is an endpoint on a t-junction and needs to be adjusted
						splineNeighbors[splineCount] = cellbuffer.pos[neighbors.w + 1];
						splineCount++;
					}
					else {
						splineNeighbors[splineCount] = cellbuffer.pos[neighbors.w];

						if ((neighborflags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE) {
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.w, NORTH_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_S) == 0))
							{
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						else if ((neighborflags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE) {
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.w, WEST_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_E) == 0))
							{
								//take this one
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						else if ((neighborflags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE) {
							int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, neighbors.w, SOUTH_G, 0);
							int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
							if (((neighborsNeighborflags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE)
								&& ((neighborsNeighborflags & DONT_OPTIMIZE_N) == 0))
							{
								splineNeighborsNeighbors[splineCount] = cellbuffer.pos[neighborsNeighborIndex];
								neighborHasNeighbor[splineCount] = true;
							}
						}
						splineCount++;
					}
				}
			}

			if ((splineCount == 2) && (!splineNoOpt)) {
				float3 shift = searchOffset(optimizedPos, splineNeighbors);
				optimizedPos = optimizedPos - vec2(shift.x, shift.y)*shift.z;
			}
		}
	}
	cellbuffer.pos[cur] = optimizedPos;
}

__global__ void UpdateCorrectedPositions(CellGraphBudffer_t cellbuffer, int width, int bufferSize){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int cur = y*width + x;
	if (cur >= bufferSize)return;
	float2 pos = cellbuffer.pos[cur];
	int flags = cellbuffer.vflags[cur];
	float2 optimizedPos = cellbuffer.pos[cur];
	if (flags == -1) {
		//get position, flags and neighborhood indices from parent vertex
		int id = cur;
		float2 parentPosition = cellbuffer.pos[id];
		int parentFlags = cellbuffer.vflags[id];
		int4 parentNeighborIndices = cellbuffer.vneighbors[id];
		float2 splinePoints[2] = { vec2(0.0, 0.0), vec2(0.0, 0.0) };

		int count = 0;
		if ((parentFlags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE) {
			splinePoints[count] = cellbuffer.pos[parentNeighborIndices.x];
			count++;
		}
		if ((parentFlags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE) {
			splinePoints[count] = cellbuffer.pos[parentNeighborIndices.y];
			count++;
		}
		if ((parentFlags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE) {
			splinePoints[count] = cellbuffer.pos[parentNeighborIndices.z];
			count++;
		}
		if ((parentFlags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE) {
			splinePoints[count] = cellbuffer.pos[parentNeighborIndices.w];
			count++;
		}
		//optimizedPos = calcAdjustedPoint(texelFetch(indexedCellPositions, parentNeighborIndices.z).rg, parentPosition, texelFetch(indexedCellPositions, parentNeighborIndices.y).rg);
		if (count == 2) {
			optimizedPos = calcAdjustedPoint(splinePoints[0], parentPosition, splinePoints[1]);
		}
		else {
			optimizedPos = parentPosition;
		}
	}
	else {
		optimizedPos = pos;
	}
	cellbuffer.pos[cur] = optimizedPos;
}
#pragma endregion OPTIMIZEENERGY

#pragma region RASTERIZER
__device__ int getNeighborIndex(int4* neighborIndices, int sourceIndex, int dir) {
	int index = -1;
	if ((dir & NORTH) == NORTH) {
		index = neighborIndices[sourceIndex].x;
	}
	else if ((dir & EAST) == EAST) {
		index = neighborIndices[sourceIndex].y;
	}
	else if ((dir & SOUTH) == SOUTH) {
		index = neighborIndices[sourceIndex].z;
	}
	else if ((dir & WEST) == WEST) {
		index = neighborIndices[sourceIndex].w;
	}
	return index;
}

__device__ float2 calcSplinePoint(float2 p0, float2 p1, float2 p2, float t) {
	float t2 = 0.5*t*t;
	float a = t2 - t + 0.5;
	float b = -2.0 * t2 + t + 0.5;
	return a*p0 + b*p1 + t2*p2;
}

__device__ float2 calcSplineEndPoint(float2 p0, float2 p1, float t) {
	float t2 = 0.5*t*t;
	return (-t2 + 1)*p0 + t2*p1;
}

__device__ bool intersects(float2 lineApointA, float2 lineApointB, float2 lineBpointA, float2 lineBpointB) {
	float2 r = lineApointB - lineApointA;
	float2 s = lineBpointB - lineBpointA;
	//r.y = -r.y;
	//s.y = -s.y;
	float rXs = r.x * s.y - r.y * s.x;
	if (rXs == 0.0) {
		return false;
	}
	float2 ba = lineBpointA - lineApointA;
	//ba.y = -ba.y;
	float t = (ba.x * s.y - ba.y * s.x) / rXs;
	if ((t < 0.0) || (t > 1.0)) {
		return false;
	}
	float u = (ba.x * r.y - ba.y * r.x) / rXs;
	if ((u < 0.0) || (u > 1.0)) {
		return false;
	}
	return true;
}

__device__ int computeValence(int flags) {
	int valence = 0;
	if ((flags & HAS_NORTHERN_NEIGHBOR) == HAS_NORTHERN_NEIGHBOR)
		valence++;
	if ((flags & HAS_EASTERN_NEIGHBOR) == HAS_EASTERN_NEIGHBOR)
		valence++;
	if ((flags & HAS_SOUTHERN_NEIGHBOR) == HAS_SOUTHERN_NEIGHBOR)
		valence++;
	if ((flags & HAS_WESTERN_NEIGHBOR) == HAS_WESTERN_NEIGHBOR)
		valence++;
	return valence;
}

__device__ int2 getCPs(CellGraphBudffer_t cellbuffer, int node0neighborIndex, int dir) {
	int2 cpArray = ivec2(node0neighborIndex, -1);
	int3 checkFwd = ivec3(0, 0, 0);
	int2 chkdirs = ivec2(0, 0);
	int checkBack = 0;
	if (dir == NORTH) {
		checkFwd.x = HAS_NORTHERN_SPLINE;
		checkFwd.y = HAS_EASTERN_SPLINE;
		checkFwd.z = HAS_WESTERN_SPLINE;
		checkBack = HAS_SOUTHERN_SPLINE;
		chkdirs.x = EAST;
		chkdirs.y = WEST;
	}
	else if (dir == EAST) {
		checkFwd.x = HAS_EASTERN_SPLINE;
		checkFwd.y = HAS_SOUTHERN_SPLINE;
		checkFwd.z = HAS_NORTHERN_SPLINE;
		checkBack = HAS_WESTERN_SPLINE;
		chkdirs.x = SOUTH;
		chkdirs.y = NORTH;
	}
	else if (dir == SOUTH) {
		checkFwd.x = HAS_SOUTHERN_SPLINE;
		checkFwd.y = HAS_WESTERN_SPLINE;
		checkFwd.z = HAS_EASTERN_SPLINE;
		checkBack = HAS_NORTHERN_SPLINE;
		chkdirs.x = WEST;
		chkdirs.y = EAST;
	}
	else if (dir == WEST) {
		checkFwd.x = HAS_WESTERN_SPLINE;
		checkFwd.y = HAS_NORTHERN_SPLINE;
		checkFwd.z = HAS_SOUTHERN_SPLINE;
		checkBack = HAS_EASTERN_SPLINE;
		chkdirs.x = NORTH;
		chkdirs.y = SOUTH;
	}

	int node0neighborFlags = cellbuffer.vflags[node0neighborIndex];
	//check for t-junktion
	if ((node0neighborFlags & checkBack) == checkBack) {
		//the spline continues through the next control point
		//get next spline control point to compute segment extension
		if ((node0neighborFlags & checkFwd.x) == checkFwd.x) {
			int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, node0neighborIndex, dir);
			int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
			if ((neighborsNeighborflags & HAS_CORRECTED_POSITION) == HAS_CORRECTED_POSITION) {
				cpArray.y = neighborsNeighborIndex + 1;
			}
			else {
				cpArray.y = neighborsNeighborIndex;
			}
		}
		else if ((node0neighborFlags & checkFwd.y) == checkFwd.y) {
			int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, node0neighborIndex, chkdirs.x);
			int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
			if ((neighborsNeighborflags & HAS_CORRECTED_POSITION) == HAS_CORRECTED_POSITION) {
				cpArray.y = neighborsNeighborIndex + 1;
			}
			else {
				cpArray.y = neighborsNeighborIndex;
			}
		}
		else if ((node0neighborFlags & checkFwd.z) == checkFwd.z) {
			int neighborsNeighborIndex = getNeighborIndex(cellbuffer.vneighbors, node0neighborIndex, chkdirs.y);
			int neighborsNeighborflags = cellbuffer.vflags[neighborsNeighborIndex];
			if ((neighborsNeighborflags & HAS_CORRECTED_POSITION) == HAS_CORRECTED_POSITION) {
				cpArray.y = neighborsNeighborIndex + 1;
			}
			else {
				cpArray.y = neighborsNeighborIndex;
			}
		}
	}
	else {
		if ((node0neighborFlags & HAS_CORRECTED_POSITION) == HAS_CORRECTED_POSITION) {
			cpArray.x++;
		}
	}

	return cpArray;
}

__device__ void findSegmentIntersections(float2 p0, float2 p1, float2 p2, float2 cellSpaceCoords, float2 ULCoords, float2 URCoords, float2 LLCoords, float2 LRCoords, int4 &influencingPixels) {
	float2 pointA = calcSplinePoint(p0, p1, p2, 0.0);
	for (float t = STEP; t < (1.0 + STEP); t = t + STEP) {
		float2 pointB = calcSplinePoint(p0, p1, p2, t);
		//evaluate interections
		if (intersects(cellSpaceCoords, ULCoords, pointA, pointB)) {
			influencingPixels.x = false;
		}
		if (intersects(cellSpaceCoords, URCoords, pointA, pointB)) {
			influencingPixels.y = false;
		}
		if (intersects(cellSpaceCoords, LLCoords, pointA, pointB)) {
			influencingPixels.z = false;
		}
		if (intersects(cellSpaceCoords, LRCoords, pointA, pointB)) {
			influencingPixels.w = false;
		}
		pointA = pointB;
	}
}

__global__ void GaussRasterizer(CellGraphBudffer_t cellbuffer, texture_t similarity, texture_t input, texture_t output){
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= output.width || y >= output.height)return;
	int4 influencingPixels;
	influencingPixels.x = influencingPixels.y = influencingPixels.z = influencingPixels.w = 1;

	// convert fragment-space coordinates to cell-space coordinates
	float2 cellSpaceCoords = output.GetUV(x, y)*vec2(input.width - 1, input.height - 1);
	// Knot Buffer lookup Coordinates
	int fragmentBaseKnotIndex = int(2 * floor(cellSpaceCoords.x) + floor(cellSpaceCoords.y) * 2 * (input.width - 1));
	//fetch flags
	int node0flags = cellbuffer.vflags[fragmentBaseKnotIndex];
	bool hasCorrectedPosition = false; //TODO!!!
	// surrounding pixel Coordinates 
	float2 ULCoords = vec2(floor(cellSpaceCoords.x), ceil(cellSpaceCoords.y));
	float2 URCoords = vec2(ceil(cellSpaceCoords.x), ceil(cellSpaceCoords.y));
	float2 LLCoords = vec2(floor(cellSpaceCoords.x), floor(cellSpaceCoords.y));
	float2 LRCoords = vec2(ceil(cellSpaceCoords.x), floor(cellSpaceCoords.y));

	if (node0flags > 0.0) {
		//gather neighbors
		int4 node0neighbors = cellbuffer.vneighbors[fragmentBaseKnotIndex];
		//compute valence
		int node0valence = computeValence(node0flags);
		float2 node0pos = cellbuffer.pos[fragmentBaseKnotIndex];
		if (node0valence == 1) {
			int2 cpArray = ivec2(-1, -1); //this array holds indices of the neighboring spline control points we need to interpolate
			if ((node0flags & HAS_NORTHERN_NEIGHBOR) == HAS_NORTHERN_NEIGHBOR) {
				cpArray = getCPs(cellbuffer, node0neighbors.x, NORTH);
			}
			else if ((node0flags & HAS_EASTERN_NEIGHBOR) == HAS_EASTERN_NEIGHBOR) {
				cpArray = getCPs(cellbuffer, node0neighbors.y, EAST);
			}
			else if ((node0flags & HAS_SOUTHERN_NEIGHBOR) == HAS_SOUTHERN_NEIGHBOR) {
				cpArray = getCPs(cellbuffer, node0neighbors.z, SOUTH);
			}
			else if ((node0flags & HAS_WESTERN_NEIGHBOR) == HAS_WESTERN_NEIGHBOR) {
				cpArray = getCPs(cellbuffer, node0neighbors.w, WEST);
			}
			float2 p1pos = cellbuffer.pos[cpArray.x];
			findSegmentIntersections(node0pos, node0pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			if (cpArray.y > -1) {
				float2 p2pos = cellbuffer.pos[cpArray.y];
				findSegmentIntersections(node0pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
		}
		else if (node0valence == 2) {
			int4 cpArray = ivec4(-1, -1, -1, -1); //this array holds indices of the neighboring spline control points we need to interpolate
			bool foundFirst = false;
			if ((node0flags & HAS_NORTHERN_NEIGHBOR) == HAS_NORTHERN_NEIGHBOR) {
				int2 temp = getCPs(cellbuffer, node0neighbors.x, NORTH);
				cpArray.x = temp.x;
				cpArray.y = temp.y;
				foundFirst = true;
			}

			if ((node0flags & HAS_EASTERN_NEIGHBOR) == HAS_EASTERN_NEIGHBOR) {
				if (foundFirst) {
					int2 temp = getCPs(cellbuffer, node0neighbors.y, EAST);
					cpArray.z = temp.x;
					cpArray.w = temp.y;
				}
				else {
					int2 temp = getCPs(cellbuffer, node0neighbors.y, EAST);
					cpArray.x = temp.x;
					cpArray.y = temp.y;
					foundFirst = true;
				}
			}
			if ((node0flags & HAS_SOUTHERN_NEIGHBOR) == HAS_SOUTHERN_NEIGHBOR) {
				if (foundFirst) {

					int2 temp = getCPs(cellbuffer, node0neighbors.z, SOUTH);
					cpArray.z = temp.x;
					cpArray.w = temp.y;
				}
				else {
					int2 temp = getCPs(cellbuffer, node0neighbors.z, SOUTH);
					cpArray.x = temp.x;
					cpArray.y = temp.y;
					foundFirst = true;
				}
			}

			if ((node0flags & HAS_WESTERN_NEIGHBOR) == HAS_WESTERN_NEIGHBOR) {
				int2 temp = getCPs(cellbuffer, node0neighbors.w, WEST);
				cpArray.z = temp.x;
				cpArray.w = temp.y;
			}
			float2 pm1pos = cellbuffer.pos[cpArray.x];
			float2 p1pos = cellbuffer.pos[cpArray.z];
			findSegmentIntersections(pm1pos, node0pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			if (cpArray.y > -1) {
				float2 pm2pos = cellbuffer.pos[cpArray.y];
				findSegmentIntersections(node0pos, pm1pos, pm2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, pm1pos, pm1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			if (cpArray.w > -1) {
				float2 p2pos = cellbuffer.pos[cpArray.w];
				findSegmentIntersections(node0pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}

		}
		else if (node0valence == 3) {
			hasCorrectedPosition = true;
			int4 cpArray = ivec4(-1, -1, -1, -1); //this array holds indices of the neighboring spline control points we need to interpolate
			bool foundFirst = false;
			int tBaseDir = 0;
			int tBaseNeighborIndex = -1;
			if ((node0flags & HAS_NORTHERN_NEIGHBOR) == HAS_NORTHERN_NEIGHBOR) {
				if ((node0flags & HAS_NORTHERN_SPLINE) == HAS_NORTHERN_SPLINE) {
					int2 temp = getCPs(cellbuffer, node0neighbors.x, NORTH);
					cpArray.x = temp.x;
					cpArray.y = temp.y;
					foundFirst = true;
				}
				else {
					tBaseDir = NORTH;
					tBaseNeighborIndex = node0neighbors.x;
				}
			}

			if ((node0flags & HAS_EASTERN_NEIGHBOR) == HAS_EASTERN_NEIGHBOR) {
				if ((node0flags & HAS_EASTERN_SPLINE) == HAS_EASTERN_SPLINE) {
					if (foundFirst) {
						int2 temp = getCPs(cellbuffer, node0neighbors.y, EAST);
						cpArray.z = temp.x;
						cpArray.w = temp.y;
					}
					else {
						int2 temp = getCPs(cellbuffer, node0neighbors.y, EAST);
						cpArray.x = temp.x;
						cpArray.y = temp.y;
						foundFirst = true;
					}
				}
				else {
					tBaseDir = EAST;
					tBaseNeighborIndex = node0neighbors.y;
				}
			}
			if ((node0flags & HAS_SOUTHERN_NEIGHBOR) == HAS_SOUTHERN_NEIGHBOR) {
				if ((node0flags & HAS_SOUTHERN_SPLINE) == HAS_SOUTHERN_SPLINE) {
					if (foundFirst) {
						int2 temp = getCPs(cellbuffer, node0neighbors.z, SOUTH);
						cpArray.z = temp.x;
						cpArray.w = temp.y;
					}
					else {
						int2 temp = getCPs(cellbuffer, node0neighbors.z, SOUTH);
						cpArray.x = temp.x;
						cpArray.y = temp.y;
						foundFirst = true;
					}
				}
				else {
					tBaseDir = SOUTH;
					tBaseNeighborIndex = node0neighbors.z;
				}
			}

			if ((node0flags & HAS_WESTERN_NEIGHBOR) == HAS_WESTERN_NEIGHBOR) {
				if ((node0flags & HAS_WESTERN_SPLINE) == HAS_WESTERN_SPLINE) {
					int2 temp = getCPs(cellbuffer, node0neighbors.w, WEST);
					cpArray.z = temp.x;
					cpArray.w = temp.y;
				}
				else {
					tBaseDir = WEST;
					tBaseNeighborIndex = node0neighbors.w;
				}
			}
			float2 pm1pos = cellbuffer.pos[cpArray.x];
			float2 p1pos = cellbuffer.pos[cpArray.z];
			findSegmentIntersections(pm1pos, node0pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			if (cpArray.y > -1) {
				float2 pm2pos = cellbuffer.pos[cpArray.y];
				findSegmentIntersections(node0pos, pm1pos, pm2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, pm1pos, pm1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			if (cpArray.w > -1) {
				float2 p2pos = cellbuffer.pos[cpArray.w];
				findSegmentIntersections(node0pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}

			//check T-Base
			int2 temp = getCPs(cellbuffer, tBaseNeighborIndex, tBaseDir);
			cpArray.x = temp.x;
			cpArray.y = temp.y;
			node0pos = cellbuffer.pos[fragmentBaseKnotIndex + 1];
			p1pos = cellbuffer.pos[cpArray.x];
			findSegmentIntersections(node0pos, node0pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			if (cpArray.y > -1) {
				float2 p2pos = cellbuffer.pos[cpArray.y];
				findSegmentIntersections(node0pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
		}
		else { // valence 4
			int2 cpArray = ivec2(-1, -1);
			cpArray = getCPs(cellbuffer, node0neighbors.x, NORTH);
			float2 p1pos = cellbuffer.pos[cpArray.x];
			findSegmentIntersections(node0pos, node0pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			if (cpArray.y > -1) {
				float2 p2pos = cellbuffer.pos[cpArray.y];
				findSegmentIntersections(node0pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			cpArray = getCPs(cellbuffer, node0neighbors.y, EAST);
			p1pos = cellbuffer.pos[cpArray.x];
			findSegmentIntersections(node0pos, node0pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			if (cpArray.y > -1) {
				float2 p2pos = cellbuffer.pos[cpArray.y];
				findSegmentIntersections(node0pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			cpArray = getCPs(cellbuffer, node0neighbors.z, SOUTH);
			p1pos = cellbuffer.pos[cpArray.x];
			findSegmentIntersections(node0pos, node0pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			if (cpArray.y > -1) {
				float2 p2pos = cellbuffer.pos[cpArray.y];
				findSegmentIntersections(node0pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			cpArray = getCPs(cellbuffer, node0neighbors.w, WEST);
			p1pos = cellbuffer.pos[cpArray.x];
			findSegmentIntersections(node0pos, node0pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			if (cpArray.y > -1) {
				float2 p2pos = cellbuffer.pos[cpArray.y];
				findSegmentIntersections(node0pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
			else {
				findSegmentIntersections(node0pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
			}
		}
	}
	if (!hasCorrectedPosition) {
		int node1flags = cellbuffer.vflags[fragmentBaseKnotIndex + 1];
		if (node1flags > 0.0) {
			//gather neighbors
			int4 node1neighbors = cellbuffer.vneighbors[fragmentBaseKnotIndex + 1];
			//compute valence
			int node1valence = computeValence(node1flags);
			float2 node1pos = cellbuffer.pos[fragmentBaseKnotIndex + 1];
			if (node1valence == 1) {
				int2 cpArray = ivec2(-1, -1); //this array holds indices of the neighboring spline control points we need to interpolate
				if ((node1flags & HAS_NORTHERN_NEIGHBOR) == HAS_NORTHERN_NEIGHBOR) {
					cpArray = getCPs(cellbuffer, node1neighbors.x, NORTH);
				}
				else if ((node1flags & HAS_EASTERN_NEIGHBOR) == HAS_EASTERN_NEIGHBOR) {
					cpArray = getCPs(cellbuffer, node1neighbors.y, EAST);
				}
				else if ((node1flags & HAS_SOUTHERN_NEIGHBOR) == HAS_SOUTHERN_NEIGHBOR) {
					cpArray = getCPs(cellbuffer, node1neighbors.z, SOUTH);
				}
				else if ((node1flags & HAS_WESTERN_NEIGHBOR) == HAS_WESTERN_NEIGHBOR) {
					cpArray = getCPs(cellbuffer, node1neighbors.w, WEST);
				}
				float2 p1pos = cellbuffer.pos[cpArray.x];
				findSegmentIntersections(node1pos, node1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
				if (cpArray.y > -1) {
					float2 p2pos = cellbuffer.pos[cpArray.y];
					findSegmentIntersections(node1pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
				}
				else {
					findSegmentIntersections(node1pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
				}
			}
			else if (node1valence == 2) {
				int4 cpArray = ivec4(-1, -1, -1, -1); //this array holds indices of the neighboring spline control points we need to interpolate
				bool foundFirst = false;
				if ((node1flags & HAS_NORTHERN_NEIGHBOR) == HAS_NORTHERN_NEIGHBOR) {
					int2 temp = getCPs(cellbuffer, node1neighbors.x, NORTH);
					cpArray.x = temp.x;
					cpArray.y = temp.y;
					foundFirst = true;
				}

				if ((node1flags & HAS_EASTERN_NEIGHBOR) == HAS_EASTERN_NEIGHBOR) {
					if (foundFirst) {
						int2 temp = getCPs(cellbuffer, node1neighbors.y, EAST);
						cpArray.z = temp.x;
						cpArray.w = temp.y;
					}
					else {
						int2 temp = getCPs(cellbuffer, node1neighbors.y, EAST);
						cpArray.x = temp.x;
						cpArray.y = temp.y;
						foundFirst = true;
					}
				}
				if ((node1flags & HAS_SOUTHERN_NEIGHBOR) == HAS_SOUTHERN_NEIGHBOR) {
					if (foundFirst) {
						int2 temp = getCPs(cellbuffer, node1neighbors.z, SOUTH);
						cpArray.z = temp.x;
						cpArray.w = temp.y;
					}
					else {
						int2 temp = getCPs(cellbuffer, node1neighbors.z, SOUTH);
						cpArray.x = temp.x;
						cpArray.y = temp.y;
						foundFirst = true;
					}
				}

				if ((node1flags & HAS_WESTERN_NEIGHBOR) == HAS_WESTERN_NEIGHBOR) {
					int2 temp = getCPs(cellbuffer, node1neighbors.w, WEST);
					cpArray.z = temp.x;
					cpArray.w = temp.y;
				}
				float2 pm1pos = cellbuffer.pos[cpArray.x];
				float2 p1pos = cellbuffer.pos[cpArray.z];
				findSegmentIntersections(pm1pos, node1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
				if (cpArray.y > -1) {
					float2 pm2pos = cellbuffer.pos[cpArray.y];
					findSegmentIntersections(node1pos, pm1pos, pm2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
				}
				else {
					findSegmentIntersections(node1pos, pm1pos, pm1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
				}
				if (cpArray.w > -1) {
					float2 p2pos = cellbuffer.pos[cpArray.w];
					findSegmentIntersections(node1pos, p1pos, p2pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
				}
				else {
					findSegmentIntersections(node1pos, p1pos, p1pos, cellSpaceCoords, ULCoords, URCoords, LLCoords, LRCoords, influencingPixels);
				}
			}
		}
	}

	float4 colorSum = vec4(0.0, 0.0, 0.0, 0.0);
	float weightSum = 0.0;
	//influencingPixels order: UL UR LL LR

	if (influencingPixels.x) {
		//calculate influence of the Pixel
		uchar* cols = input.Texture2D(ULCoords.x, ULCoords.y);
		float4 col;
		col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
		float dist = distance(cellSpaceCoords, ULCoords);
		float weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
		colorSum += col * weight;
		weightSum += weight;
		//checkout this pixels connected neigbors
		int2 lookupcoords = ivec2(2 * ULCoords.x + 1, 2 * ULCoords.y + 1);
		int edges = similarity.Texture2D(lookupcoords.x, lookupcoords.y)[1];
		//calculate weights for those pixels
		if ((edges & SOUTHWEST) == SOUTHWEST) {
			cols = input.Texture2D(ULCoords.x - 1, ULCoords.y - 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(ULCoords.x - 1, ULCoords.y - 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & WEST) == WEST) {
			cols = input.Texture2D(ULCoords.x - 1, ULCoords.y);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(ULCoords.x - 1, ULCoords.y));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & NORTHWEST) == NORTHWEST) {
			cols = input.Texture2D(ULCoords.x - 1, ULCoords.y + 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(ULCoords.x - 1, ULCoords.y + 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & NORTH) == NORTH) {
			cols = input.Texture2D(ULCoords.x, ULCoords.y + 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(ULCoords.x, ULCoords.y + 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & NORTHEAST) == NORTHEAST) {
			cols = input.Texture2D(ULCoords.x + 1, ULCoords.y + 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(ULCoords.x + 1, ULCoords.y + 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
	}

	if (influencingPixels.y) {
		//calculate influence of the Pixel
		uchar* cols = input.Texture2D(URCoords.x, URCoords.y);
		float4 col;
		col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
		float dist = distance(cellSpaceCoords, URCoords);
		float weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
		colorSum += col * weight;
		weightSum += weight;
		//checkout this pixels connected neigbors
		int2 lookupcoords = ivec2(2 * URCoords.x + 1, 2 * URCoords.y + 1);
		int edges = similarity.Texture2D(lookupcoords.x, lookupcoords.y)[1];
		//calculate weights for those pixels
		//TODO: THIS ACTUALLY CAUSES ARTIFACTS - NO IDEA WHY
		if ((edges & NORTH) == NORTH) {
			cols = input.Texture2D(URCoords.x, URCoords.y + 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(URCoords.x, URCoords.y + 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & NORTHEAST) == NORTHEAST) {
			cols = input.Texture2D(URCoords.x + 1, URCoords.y + 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(URCoords.x + 1, URCoords.y + 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & EAST) == EAST) {
			cols = input.Texture2D(URCoords.x + 1, URCoords.y);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(URCoords.x + 1, URCoords.y));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & SOUTHEAST) == SOUTHEAST) {
			cols = input.Texture2D(URCoords.x + 1, URCoords.y - 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(URCoords.x + 1, URCoords.y - 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}

	}


	if (influencingPixels.z) {
		//calculate influence of the Pixel
		uchar* cols = input.Texture2D(LLCoords.x, LLCoords.y);
		float4 col;
		col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
		float dist = distance(cellSpaceCoords, LLCoords);
		float weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
		colorSum += col * weight;
		weightSum += weight;
		//checkout this pixels connected neigbors
		int2 lookupcoords = ivec2(2 * LLCoords.x + 1, 2 * LLCoords.y + 1);
		int edges = similarity.Texture2D(lookupcoords.x, lookupcoords.y)[1];
		//calculate weights for those pixels
		//TODO: THIS ACTUALLY CAUSES ARTIFACTS - NO IDEA WHY
		if ((edges & WEST) == WEST) {
			cols = input.Texture2D(LLCoords.x - 1, LLCoords.y);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(LLCoords.x - 1, LLCoords.y));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & SOUTHWEST) == SOUTHWEST) {
			cols = input.Texture2D(LLCoords.x - 1, LLCoords.y - 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(LLCoords.x - 1, LLCoords.y - 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & SOUTH) == SOUTH) {
			cols = input.Texture2D(LLCoords.x, LLCoords.y - 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(LLCoords.x, LLCoords.y - 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & SOUTHEAST) == SOUTHEAST) {
			cols = input.Texture2D(LLCoords.x + 1, LLCoords.y - 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(LLCoords.x + 1, LLCoords.y - 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
	}
	if (influencingPixels.w) {
		uchar* cols = input.Texture2D(LRCoords.x, LRCoords.y);
		float4 col;
		col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
		float dist = distance(cellSpaceCoords, LRCoords);
		float weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
		colorSum += col * weight;
		weightSum += weight;
		//checkout this pixels connected neigbors
		int2 lookupcoords = ivec2(2 * LRCoords.x + 1, 2 * LRCoords.y + 1);
		int edges = similarity.Texture2D(lookupcoords.x, lookupcoords.y)[1];
		//calculate weights for those pixels
		if ((edges & NORTHEAST) == NORTHEAST) {
			cols = input.Texture2D(LRCoords.x + 1, LRCoords.y + 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(LRCoords.x + 1, LRCoords.y + 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & EAST) == EAST) {
			cols = input.Texture2D(LRCoords.x + 1, LRCoords.y);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(LRCoords.x + 1, LRCoords.y));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & SOUTHWEST) == SOUTHWEST) {
			cols = input.Texture2D(LRCoords.x - 1, LRCoords.y - 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(LRCoords.x - 1, LRCoords.y - 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & SOUTH) == SOUTH) {
			cols = input.Texture2D(LRCoords.x, LRCoords.y - 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(LRCoords.x, LRCoords.y - 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
		if ((edges & SOUTHEAST) == SOUTHEAST) {
			cols = input.Texture2D(LRCoords.x + 1, LRCoords.y - 1);
			col.x = cols[0]; col.y = cols[1]; col.z = cols[2]; col.w = cols[3];
			dist = distance(cellSpaceCoords, vec2(LRCoords.x + 1, LRCoords.y - 1));
			weight = exp(-(dist*dist)*GAUSS_MULTIPLIER);
			colorSum += col * weight;
			weightSum += weight;
		}
	}
	output.SetPixel(output.GetPos(x, y), colorSum.x / weightSum, colorSum.y / weightSum, colorSum.z / weightSum);
}
#pragma endregion RASTERIZER
GraphBuilder::GraphBuilder()
{
#ifdef LOG
	logFile = std::ofstream("log.txt");
#endif //LOG
}

GraphBuilder::~GraphBuilder()
{
#ifdef LOG
	logFile.close();
#endif //LOG
}

Image* GraphBuilder::Bulid_DisSimilarGraph(Image *source)
{
	cudaError_t error;
	Image *disSimTex = new Image(source->GetSize(2, 1));
	Image *update_disSimTex = new Image(source->GetSize(2, 1));
	int width_tar = disSimTex->GetSize().width, heigh_tar = disSimTex->GetSize().height;

#ifdef LOG
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
#endif //LOG
	DisSimilarGraph_Kernel << <dim3(CeilDiv(width_tar, 32), CeilDiv(heigh_tar, 16)), dim3(32, 16) >> >(source->GetTexture(), disSimTex->GetTexture());
#ifdef LOG
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	logFile << "DisSimilarGraph_Kernel Time : \t" << elapsedTime <<"ms\n";

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
#endif //LOG
	//disSimTex->toHost();
	//disSimTex->Show(true,16);
	//ans[0]->Show(true,1);
	ValenceUpdate_Kernel << <dim3(CeilDiv(width_tar, 32), CeilDiv(heigh_tar, 16)), dim3(32, 16) >> >(disSimTex->GetTexture(), update_disSimTex->GetTexture());
#ifdef LOG
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	logFile << "ValenceUpdate_Kernel Time : \t" << elapsedTime << "ms\n";

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
#endif //LOG
	//update_disSimTex->toHost();
	//update_disSimTex->Show(true,16);
	//ans[1]->Show(true,1);
	eliminateCrossings_Kernel << <dim3(CeilDiv(width_tar, 32), CeilDiv(heigh_tar, 16)), dim3(32, 16) >> >(update_disSimTex->GetTexture(), disSimTex->GetTexture());
#ifdef LOG
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	logFile << "eliminateCrossings_Kernel Time : \t" << elapsedTime << "ms\n";

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
#endif //LOG
	//disSimTex->toHost();
	//disSimTex->Show(true,16);
	//ans[2]->Show(true,1);
	ValenceUpdate_Kernel << <dim3(CeilDiv(width_tar, 32), CeilDiv(heigh_tar, 16)), dim3(32, 16) >> >(disSimTex->GetTexture(), update_disSimTex->GetTexture());
#ifdef LOG
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	logFile << "DisSimilarGraph_Kernel Second Pass Time : \t" << elapsedTime << "ms\n";
#endif //LOG
	//update_disSimTex->toHost();
	//update_disSimTex->Show(true,16);
	//ans[3]->Show(true,1);
	error = cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("%s\n", cudaGetErrorString(error));
		system("pause");
	}
	return update_disSimTex;

}
CellGraphBudffer_t* GraphBuilder::BulidCellGraph(Image *source, Image *simGraph)
{
	cudaError_t error;
	CellGraphBudffer_t *cgBuffer = GenCGBuffer(source->GetSize().width, source->GetSize().height);
	int width_tar = source->GetSize(1, -1).width, heigh_tar = source->GetSize(1, -1).height;
#ifdef LOG
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
#endif //LOG
	FullCellGraphCOnstruction_Kernel << <dim3(CeilDiv(width_tar, 32), CeilDiv(heigh_tar, 16)), dim3(32, 16) >> >(simGraph->GetTexture(), source->GetTexture(), *cgBuffer);
#ifdef LOG
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	logFile << "FullCellGraphCOnstruction_Kernel Time : \t" << elapsedTime << "ms\n";
#endif //LOG
	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("%s\n", cudaGetErrorString(error));
		system("pause");
	}
	return cgBuffer;
}
void GraphBuilder::OptimizeCurve(CellGraphBudffer_t* cgBuffer, cv::Size sourceSize){
	cudaError_t error;

	int width_tar = sourceSize.width - 1, heigh_tar = (sourceSize.height - 1) * 2;
#ifdef LOG
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
#endif //LOG
	OptimizeEnergy << <dim3(CeilDiv(width_tar, 32), CeilDiv(heigh_tar, 16)), dim3(32, 16) >> >(*cgBuffer, width_tar, width_tar*heigh_tar);
#ifdef LOG
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	logFile << "OptimizeEnergy Pass Time : \t" << elapsedTime << "ms\n";

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
#endif //LOG
	UpdateCorrectedPositions << <dim3(CeilDiv(width_tar, 32), CeilDiv(heigh_tar, 16)), dim3(32, 16) >> >(*cgBuffer, width_tar, width_tar*heigh_tar);
#ifdef LOG
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	logFile << "UpdateCorrectedPositions Pass Time : \t" << elapsedTime << "ms\n";
#endif //LOG
	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("%s\n", cudaGetErrorString(error));
		system("pause");
	}
}
void GraphBuilder::Rasterizer(CellGraphBudffer_t *cgBuffer, Image *simGraph, Image *source, Image *result){
	cudaError_t error;
	int width_tar = result->GetSize().width, heigh_tar = result->GetSize().height;
#ifdef LOG
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
#endif //LOG
	GaussRasterizer << <dim3(CeilDiv(width_tar, 32), CeilDiv(heigh_tar, 16)), dim3(32, 16) >> >(*cgBuffer, simGraph->GetTexture(), source->GetTexture(), result->GetTexture());
#ifdef LOG
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	logFile << "GaussRasterizer Pass Time : \t" << elapsedTime << "ms\n";
#endif //LOG
	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("%s\n", cudaGetErrorString(error));
		system("pause");
	}
}