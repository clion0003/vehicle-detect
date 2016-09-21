#include "func_cuda.h"
#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdlib.h>
#include "alg_bgmodel.h"
#define POINT_PER_THREAD 15
#define EDGE_WIDTH 6

//shared with other files
TVehLprLcRect atCudaRoughRectsOut[MAX_LOCNUM];
l32 l32CudaRectCountOut;
u8 *pu8CudaInputSrc;
u8 *pu8CudaRGBOut;
u8 *pu8CudaZoomOut;
u8 *pu8CudaGrayOut;
u8 *pu8CudaFrameCurOut;
u8 *pu8CudaFGFrameOut;
sem_t sem_empty, sem_full, sem_ready, sem_finish;


//only used in this file
u8 *pu8CudaImgBuf;
u8 *pu8CudaRGBBuf;
u8 *pu8CudaZoomBuf;
u8 *pu8CudaGrayBuf;
u8 *pu8CudaFrameCurBuf;
u8 *pu8CudaFGFrameBuf;
void *pvCudaBgModel;
struct timeval tstart[20];
int t_cnt = -1;
//thread param
pthread_t tid;
struct _cuda_thread_arg {
	l32 l32Width;
	l32 l32Height;
	l32 l32Stride;
}thread_arg;

//inner function param
typedef struct _LCMAX_POS_
{
    l32 l32PosX;
    l32 l32PosY;
    l32 l32Val;
}LCMAX_POS;

struct stLocLcMax {
	u8 *pu8EdgeDensMap;
	u8 *pu8EdgeDensMapOrg;
	u8 *pu8EdgeDensMap2;
	u8 *pu8EdgeDensMapMoph;
	u8 *pu8Temp;
	u8 *tRoiRect;
} ptLocLcMax0;

struct stImage {
	u8 *pu8Y;
	u8 *pu8U;
	u8 *pu8V;
	l32 l32Width;
	l32 l32Height;
} ptImage0;

struct stLocInput {
	u8 *pu8SrcImg;
	l32 l32Width;
	l32 l32Height;
	l32 l32Stride;
} ptLocInput0;
    
struct stLocOutput {
	TVehLprLcRect *ptRect;
	l32 l32RectCount;
} ptLocOutput0;
	
//cuda thread function
void *thread_func_cuda(void *arg);

void init_cuda(l32 l32OrgWidth, l32 l32OrgHeight, l32 l32FrameWidth, l32 l32FrameHeight)
{
	l32 bufLength = (l32OrgWidth * l32OrgHeight * 3) >> 1;

	//init semaphore
	sem_init(&sem_empty, 0, 1);
	sem_init(&sem_full, 0, 0);
	sem_init(&sem_ready, 0, 0);
	sem_init(&sem_finish, 0, 1);

	//allocate cuda buffer
	checkCudaErrors(cudaMalloc(&pu8CudaImgBuf, sizeof(u8) * bufLength));
	checkCudaErrors(cudaMallocManaged(&pu8CudaRGBBuf, sizeof(u8) * bufLength * 2));
	checkCudaErrors(cudaMallocManaged(&pu8CudaZoomBuf, sizeof(u8) * SCALEWIDTH * SCALEHEIGHT));
	checkCudaErrors(cudaMallocManaged(&pu8CudaGrayBuf, sizeof(u8) * SCALEWIDTH * SCALEHEIGHT));
	checkCudaErrors(cudaMallocManaged(&pu8CudaFrameCurBuf, sizeof(u8) * SCALEWIDTH * SCALEHEIGHT));
	checkCudaErrors(cudaMallocManaged(&pu8CudaFGFrameBuf, sizeof(u8) * SCALEWIDTH * SCALEHEIGHT));

	//allocate cuda output buffer
	pu8CudaRGBOut = (u8 *)malloc(sizeof(u8) * bufLength * 2);
	pu8CudaZoomOut = (u8 *)malloc(sizeof(u8) * SCALEWIDTH * SCALEHEIGHT);
	pu8CudaGrayOut = (u8 *)malloc(sizeof(u8) * SCALEWIDTH * SCALEHEIGHT);
	pu8CudaFrameCurOut = (u8 *)malloc(sizeof(u8) * SCALEWIDTH * SCALEHEIGHT);
	pu8CudaFGFrameOut = (u8 *)malloc(sizeof(u8) * SCALEWIDTH * SCALEHEIGHT);
	if (pu8CudaRGBOut == NULL || pu8CudaZoomOut == NULL || pu8CudaGrayOut == NULL || pu8CudaFrameCurOut == NULL || pu8CudaFGFrameOut == NULL) {
		printf("cuda init malloc error\n");
		exit(1);
	}
	
	//allocate rough locate buffer
	checkCudaErrors(cudaMallocManaged(&ptLocLcMax0.pu8EdgeDensMapOrg, sizeof(u8) * bufLength));
	checkCudaErrors(cudaMallocManaged(&ptLocLcMax0.pu8EdgeDensMap, sizeof(u8) * bufLength));
	checkCudaErrors(cudaMallocManaged(&ptLocLcMax0.pu8EdgeDensMap2, sizeof(u8) * bufLength));
	checkCudaErrors(cudaMallocManaged(&ptLocLcMax0.pu8EdgeDensMapMoph, sizeof(u8) * bufLength));
	checkCudaErrors(cudaMallocManaged(&ptLocLcMax0.pu8Temp, sizeof(u8) * bufLength));

	//open bgm
	if (BGMGuassMogOpen((void **)&pvCudaBgModel, l32FrameWidth, l32FrameHeight) != EStatus_Success) {
		printf("BGM Open error");	
		exit(1);
	}

	//create cuda thread
	thread_arg.l32Width = l32OrgWidth;
	thread_arg.l32Stride = l32OrgWidth;
	thread_arg.l32Height = l32OrgHeight;
	int error = pthread_create(&tid, NULL, thread_func_cuda, &thread_arg);
	if (error != 0) {
		printf("create thread error: %d\n", error);
	} 	
}

void uninit_cuda()
{
	//let cuda thread exit
	sem_wait(&sem_empty);
	pu8CudaInputSrc = NULL;
	sem_post(&sem_full);
	pthread_join(tid, NULL);

	//free cuda buffer
	checkCudaErrors(cudaFree(pu8CudaImgBuf));
	checkCudaErrors(cudaFree(pu8CudaRGBBuf));
	checkCudaErrors(cudaFree(pu8CudaZoomBuf));
	checkCudaErrors(cudaFree(pu8CudaGrayBuf));
	checkCudaErrors(cudaFree(pu8CudaFrameCurBuf));
	checkCudaErrors(cudaFree(pu8CudaFGFrameBuf));

	//free cuda output buffer
	free(pu8CudaRGBOut);
	free(pu8CudaZoomOut);
	free(pu8CudaGrayOut);
	free(pu8CudaFrameCurOut);
	free(pu8CudaFGFrameOut);

	//free buffer for rough locate
	checkCudaErrors(cudaFree(ptLocLcMax0.pu8EdgeDensMapOrg));
	checkCudaErrors(cudaFree(ptLocLcMax0.pu8EdgeDensMap));
	checkCudaErrors(cudaFree(ptLocLcMax0.pu8EdgeDensMap2));
	checkCudaErrors(cudaFree(ptLocLcMax0.pu8EdgeDensMapMoph));
	checkCudaErrors(cudaFree(ptLocLcMax0.pu8Temp));

	//destroy semaphore
	sem_destroy(&sem_empty);
	sem_destroy(&sem_full);
	sem_destroy(&sem_ready);
	sem_destroy(&sem_finish);

	//close bgm
	BGMGuassMogClose(pvCudaBgModel);
}

void check(unsigned int result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
                file, line, result, func);
        // Make sure we call CUDA Device Reset before exiting
        exit(result);
    }
}

void startTimer()
{
	if (t_cnt < 20){
    	gettimeofday(&tstart[++t_cnt], NULL);
	}
}

void dispTimer(char f_name[])
{
    struct timeval tend;
    gettimeofday(&tend, NULL);
    long time_used = (tend.tv_sec-tstart[t_cnt].tv_sec)*1000000+tend.tv_usec-tstart[t_cnt].tv_usec;
    printf("%s time: %.3lf ms\n", f_name, time_used / 1000.0);
	t_cnt--;
}

class myTimer {
private:
	struct timeval tstart, tend;
	long time_used;

public:
	void Timer()
	{
		start();	
	}

	void start()
	{
    		gettimeofday(&tstart, NULL);
	}
	
	void disp(char *str)
	{
    		gettimeofday(&tend, NULL);
    		time_used = (tend.tv_sec - tstart.tv_sec) * 1000000 + tend.tv_usec - tstart.tv_usec;
    		printf("%s time: %.3lf ms\n", str, time_used / 1000.0);
	}
	void disp()
	{
    		gettimeofday(&tend, NULL);
    		time_used = (tend.tv_sec - tstart.tv_sec) * 1000000 + tend.tv_usec - tstart.tv_usec;
    		printf("total time: %.3lf ms\n", time_used / 1000.0);
	}
};

__global__ void kerUpdatePixelBackgroundGMM2(
	TGuass2Value* ptGuassValueSrc,
	u8 *pu8PixelValueSrc,
	u8 *pu8GuassUsedSrc,
	u8 *pu8ForgroundPixelSrc,
	f32 fLearnRate,
	f32 fForegroundThreshod,
	f32 fDeviationThreshold,
	f32 fTb,
	f32 fCT,
	l32 bShadowDetection,
	f32 fShadowThreshold,
	l32 l32width,
	l32 l32height,
	l32 l32stride
	)
{
	u32 l32ImgIndexX, l32ImgIndexY,l32ImgIndex;
	l32 bMatched = 0;
	l32  l32Index, l32Local;
	l32 bBackGroundPixel = 0;
	TGuass2Value* ptGuassComponent;
	f32 fTotalWeight = 0;
	f32 fPrune = -fLearnRate * fCT;
	u8 u8PixelValue, *pu8ForgroundPixel, *pu8GuassUsed;
	TGuass2Value* ptGuassValue;
	l32 bShadowRe;
	l32ImgIndexX = blockIdx.x*blockDim.x + threadIdx.x;
	l32ImgIndexY = blockIdx.y*blockDim.y + threadIdx.y;
	if ((l32ImgIndexX < l32width) && (l32ImgIndexY < l32height)){
		l32ImgIndex = l32ImgIndexY*l32stride + l32ImgIndexX;

		ptGuassValue = ptGuassValueSrc + BGFG_MOG2_MAX_GUASSNUM*l32ImgIndex;
		ptGuassComponent = ptGuassValue;
		u8PixelValue = pu8PixelValueSrc[l32ImgIndex];
		pu8ForgroundPixel = pu8ForgroundPixelSrc + l32ImgIndex;
		pu8GuassUsed = pu8GuassUsedSrc + l32ImgIndex;

		for (l32Index = 0; l32Index < (*pu8GuassUsed); l32Index++, ptGuassComponent++)
		{
			f32 fWeight = ptGuassComponent->fWeight;
			f32 fAlpha;
			fWeight = (1 - fLearnRate) * fWeight + fPrune;
			if (!bMatched)
			{
				f32 fDif;
				fDif = (ptGuassComponent->fmean - u8PixelValue) * (ptGuassComponent->fmean - u8PixelValue);
				if (fTotalWeight < fForegroundThreshod && fDif < fTb * ptGuassComponent->fVar)
				{
					bBackGroundPixel = 1;
				}
				if (fDif < fDeviationThreshold * ptGuassComponent->fVar)
				{
					bMatched = 1;
					fWeight = fWeight + fLearnRate;
					fAlpha = fLearnRate / fWeight;
					ptGuassComponent->fmean = (1 - fAlpha) * ptGuassComponent->fmean + fAlpha * u8PixelValue;
					ptGuassComponent->fVar = MIN(BGFG_MOG2_VAR_MAX, MAX(BGFG_MOG2_VAR_MIN, (1 - fAlpha) * ptGuassComponent->fVar + fAlpha * fDif));
					for (l32Local = l32Index; l32Local > 0; l32Local--)
					{
						if (fWeight < (ptGuassValue[l32Local - 1].fWeight))
						{
							break;
						}
						else
						{
							TGuass2Value tTempGuass = ptGuassValue[l32Local];
							ptGuassValue[l32Local] = ptGuassValue[l32Local - 1];
							ptGuassValue[l32Local - 1] = tTempGuass;
							ptGuassComponent--;
						}
					}
				}
			}
			if (fWeight < -fPrune)
			{
				fWeight = 0.0;
				(*pu8GuassUsed)--;
			}

			ptGuassComponent->fWeight = fWeight;
			fTotalWeight += fWeight;
		}
		ptGuassComponent = ptGuassValue;
		for (l32Index = 0; l32Index < (*pu8GuassUsed); l32Index++, ptGuassComponent++)
		{
			ptGuassComponent->fWeight /= fTotalWeight;
		}
		if (!bMatched)
		{
			if (BGFG_MOG2_MAX_GUASSNUM == (*pu8GuassUsed))
			{
				ptGuassComponent = ptGuassValue + BGFG_MOG2_MAX_GUASSNUM - 1;
			}
			else
			{
				ptGuassComponent = ptGuassValue + (*pu8GuassUsed);
				(*pu8GuassUsed)++;
			}
			if ((*pu8GuassUsed) == 1)
			{
				ptGuassComponent->fWeight = 1;
			}
			else
			{
				ptGuassComponent->fWeight = fLearnRate;
				for (l32Index = 0; l32Index < (*pu8GuassUsed) - 1; l32Index++)
				{
					ptGuassValue[l32Index].fWeight = ptGuassValue[l32Index].fWeight * (1 - fLearnRate);
				}
			}
			ptGuassComponent->fmean = u8PixelValue;
			ptGuassComponent->fVar = BGFG_MOG2_VAR_INIT;
			for (l32Local = (*pu8GuassUsed) - 1; l32Local > 0; l32Local--)
			{
				if (fLearnRate < (ptGuassValue[l32Local - 1].fWeight))
				{
					break;
				}
				else
				{
					TGuass2Value tTempGuass = ptGuassValue[l32Local];
					ptGuassValue[l32Local] = ptGuassValue[l32Local - 1];
					ptGuassValue[l32Local - 1] = tTempGuass;
					ptGuassComponent--;
				}
			}
		}


		if (bBackGroundPixel)
		{
			*pu8ForgroundPixel = 0;
		}
		else
		{
			if (bShadowDetection)
			{
#if 0
				f32 fWeight = 0;
				f32 fnumerator, fdenominator;
				l32 l32IModes, l32IndexD;
				TGuass2Value tGass2value;
				f32 fRate;
				f32 fDist2Rate;
				f32 fDistD;

				// check all the components  marked as background:
				for (l32IModes = 0; l32IModes < *pu8GuassUsed; l32IModes++)
				{

					tGass2value = ptGuassValue[l32IModes];

					fnumerator = 0.0f;
					fdenominator = 0.0f;
					fnumerator += u8PixelValue * tGass2value.fmean;
					fdenominator += tGass2value.fmean * tGass2value.fmean;

					// no division by zero allowed
					if (fdenominator == 0)
					{
						bShadowRe = 0;
					}
					fRate = fnumerator / fdenominator;

					// if tau < a < 1 then also check the color distortion
					if ((fRate <= 1) && (fRate >= fShadowThreshold))
					{
						fDist2Rate = 0.0f;
						fDistD = fRate * tGass2value.fmean - u8PixelValue;
						fDist2Rate += (fDistD * fDistD);
						if (fDist2Rate < fTb * tGass2value.fVar * fRate * fRate)
						{
							bShadowRe = 1;
						}
					}

					fWeight += tGass2value.fWeight;
					if (fWeight > fForegroundThreshod)
					{
						bShadowRe = 0;
					}
				}
				bShadowRe = 0;
#endif
			}
			else
			{
				bShadowRe = 0;
			}
			if (bShadowRe == 1)
			{
				*pu8ForgroundPixel = 0;
			}
			else
			{
				*pu8ForgroundPixel = 255;
			}
		}
	}
}

l32 BGMGuassMogProcess_cuda(void *pvModelParam, u8 *pu8CurrentImage, u8 *puForegroundImage)
{
	l32 l32Row, l32Col, l32Temp, l32I;
	l32 bBackground, l32Fg, l32Diff;
	TGuass2Model* ptGuassModel;
	TGuass2Value* ptGuassValue;
	u8* pu8GuassUsed;
	u8* pu8CurrentPixel;
	u8* pu8ForgroundPixel;
	l32 bShadowRe;

	if (pvModelParam == NULL || pu8CurrentImage == NULL)
	{
		printf("input param pu8CurrentImage == NULL\n");
		return ERR_VEH_DETECT_PROCESS_INPUT_PARAM;
	}

	ptGuassModel = (TGuass2Model*)pvModelParam;
	ptGuassValue = &ptGuassModel->ptGuassValue[0];
	pu8GuassUsed = &ptGuassModel->puUsedGuassNum[0];
	pu8CurrentPixel = &pu8CurrentImage[0];
	pu8ForgroundPixel = &ptGuassModel->pu8ForgroundImage[0];
	ptGuassModel->l32Count++;
	l32Temp = MIN(2 * ptGuassModel->l32Count, BGFG_MOG2_HISTORY);
	ptGuassModel->fLearnRate = 1 / (f32)l32Temp;

	//估计光照
	BGGuass2EstimateLum(ptGuassModel, pu8CurrentPixel);

	//根据估计结果检测前景
	for (l32I = 0; l32I < ptGuassModel->l32Size; l32I++)
	{
		l32Fg = pu8CurrentPixel[l32I];
		if (l32Fg == 0)
		{
			//为零的前景认为是超出roi区域的
			puForegroundImage[l32I] = 0;
		}
		else
		{
			bBackground = ptGuassModel->pu8BackgroundImage[l32I] + ptGuassModel->l32BgLumAdj;
			l32Diff = (l32Fg > bBackground) ? (l32Fg - bBackground) : (bBackground - l32Fg);

			if (l32Diff > 20)
			{
				puForegroundImage[l32I] = 255;
				if ((l32Fg < bBackground) && (l32Fg >(bBackground * 50 / 100)))
				{
					puForegroundImage[l32I] = 128;
				}
			}
			else
			{
				puForegroundImage[l32I] = 0;
			}
		}
	}

	dim3 dim3Block_rect((ptGuassModel->l32ImageWidth+31) / 32, (ptGuassModel->l32ImageHeight+7) / 8, 1);
	dim3 dim3threads_rect(32, 8, 1);
	kerUpdatePixelBackgroundGMM2<< <dim3Block_rect, dim3threads_rect >> >(
		ptGuassModel->ptGuassValue,
		pu8CurrentImage,
		ptGuassModel->puUsedGuassNum,
		ptGuassModel->pu8ForgroundImage,
		ptGuassModel->fLearnRate,
		ptGuassModel->fForegroundThreshod,
		ptGuassModel->fDeviationThreshold,
		ptGuassModel->fTb,
		ptGuassModel->fCT,
		ptGuassModel->bShadowDetection,
		ptGuassModel->fShadowThreshold,
		ptGuassModel->l32ImageWidth,
		ptGuassModel->l32ImageHeight,
		ptGuassModel->l32ImageWidth
		);
	checkCudaErrors(cudaDeviceSynchronize());

	//memcpy(puForegroundImage , ptGuassModel->pu8ForgroundImage, ptGuassModel->l32ImageWidth * ptGuassModel->l32ImageHeight);
	BGMGuassMogGetBGImage(ptGuassModel);
	return EStatus_Success;
}
__global__ void BilinearZoom_CheckBoundary_Kernel(u8 *pu8Image1, u8 *pu8Image2,
	u32 u32SrcWidth, u32 u32SrcHeight, u32 u32SrcStride, u32 u32XStride, u32 u32XPositionInit,
	u32 u32DstWidth, u32 u32DstHeight, u32 u32DstStride, u32 u32YStride, u32 u32YPositionInit)
{
	u32 u32YSrc, u32RowIndex, u32LineIndex, u32WY2, u32WY1;
	u32 u32XPosition, u32XSrc, u32WX2, u32WX1, u32Vn0, u32Vn1;
	u8 *pu8SrcLine1;
	u8 *pu8SrcLine2;
	u8 *pu8Dst;


	u32LineIndex = (blockIdx.x) * blockDim.x + threadIdx.x;
	u32RowIndex = (blockIdx.y) * blockDim.y + threadIdx.y;
	if(u32LineIndex<u32DstWidth && u32RowIndex<u32DstHeight){
		u32 u32YPosition = u32YPositionInit + (u32RowIndex)*u32YStride;
		u32YSrc = u32YPosition >> 16;

		u32WY2 = (u32YPosition << 16) >> 29;
		u32WY1 = 8 - u32WY2;

		u32WY2 *= 1024;
		u32WY1 *= 1024;

		pu8SrcLine1 = pu8Image1 + u32YSrc * u32SrcStride;
		pu8SrcLine2 = u32WY2 == 0 ? pu8SrcLine1 : pu8SrcLine1 + u32SrcStride; 
		pu8Dst = pu8Image2 + u32RowIndex * u32DstStride + u32LineIndex;


		u32XPosition = u32XPositionInit + (u32LineIndex)*u32XStride;
	
		u32XSrc = u32XPosition >> 16;
		u32WX2 = (u32XPosition << 16) >> 29;
		u32WX1 = 8 - u32WX2;
		u32Vn0 = (pu8SrcLine1[u32XSrc] * u32WX1 + pu8SrcLine1[u32XSrc + 1] * u32WX2);
		u32Vn1 = (pu8SrcLine2[u32XSrc] * u32WX1 + pu8SrcLine2[u32XSrc + 1] * u32WX2);
		*pu8Dst = (u8)((u32Vn0 * u32WY1 + u32Vn1 * u32WY2 + 0x8000) >> 16);
	}
}


void BilinearZoom_c_cuda(u8 *pu8Image1, u8 *pu8Image2,
	u32 u32SrcWidth, u32 u32SrcHeight, u32 u32SrcStride,
	u32 u32DstWidth, u32 u32DstHeight, u32 u32DstStride)
{

	u32 u32XStride, u32YStride;

	u32 u32YPositionInit = 0;
	u32 u32XPositionInit = 0;


	u32XStride = ((u32SrcWidth - 1) << 16) / (u32DstWidth - 1);//pow(2,16)*Src_width/Dst_width
	u32YStride = ((u32SrcHeight - 1) << 16) / (u32DstHeight - 1);//pow(2,16)*Src_height/Dst_height

	if (0 == ((u32XStride << 16) >> 27))
	{
		u32XStride = ((u32SrcWidth - 2) << 16) / (u32DstWidth - 1);//pow(2,16)*Src_width/Dst_width
		u32XPositionInit = 5 << 15;
	}

	if ((u32SrcHeight != u32DstHeight) && (0 == ((u32YStride << 16) >> 27)))
	{
		u32YStride = ((u32SrcHeight - 2) << 16) / (u32DstHeight - 1);//pow(2,16)*Src_height/Dst_height
		u32YPositionInit = 1 << 15;
	}

	dim3 dimGrid((u32DstWidth+31) / 32, (u32DstHeight+7)/8);
	dim3 dimBlock(32, 8);
	BilinearZoom_CheckBoundary_Kernel << <dimGrid, dimBlock >> >(pu8Image1, pu8Image2,
		u32SrcWidth, u32SrcHeight, u32SrcStride, u32XStride, u32XPositionInit,
		u32DstWidth, u32DstHeight, u32DstStride, u32YStride, u32YPositionInit);
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void kerBilinear_downgray(u8 *pu8Image1, u8 *pu8Image2,
	u32 u32SrcWidth, u32 u32SrcHeight, u32 u32SrcStride, u32 u32XStride, u32 u32XPositionInit,
	u32 u32DstWidth, u32 u32DstHeight, u32 u32DstStride, u32 u32YStride, u32 u32YPositionInit,
	u8 *pu8GrayImg,u32 u32GrayWidth,u32 u32GrayHeight,u32 u32GrayStride)
{
	u32 u32YSrc, u32RowIndex, u32LineIndex, u32WY2, u32WY1;
	u32 u32XPosition, u32XSrc, u32WX2, u32WX1, u32Vn0, u32Vn1;
	u8 *pu8SrcLine1;
	u8 *pu8SrcLine2;
	u8 *pu8Dst;
	u32 indexX,indexY;
	u32 sum[4],i,j;
	indexX = (blockIdx.x) * blockDim.x + threadIdx.x;
	indexY = (blockIdx.y) * blockDim.y + threadIdx.y;

	sum[0] = 0;
	sum[1] = 0;
	sum[2] = 0;
	sum[3] = 0;
	for(i=0;i<4;i++)
		for(j=0;j<4;j++)
		{
	u32LineIndex = indexX*4+j;
	u32RowIndex = indexY*4+i;
	u32 u32YPosition = u32YPositionInit + (u32RowIndex)*u32YStride;
	u32YSrc = u32YPosition >> 16;

	//垂直方向权值
	u32WY2 = (u32YPosition << 16) >> 29;
	u32WY1 = 8 - u32WY2;

	//放大权值以利用_dotprsu2指令右移16位特点
	u32WY2 *= 1024;
	u32WY1 *= 1024;

	pu8SrcLine1 = pu8Image1 + u32YSrc * u32SrcStride;
	pu8SrcLine2 = u32WY2 == 0 ? pu8SrcLine1 : pu8SrcLine1 + u32SrcStride; //在最后一行的时候，读的数据是不参与运算的，但VC不允许越界读，因此这里加判断防止读越界

	pu8Dst = pu8Image2 + u32RowIndex * u32DstStride + u32LineIndex;//u32DstWidth;

	u32XPosition = u32XPositionInit + (u32LineIndex)*u32XStride;
	//定点数的高16位就是整数部分
	u32XSrc = u32XPosition >> 16;

	//定点数的低16位就是小数部分，我们使用其高3位作为权值(范围0-8)
	u32WX2 = (u32XPosition << 16) >> 29;
	u32WX1 = 8 - u32WX2;

	//水平线性滤波--下面的操作对应于_dotpu4指令
	u32Vn0 = (pu8SrcLine1[u32XSrc] * u32WX1 + pu8SrcLine1[u32XSrc + 1] * u32WX2);
	u32Vn1 = (pu8SrcLine2[u32XSrc] * u32WX1 + pu8SrcLine2[u32XSrc + 1] * u32WX2);

	//垂直线性滤波--下面的操作对应于_dotprsu2指令
	*pu8Dst = (u8)((u32Vn0 * u32WY1 + u32Vn1 * u32WY2 + 0x8000) >> 16);
	//sum[i] = sum[i] + (u32)(*pu8Dst);
	sum[i] += *pu8Dst;
		}


	pu8GrayImg[indexY*u32GrayStride+indexX] = (((sum[0]+2)>>2)+((sum[1]+2)>>2)+((sum[2]+2)>>2)+((sum[3]+2)>>2)+2)>>2;
}

void BilinearZoom_c_DownSample4x4GRAY_cuda(u8 *pu8Image1, u8 *pu8Image2,
	u32 u32SrcWidth, u32 u32SrcHeight, u32 u32SrcStride,
	u32 u32DstWidth, u32 u32DstHeight, u32 u32DstStride, u8 *pu8GrayImg)
{
	u32 u32XStride, u32YStride;
	u32 u32YPositionInit = 0;
	u32 u32XPositionInit = 0;
	u32XStride = ((u32SrcWidth - 1) << 16) / (u32DstWidth - 1);//pow(2,16)*Src_width/Dst_width
	u32YStride = ((u32SrcHeight - 1) << 16) / (u32DstHeight - 1);//pow(2,16)*Src_height/Dst_height
	dim3 dimGrid(u32DstWidth/ 64, u32DstHeight/64);
	dim3 dimBlock(16, 16);

	u32 u32GrayWidth = u32DstWidth/4;
	u32 u32GrayHeight = u32DstHeight/4;
	u32 u32GrayStride = u32GrayWidth;
	kerBilinear_downgray << <dimGrid, dimBlock >> >(pu8Image1, pu8Image2,
		u32SrcWidth, u32SrcHeight, u32SrcStride, u32XStride, u32XPositionInit,
		u32DstWidth, u32DstHeight, u32DstStride, u32YStride, u32YPositionInit,
		pu8GrayImg,u32GrayWidth,u32GrayHeight,u32GrayStride);
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void kerYUV2RGB(u8 *pu8SrcY, u8 *pu8SrcU, u8 *pu8SrcV, u8 *pu8Dst, l32 l32Width, l32 l32Height, l32 l32YStride, l32 l32UVStride, l32 l32RGBStride)
{
	int xx, yy;
	gpu_type gY[4];
	gpu_type gU, gV, gR, gG, gB;
	u8 *pDst[4];
	int i;

	xx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	yy = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	pDst[0] = pu8Dst + (l32Height - 1 - yy) * l32RGBStride + xx * 3;
	pDst[1] = pu8Dst + (l32Height - 1 - yy) * l32RGBStride + xx * 3 + 3;
	pDst[2] = pu8Dst + (l32Height - 1 - yy - 1) * l32RGBStride + xx * 3;
	pDst[3] = pu8Dst + (l32Height - 1 - yy - 1) * l32RGBStride + xx * 3 + 3;


	gY[0] = pu8SrcY[yy * l32YStride + xx] - 16;
	gY[1] = pu8SrcY[yy * l32YStride + xx + 1] - 16;
	gY[2] = pu8SrcY[(yy + 1) * l32YStride + xx] - 16;
	gY[3] = pu8SrcY[(yy + 1) * l32YStride + xx + 1] - 16;
	gU = pu8SrcU[yy / 2 * l32UVStride + xx / 2] - 128;
	gV = pu8SrcV[yy / 2 * l32UVStride + xx / 2] - 128;
	
	for (i = 0; i < 4; i++) {
		gR = (298 * gY[i] + 516 * gU + 128) / (gpu_type)256;
		gG = (298 * gY[i] - 100 * gU - 208 * gV + 128) / (gpu_type)256;
		gB = (298 * gY[i] + 409 * gV + 128) / (gpu_type)256;
#ifdef __GPU_FLOAT__
		gR = fmax(0, gR); gR = fmin(255, gR);
		gG = fmax(0, gG); gG = fmin(255, gG);
		gB = fmax(0, gB); gB = fmin(255, gB);
#else
		gR = max(0, gR); gR = min(255, gR);
		gG = max(0, gG); gG = min(255, gG);
		gB = max(0, gB); gB = min(255, gB);
#endif
		
		pDst[i][0] = gR;
		pDst[i][1] = gG;
		pDst[i][2] = gB;
	}
}


void YUV2RGB24Roi_cuda(stImage *ptImage, u8 *pu8RGB24Dst, l32 l32DstStride)
{
	l32 l32Width, l32Height;
	u8 *pu8Y = (u8 *)ptImage->pu8Y;
	u8 *pu8U = (u8 *)ptImage->pu8U;
        u8 *pu8V = (u8 *)ptImage->pu8V;

	l32Width = ptImage->l32Width;
	l32Height = ptImage->l32Height;

    	kerYUV2RGB<<<dim3(l32Width / (2 * 32), l32Height / (2 * 8)), dim3(32, 8)>>>(pu8Y, pu8U, pu8V, pu8RGB24Dst, 
						l32Width, l32Height, l32Width, l32Width / 2, l32DstStride);
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void kerDia0(u8 *pu8Src, u8 *pu8Dst, l32 l32Width, l32 l32Stride)
{
#define MAX_OFFSET 3

	u8 u8Src0[POINT_PER_THREAD + 2 * MAX_OFFSET], u8Src1[POINT_PER_THREAD + 2 * MAX_OFFSET];
	__shared__ u8 u8Sh[(16 + 1) * 2 * MAX_OFFSET * 8];	//warning: the number "16" should be equal to blockDim.x and "8" should be equal to blockDim.y
	u8 *ptr0, *ptr1, *pSh;
	u8 umin, umax;	
	int i, j;
	int offset, pointPerThread;
	
	pointPerThread = POINT_PER_THREAD;
	pu8Src = pu8Src + (blockDim.y * blockIdx.y + threadIdx.y) * l32Stride + (threadIdx.x + blockIdx.x * blockDim.x) * pointPerThread; 
	pu8Dst = pu8Dst + (blockDim.y * blockIdx.y + threadIdx.y) * l32Width +  (threadIdx.x + blockIdx.x * blockDim.x) * pointPerThread; 

	ptr0 = u8Src0 + MAX_OFFSET;
	ptr1 = u8Src1 + MAX_OFFSET;
	for (i = -1 * MAX_OFFSET; i < pointPerThread + MAX_OFFSET; i++) {
		ptr0[i] = pu8Src[i];	
	}

	//dialate 1
	offset = 1;
	for (i = 0; i < pointPerThread; i++ ) {
		umax = 0;
		for (j = offset; j > 0; j--) {
			if (umax < ptr0[i + j]) {
				umax = ptr0[i + j]; 
			}
			if (umax < ptr0[i - j]) {
				umax = ptr0[i - j];
			}
		}	
		if (umax < ptr0[i]) {
			umax = ptr0[i];
		}
		ptr1[i] = umax;
	}

	//dialate 2
	offset = 3;
	pSh = offset * (1 + 2 * threadIdx.x + 2 * (1 + blockDim.x) * threadIdx.y) +  u8Sh;
	for (i = 0; i < offset; i++) {
		pSh[i] = ptr1[i];
		pSh[i + offset] = ptr1[pointPerThread - offset + i];
	}	//store edge data
	__syncthreads();
//revise the edge
	if (blockIdx.x == 0 && threadIdx.x == 0) { 
		for (i = 1; i <= offset; i++) {
			pSh[0 - i] = pSh[0];
		}	
	}
	if (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
		for (i = 2 * offset; i < 3 * offset; i++) {
			pSh[i] = pSh[offset * 2 - 1];
		}
	}
	__syncthreads();
	for (i = 0; i < offset; i++) {
		ptr1[i - offset] = pSh[i - offset];
		ptr1[pointPerThread + i] = pSh[i + 2 * offset];
	}  //load edge data
	__syncthreads();

	for (i = 0; i < pointPerThread; i++ ) {
		umin = 255;
		for (j = offset; j > 0; j--) {
			if (umin > ptr1[i + j]) {
				umin = ptr1[i + j]; 
			}
			if (umin > ptr1[i - j]) {
				umin = ptr1[i - j];
			}
		}	
		if (umin > ptr1[i]) {
			umin = ptr1[i];
		}
		ptr0[i] = umin;
		pu8Dst[i] = umin;
	}

	//dialate 3
	offset = 2;
	pSh = offset * (1 + 2 * threadIdx.x + 2 * (1 + blockDim.x) * threadIdx.y) +  u8Sh;
	for (i = 0; i < offset; i++) {
		pSh[i] = ptr0[i];
		pSh[i + offset] = ptr0[pointPerThread - offset + i];
	}	//store edge data
	__syncthreads();
//revise the edge
	if (blockIdx.x == 0 && threadIdx.x == 0) { 
		for (i = 1; i <= offset; i++) {
			pSh[0 - i] = pSh[0];
		}	
	}
	if (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
		for (i = 2 * offset; i < 3 * offset; i++) {
			pSh[i] = pSh[offset * 2 - 1];
		}
	}
	__syncthreads();
	for (i = 0; i < offset; i++) {
		ptr0[i - offset] = pSh[i - offset];
		ptr0[pointPerThread + i] = pSh[i + 2 * offset];
	}  //load edge data
	__syncthreads();
	for (i = 0; i < pointPerThread; i++ ) {
		umax = 0;
		for (j = offset; j > 0; j--) {
			if (umax < ptr0[i + j]) {
				umax = ptr0[i + j]; 
			}
			if (umax < ptr0[i - j]) {
				umax = ptr0[i - j];
			}
		}	
		if (umax < ptr0[i]) {
			umax = ptr0[i];
		}
		pu8Src[i] = umax;
		pu8Dst[i] = umax;
	}

#undef MAX_OFFSET
}

__global__ void kerDia1(u8 *pu8Src, u8 *pu8Dst, l32 l32Width, l32 l32Stride)
{
#define MAX_OFFSET 4

	u8 u8Src0[POINT_PER_THREAD + 2 * MAX_OFFSET], u8Src1[POINT_PER_THREAD + 2 * MAX_OFFSET];
	__shared__ u8 u8Sh[(16 + 1) * 2 * MAX_OFFSET * 8];	//warning: the number "16" should be equal to blockDim.x and "8" should be equal to blockDim.y
	u8 *ptr0, *ptr1, *pSh;
	u8 umin, umax;	
	int i, j;
	int offset, pointPerThread;
	
	pointPerThread = POINT_PER_THREAD;
	pu8Src = pu8Src + (blockDim.y * blockIdx.y + threadIdx.y) * l32Stride + (threadIdx.x + blockIdx.x * blockDim.x) * pointPerThread; 
	pu8Dst = pu8Dst + (blockDim.y * blockIdx.y + threadIdx.y) * l32Width +  (threadIdx.x + blockIdx.x * blockDim.x) * pointPerThread; 

	ptr0 = u8Src0 + MAX_OFFSET;
	ptr1 = u8Src1 + MAX_OFFSET;
	for (i = -1 * MAX_OFFSET; i < pointPerThread + MAX_OFFSET; i++) {
		ptr0[i] = pu8Src[i];	
	}

	//dialate 1
	offset = 2;
	for (i = 0; i < pointPerThread; i++ ) {
		umax = 0;
		for (j = offset; j > 0; j--) {
			if (umax < ptr0[i + j]) {
				umax = ptr0[i + j]; 
			}
			if (umax < ptr0[i - j]) {
				umax = ptr0[i - j];
			}
		}	
		if (umax < ptr0[i]) {
			umax = ptr0[i];
		}
		ptr1[i] = umax;
	}

	//dialate 2
	offset = 4;
	pSh = offset * (1 + 2 * threadIdx.x + 2 * (1 + blockDim.x) * threadIdx.y) +  u8Sh;
	for (i = 0; i < offset; i++) {
		pSh[i] = ptr1[i];
		pSh[i + offset] = ptr1[pointPerThread - offset + i];
	}	//store edge data
	__syncthreads();
//revise the edge
	if (blockIdx.x == 0 && threadIdx.x == 0) { 
		for (i = 1; i <= offset; i++) {
			pSh[0 - i] = pSh[0];
		}	
	}
	if (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
		for (i = 2 * offset; i < 3 * offset; i++) {
			pSh[i] = pSh[offset * 2 - 1];
		}
	}
	__syncthreads();
	for (i = 0; i < offset; i++) {
		ptr1[i - offset] = pSh[i - offset];
		ptr1[pointPerThread + i] = pSh[i + 2 * offset];
	}  //load edge data
	__syncthreads();

	for (i = 0; i < pointPerThread; i++ ) {
		umin = 255;
		for (j = offset; j > 0; j--) {
			if (umin > ptr1[i + j]) {
				umin = ptr1[i + j]; 
			}
			if (umin > ptr1[i - j]) {
				umin = ptr1[i - j];
			}
		}	
		if (umin > ptr1[i]) {
			umin = ptr1[i];
		}
		ptr0[i] = umin;
		pu8Dst[i] = umin;
	}

	//dialate 3
	offset = 2;
	pSh = offset * (1 + 2 * threadIdx.x + 2 * (1 + blockDim.x) * threadIdx.y) +  u8Sh;
	for (i = 0; i < offset; i++) {
		pSh[i] = ptr0[i];
		pSh[i + offset] = ptr0[pointPerThread - offset + i];
	}	//store edge data
	__syncthreads();
//revise the edge
	if (blockIdx.x == 0 && threadIdx.x == 0) { 
		for (i = 1; i <= offset; i++) {
			pSh[0 - i] = pSh[0];
		}	
	}
	if (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
		for (i = 2 * offset; i < 3 * offset; i++) {
			pSh[i] = pSh[offset * 2 - 1];
		}
	}
	__syncthreads();
	for (i = 0; i < offset; i++) {
		ptr0[i - offset] = pSh[i - offset];
		ptr0[pointPerThread + i] = pSh[i + 2 * offset];
	}  //load edge data
	__syncthreads();
	for (i = 0; i < pointPerThread; i++ ) {
		umax = 0;
		for (j = offset; j > 0; j--) {
			if (umax < ptr0[i + j]) {
				umax = ptr0[i + j]; 
			}
			if (umax < ptr0[i - j]) {
				umax = ptr0[i - j];
			}
		}	
		if (umax < ptr0[i]) {
			umax = ptr0[i];
		}
		pu8Src[i] = umax;
		pu8Dst[i] = umax;
	}

#undef MAX_OFFSET
}

void roughPart4(u8 *pu8Src, u8 *pu8Dst, l32 l32Width, l32 l32Height, l32 l32Stride)
{
    kerDia0<<<dim3(l32Width / (POINT_PER_THREAD * 16), l32Height / (8 * 2)), dim3(16, 8)>>>(pu8Src + 6, pu8Dst, l32Width, l32Stride);	//the dimension should not be changed!
    kerDia1<<<dim3(l32Width / (POINT_PER_THREAD * 16), l32Height / (8 * 2)), dim3(16, 8)>>>(pu8Src + 6 + (l32Width + 12) * l32Height / 2, pu8Dst + l32Width * l32Height / 2, l32Width, l32Stride);
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void kerLPFilter(u8 *pu8Src, u8 *pu8Dst, l32 l32Max, l32 l32Width, l32 l32Stride)
{
#define FILTER_LENGTH 13
	gpu_type qu[FILTER_LENGTH];
	gpu_type lpf[]={3, 5, 5, 5, 5, 6, 6, 6, 5, 5, 5, 5, 3};
	gpu_type gSum;
	int head;
	int i, j;
	int pointPerThread;

	pointPerThread = POINT_PER_THREAD;
	pu8Src = pu8Src + (blockDim.y * blockIdx.y + threadIdx.y) * l32Stride + (threadIdx.x + blockIdx.x * blockDim.x) * pointPerThread; 
	pu8Dst = pu8Dst + (blockDim.y * blockIdx.y + threadIdx.y) * l32Stride + (threadIdx.x + blockIdx.x * blockDim.x) * pointPerThread; 

	for (i = -6; i <= 6; i++) {
		qu[i + 6] = *(pu8Src + i) * 250 / l32Max;	
	}
	
	head = 0; 
	for (i = 0; i < pointPerThread; i++) {
		gSum = 0;
		for (j = 0; j < FILTER_LENGTH; j++) {
			gSum += qu[(j + head) % FILTER_LENGTH] * lpf[j];	
		}
		pu8Dst[i] = gSum / 64;
		
		qu[(j + head) % FILTER_LENGTH] = pu8Src[i + (FILTER_LENGTH - 1) / 2 + 1] * 250 / l32Max;	
		head++;	
	}

#undef FILTER_LENGTH 
}

void roughPart6(u8 *pu8Src, u8 *pu8Dst, l32 l32Max, l32 l32Width, l32 l32Height, l32 l32Stride)
{
    kerLPFilter<<<dim3(l32Width / (16 * POINT_PER_THREAD), l32Height / 4), dim3(16, 4)>>>(pu8Src + 6, pu8Dst + 6, l32Max, l32Width, l32Stride);
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void kerGetFeature_opt(u8 *pu8RGB24, l32 l32Width, l32 l32Height, l32 l32Stride, u8 *pu8FeatureImage)
{
	l32 xx = blockIdx.x * blockDim.x + threadIdx.x;
	l32 yy = blockIdx.y * blockDim.y + threadIdx.y;
	l32 xx_, xx0, xx1, yy_, yy0, yy1, bxx1;
	u8 *pSrc_, *pSrc0, *pSrc1;
	gpu_type f__, f_0, f_1;
	gpu_type f0_, f00, f01;
	gpu_type f1_, f10, f11;
	gpu_type b__, b_0, b_1;
	gpu_type b0_, b00, b01;
	gpu_type tmp_, tmp0, tmp1;
	gpu_type res1, res2;
	gpu_type fsum;
	int i, j;

	yy0 = yy * 4;
	yy_ = max(0, yy0 - 1);
	yy1 = min(l32Height - 1, yy0 + 1);
	pSrc_ = yy_ * l32Stride + pu8RGB24;
	pSrc0 = yy0 * l32Stride + pu8RGB24;
	pSrc1 = yy1 * l32Stride + pu8RGB24;

	xx0 = xx * 4;
	xx_ = max(0, xx0 - 1);
	xx1 = min(l32Width - 1, xx0 + 1);
	bxx1 = xx1;

	f__ = (gpu_type)pSrc_[xx_]; f_0 = (gpu_type)pSrc_[xx0]; f_1 = (gpu_type)pSrc_[xx1];
	f0_ = (gpu_type)pSrc0[xx_]; f00 = (gpu_type)pSrc0[xx0]; f01 = (gpu_type)pSrc0[xx1];
	f1_ = (gpu_type)pSrc1[xx_]; f10 = (gpu_type)pSrc1[xx0]; f11 = (gpu_type)pSrc1[xx1];

	fsum = 0;
	for (j = 0; j < 4; j++) {
		b__ = f0_; b_0 = f00; b_1 = f01;
		b0_ = f1_; b00 = f10; b01 = f11;

		for (i = 0; i < 4; i++) {
#ifdef	__GPU_FLOAT__
			tmp_ = fabs(f__ - f_1); tmp0 = fabs(f0_ - f01); tmp1 = fabs(f1_ - f11);
#else
			tmp_ = abs(f__ - f_1); tmp0 = abs(f0_ - f01); tmp1 = abs(f1_ - f11);
#endif
			res1 = ((tmp_ + tmp0 + 1) / 2 + (tmp0 + tmp1 + 1) / 2 + 1) / 2;

#ifdef	__GPU_FLOAT__
			tmp_ = fabs(f__ - f1_); tmp0 = fabs(f_0 - f10); tmp1 = fabs(f_1 - f11);
#else
			tmp_ = abs(f__ - f1_); tmp0 = abs(f_0 - f10); tmp1 = abs(f_1 - f11);
#endif
			res2 = ((tmp_ + tmp0 + 1) / 2 + (tmp0 + tmp1 + 1) / 2 + 1) / 2;

			if (res1 < res2) {
				res1 = 0.0;
			}
			fsum += res1;

			xx1 = min(l32Width - 1, xx1 + 1);
			f__ = f_0; f_0 = f_1; f_1 = (gpu_type)pSrc_[xx1];
			f0_ = f00; f00 = f01; f01 = (gpu_type)pSrc0[xx1];
			f1_ = f10; f10 = f11; f11 = (gpu_type)pSrc1[xx1];			
		}
		
		f__ = b__; f_0 = b_0; f_1 = b_1;
		f0_ = b0_; f00 = b00; f01 = b01;
	
		pSrc_ = pSrc0;
		pSrc0 = pSrc1;
		xx1 = bxx1;
		yy1 = min(l32Height - 1, yy1 + 1);
		pSrc1 = yy1 * l32Stride + pu8RGB24;
		f1_ = (gpu_type)pSrc1[xx_]; f10 = (gpu_type)pSrc1[xx0]; f11 = (gpu_type)pSrc1[xx1];
	}

	fsum /= (gpu_type)8;
	
#ifdef	__GPU_FLOAT__
	fsum = fmin(fsum, (gpu_type)255.0);
#else
	fsum = min(fsum, (gpu_type)255);
#endif

	l32Stride /= 4;
	(pu8FeatureImage + yy * l32Stride)[xx] = (u8)fsum;
}

void GetFeatureYUVImage_SobelDownsample4x4_cuda(u8 *pu8RGB24, l32 l32Width, l32 l32Height, l32 l32Stride,
	u8 *pu8FeatureImage, u8 *pu8Temp)
{
	kerGetFeature_opt<<<dim3(l32Width / (4 * 32), l32Height / (4 * 8)), dim3(32, 8)>>>(pu8RGB24, l32Width, l32Height, l32Stride, pu8FeatureImage);
	checkCudaErrors(cudaDeviceSynchronize());
}

void DilateLine(u8 *pu8Src, u8 *pu8Dst, l32 l32Width, l32 l32DilateEleWidth)
{
    l32 l32x, l32i1, l32i2, l32i, l32vmin, l32vmax;
    l32 l32EleWidth;
    l32 l32IsErode;

    if(l32DilateEleWidth > 0)
    {
        l32EleWidth = l32DilateEleWidth;
        l32IsErode = 0;
    }
    else
    {
        l32EleWidth = -l32DilateEleWidth;
        l32IsErode = 1;
    }

    for(l32x = 0; l32x < l32Width; l32x++)
    {
        l32i1 = l32x - l32EleWidth;
        l32i2 = l32x + l32EleWidth;
        if(l32i1 < 0) 
        {
            l32i1 = 0;
        }
        if(l32i2 > l32Width - 1) 
        {
            l32i2 = l32Width - 1;
        }
        
        l32vmin = 255;
        l32vmax = 0;
        if(l32IsErode)
        {
            for(l32i = l32i1; l32i <= l32i2; l32i++)
            {
                if(l32vmin > pu8Src[l32i])
                {
                    l32vmin = pu8Src[l32i];
                }
            }
            pu8Dst[l32x] = l32vmin;
        }
        else
        {
            for(l32i = l32i1; l32i <= l32i2; l32i++)
            {
                if(l32vmax < pu8Src[l32i])
                {
                    l32vmax = pu8Src[l32i];
                }
            }
            pu8Dst[l32x] = l32vmax;
        }
    }
}

l32 VehLprLocMaxProcess_cuda(stLocLcMax * ptLocLcMax, stLocInput *ptLocInput, stLocOutput *ptLocOutput)
{
    l32 l32X, l32Y, l32i0, l32i1, l32i;
    l32 l32Xs, l32Xe, l32Ys, l32Ye;
    u8 *pu8Dst, *pu8Src, *pu8Temp, *pu8TempH, *pu8TempV;
	s16 *ps16SumTmp;
	u8 *pu8SrcTmp;
    l32 l32Width, l32Height;
    l32 l32PlateWidth, l32PlateHeight;
    l32 l32Temp0, l32Temp1, l32Temp2;
  	l32 l32OffSet;
    u32 u32Temp0, u32Temp1;
  	u32 u32Temp2, u32Temp3;
  	u32 u32Temp4, u32Temp5;
  	u32 u32Temp6, u32Temp7;
    l32 l32Val, l32Max;
    l32 l32Xs0,l32Xe0,l32Ys0,l32Ye0;
    l32 l32RectID;
    LCMAX_POS aLCMaxPos[20];
    l32 l32PeakS, l32LcX, l32LcY, l32MeetMax;
	TVehLprLcRect tRoiRect;
	l32 l32DesMapExtStride;
	l32 l32DensMapStride;

	u32 u32V0, u32V1, u32V2, u32V3;
	u32 u32V4, u32V5, u32V6, u32V7;
	u32 u32V8, u32V9, u32V10, u32V11, u32V12;
    u8 *RESTRICT pu8TmpDst1 = NULL;
    u16 *RESTRICT pu16TmpDstV = NULL;
    u16 *RESTRICT pu16TmpDstH = NULL;
    u16 *RESTRICT pu16TmpSrc = NULL;
    u8 *RESTRICT pu8TmpSrc1 = NULL;
    u8 *RESTRICT pu8TmpSrc2 = NULL;
    u8 *RESTRICT pu8TmpSrc3 = NULL;
    s64 Res0, Res1, lMask;
    u64 u64X0, u64X1, u64X2, u64X3, u64X4, u64X5, u64X6, u64X7;
    u64 u64Data0, u64Data1, u64Data2, u64Data3, u64Data4, u64Data5, u64Data6, u64Data7;
    u64 u64Mask0, u64Mask1, u64Mask2, u64Mask3;
    u64 u64Tmp0, u64Tmp1, u64Tmp2, u64Tmp3;
    u64 u64Const9;
    l32 l32LoopA;    
    myTimer timer;

    memset(ptLocLcMax->pu8EdgeDensMap, 0, ptLocInput->l32Width * ptLocInput->l32Height / 16);
    
    for(l32RectID = 0; l32RectID < MAX_LOCNUM; l32RectID++)
    {
        ptLocOutput->ptRect[l32RectID].l32Valid = 0;
    }
	ptLocOutput->l32RectCount = MAX_LOCNUM;


    GetFeatureYUVImage_SobelDownsample4x4_cuda
            (ptLocInput->pu8SrcImg, 
        	    ptLocInput->l32Width, ptLocInput->l32Height, ptLocInput->l32Stride, 
   		      	ptLocLcMax->pu8EdgeDensMapOrg, NULL);

    l32Width = (ptLocInput->l32Width/4);
    l32Height = (ptLocInput->l32Height/4);


/*
	tRoiRect = ptLocLcMax->tRoiRect;
	tRoiRect.l32top = tRoiRect.l32top/4;
	tRoiRect.l32bottom = tRoiRect.l32bottom/4;
	tRoiRect.l32left = tRoiRect.l32left/4;
	tRoiRect.l32right = tRoiRect.l32right/4;


	startTimer();
	pu8Dst = ptLocLcMax->pu8EdgeDensMapOrg;
	for(l32Y = 0; l32Y<l32Height; l32Y++)
    {
		if((l32Y < tRoiRect.l32top)||(l32Y > tRoiRect.l32bottom))
		{
			memset(pu8Dst, 0, l32Width);
		}
		else
		{
			memset(pu8Dst, 0, tRoiRect.l32left);
			memset(pu8Dst + tRoiRect.l32right, 0, l32Width - tRoiRect.l32right);
		}
		pu8Dst += l32Width;
	}
	dispTimer("part1");
*/	

    //Really stupid. 
    for (l32Y = 0; l32Y < l32Height; l32Y++) {
	    ptLocLcMax->pu8EdgeDensMapOrg[l32Y * l32Width + l32Width - 1] = 0;
    }

    pu8Src = ptLocLcMax->pu8EdgeDensMapOrg;
    pu8Dst = ptLocLcMax->pu8EdgeDensMap;
	l32DensMapStride = ((l32Width/2) + 7)& 0xFFFFFFF8;
	l32DesMapExtStride = l32DensMapStride + 2 * EDGE_WIDTH; //多扩展8字节，是为了当(l32Width/2)已经8字节对齐时，边界扩展8字节，便于Dilate操作，利用SIMD优化

    for(l32Y = 0; l32Y<l32Height; l32Y++)
    {
		for (l32X = 0; l32X < EDGE_WIDTH; l32X++) {
			pu8Dst[l32X] = 0;
		}		

        for(; l32X < EDGE_WIDTH + l32Width / 2; l32X++)
        {
            pu8Dst[l32X] = (pu8Src[2 * (l32X - EDGE_WIDTH)] + pu8Src[2 * (l32X - EDGE_WIDTH) + 1] + 1) >> 1;
        }
		for(; l32X < l32DesMapExtStride; l32X++)
		{
			pu8Dst[l32X] = 0;
		}
        pu8Src += l32Width;
		pu8Dst += l32DesMapExtStride;
    }

    l32Width = l32Width / 2;    


	pu8Dst = (u8*)ptLocLcMax->pu8Temp + l32Width; //临时复用此缓存

	ps16SumTmp = (s16 *)ptLocLcMax->pu8Temp;
	ps16SumTmp = (s16 *)(((u32)ps16SumTmp + 32) & (~7));
	memset(ps16SumTmp, 0, sizeof(s16) * l32Width);

	pu8Dst = (u8 *)(ps16SumTmp + l32Width + 16);
	pu8TempH = pu8Dst + l32Width;
	pu8TempV = pu8TempH + l32Width;

	//首先进行预处理的加法
	//for(l32i = l32Y - 8;l32i <= l32Y + 8; l32i++)
#define HEIGHT_RADIUS 8  //定义垂直半径为8
	for(l32X = 0; l32X < l32Width; l32X++)
	{
		l32Temp0 = 0;
		for(l32i = -(HEIGHT_RADIUS + 1); l32i <= (HEIGHT_RADIUS - 1); l32i++)
		{
			l32i0 = l32i;
			if(l32i0 < 0)
			{
				l32i0 = 0;
			}
			if(l32i0 > l32Height - 1) 
			{
				l32i0 = l32Height - 1;
			}
			l32Temp0 += ptLocLcMax->pu8EdgeDensMap[EDGE_WIDTH + l32X + l32i0 * l32DesMapExtStride];
		}
		ps16SumTmp[l32X] = l32Temp0;
	}

	for(l32Y = 0; l32Y < l32Height; l32Y ++)
	{		
		u8 *pu8SrcRowH, *pu8SrcRowT;
		//此种方式l32Width 不能小于 5 否则程序报错
		//但是一般情况下l32Width 不会小于 5，故从性能优化出发，此特种情况不予考虑
		//for(l32i = l32X - 4; l32i <= l32X + 4; l32i++)
		//水平9点滤波

#define WIDTH_RADIUS 4  //定义半径为4
		pu8SrcRowH = ptLocLcMax->pu8EdgeDensMap + EDGE_WIDTH + l32Y * l32DesMapExtStride;
		pu8SrcRowT = pu8SrcRowH + WIDTH_RADIUS;

		//Sum: -5,-4,-3,-2,-1,0,1,2,3
		l32Temp2 = pu8SrcRowH[0] * 6 + 
			pu8SrcRowH[1] + pu8SrcRowH[2] + 
			pu8SrcRowH[3];

		for(l32X = 0; l32X < WIDTH_RADIUS + 1; l32X++)
		{
			l32Temp2 -= *pu8SrcRowH;
			l32Temp2 += *pu8SrcRowT;
			pu8TempH[l32X] = l32Temp2 / (WIDTH_RADIUS * 2 + 1);			
			pu8SrcRowT++;
		}
		for(; l32X < l32Width - (WIDTH_RADIUS + 1); l32X++)
		{
			l32Temp2 -= *pu8SrcRowH;
			l32Temp2 += *pu8SrcRowT;			
			pu8TempH[l32X] = l32Temp2 / (WIDTH_RADIUS * 2 + 1);
			pu8SrcRowH++;
			pu8SrcRowT++;
		}
		for(; l32X < l32Width; l32X++)
		{
			l32Temp2 -= *pu8SrcRowH;
			l32Temp2 += *pu8SrcRowT;			
			pu8TempH[l32X] = l32Temp2 / (WIDTH_RADIUS * 2 + 1);
			pu8SrcRowH++;
		}

		l32i0 = l32Y - (HEIGHT_RADIUS + 1);
		if(l32i0 < 0)
		{
			l32i0 = 0;
		}
		l32i1 = l32Y + HEIGHT_RADIUS;
		if(l32i1 > l32Height - 1)
		{
			l32i1 = l32Height - 1;
		}

		pu8SrcRowH = ptLocLcMax->pu8EdgeDensMap + EDGE_WIDTH + l32i0 * l32DesMapExtStride;
		pu8SrcRowT = ptLocLcMax->pu8EdgeDensMap + EDGE_WIDTH + l32i1 * l32DesMapExtStride;
		for(l32X = 0; l32X < l32Width; l32X++)
		{
			//垂直滤波			
			ps16SumTmp[l32X] -= pu8SrcRowH[l32X];
			ps16SumTmp[l32X] += pu8SrcRowT[l32X];

			l32Temp1 = ps16SumTmp[l32X];
			pu8TempV[l32X] = l32Temp1 / (HEIGHT_RADIUS * 2 + 1);			
		}	

		DilateLine(pu8TempV, pu8Dst,  l32Width, 1);		
	
		for(l32X = EDGE_WIDTH; l32X < l32Width + EDGE_WIDTH; l32X++)
		{
			if(pu8Dst[l32X - EDGE_WIDTH] < pu8TempH[l32X - EDGE_WIDTH])
			{
				ptLocLcMax->pu8EdgeDensMap2[l32X + l32Y * l32DesMapExtStride] = ptLocLcMax->pu8EdgeDensMap[l32X + l32Y * l32DesMapExtStride];
			}
			else
			{
				ptLocLcMax->pu8EdgeDensMap2[l32X + l32Y * l32DesMapExtStride] = 0;
			}
		}			
//edge point is handled here	
		for (l32X = 0; l32X < EDGE_WIDTH; l32X++) {
			ptLocLcMax->pu8EdgeDensMap2[l32X + l32Y * l32DesMapExtStride] = ptLocLcMax->pu8EdgeDensMap2[EDGE_WIDTH + l32Y * l32DesMapExtStride];
			ptLocLcMax->pu8EdgeDensMap2[EDGE_WIDTH + l32Width + l32X + l32Y * l32DesMapExtStride] = ptLocLcMax->pu8EdgeDensMap2[EDGE_WIDTH + l32Width - 1 + l32Y * l32DesMapExtStride];
		}
	}	

    roughPart4(ptLocLcMax->pu8EdgeDensMap2, ptLocLcMax->pu8EdgeDensMapMoph, l32Width, l32Height, l32DesMapExtStride);

    l32Max = 0;
    for(l32Y = 0; l32Y < l32Height; l32Y++)
    {
        pu8Src = ptLocLcMax->pu8EdgeDensMap2 + l32Y * l32DesMapExtStride;
        for(l32X = EDGE_WIDTH; l32X < l32Width + EDGE_WIDTH; l32X++)
        {
            if(pu8Src[l32X] < 10) {
				pu8Src[l32X] = 0; 
			}
            if(l32Max < pu8Src[l32X]) {
				l32Max = pu8Src[l32X];
			}
        }

	for (l32X = 0; l32X < EDGE_WIDTH; l32X++) {
		pu8Src[l32X] = pu8Src[EDGE_WIDTH];
	} 		
	for (l32X = l32Width + EDGE_WIDTH; l32X < l32Width + EDGE_WIDTH * 2; l32X++) {
		pu8Src[l32X] = pu8Src[l32Width + EDGE_WIDTH - 1];
	}
    }

	if(l32Max < 0) {
		l32Max = 250;
	}    

	roughPart6(ptLocLcMax->pu8EdgeDensMap2, ptLocLcMax->pu8EdgeDensMap, l32Max, l32Width, l32Height, l32DesMapExtStride);

    //下面是一个垂直平滑滤波，权值1 2 1
    for(l32Y = 0; l32Y < l32Height; l32Y++)
    {
//        pu8Src = ptLocLcMax->pu8EdgeDensMap + l32DesMapExtStride + EDGE_WIDTH + l32Y * l32DesMapExtStride;
        pu8Src = ptLocLcMax->pu8EdgeDensMap + EDGE_WIDTH + l32Y * l32DesMapExtStride;
        pu8Dst = ptLocLcMax->pu8EdgeDensMap2 + EDGE_WIDTH + l32Y * l32DesMapExtStride;
        if(l32Y == 0)
        {
            for(l32X = 0; l32X < l32Width; l32X++)
            {
                pu8Dst[l32X] = (pu8Src[l32X] + pu8Src[l32X + l32DesMapExtStride]) >> 1;
            }
        }
        else
        if(l32Y == l32Height - 1)
        {
            for(l32X=0; l32X<l32Width; l32X++)
            {
                pu8Dst[l32X] = (pu8Src[l32X] + pu8Src[l32X - l32DesMapExtStride]) >> 1;
            }
        }
        else
        {
            for(l32X = 0; l32X < l32Width; l32X++)
            {
                pu8Dst[l32X] = (pu8Src[l32X - l32DesMapExtStride] + 2*pu8Src[l32X] + pu8Src[l32X + l32DesMapExtStride]) >> 2;
            }
        }

        for(l32X = 0; l32X < l32Width; l32X++)
        {
            if(pu8Dst[l32X] < 10) 
            {
                pu8Dst[l32X] = 0;
            }
        }
    }
    
	for(l32Y = 0; l32Y < l32Height; l32Y++)
	{
		memset(ptLocLcMax->pu8EdgeDensMap + EDGE_WIDTH + l32Y * l32DesMapExtStride, 0, l32Width);
	}

      memset(aLCMaxPos, 0, sizeof(aLCMaxPos));
    for(l32Y = 1; l32Y < l32Height; l32Y++)
    {
        pu8Src = ptLocLcMax->pu8EdgeDensMap2 + EDGE_WIDTH + l32Y * l32DesMapExtStride;
        pu8Dst = ptLocLcMax->pu8EdgeDensMap + EDGE_WIDTH + l32Y * l32DesMapExtStride;

        l32PeakS = -1;
        l32Val = 0;
        for(l32X = 1; l32X < l32Width; l32X++)
        {
            if((pu8Src[l32X] < pu8Src[l32X-1]) || (l32X == (l32Width - 1)))
            {
                if(l32Val == 1) 
                {
                    if(l32PeakS < 0) l32PeakS = l32X - 1;

                    l32MeetMax = 0;
                    for(l32i = l32PeakS; l32i < l32X; l32i++)
                    {
                        if(pu8Src[l32i] >= pu8Src[l32i-l32DesMapExtStride] &&
                            pu8Src[l32i] >= pu8Src[l32i-l32DesMapExtStride-1] &&
                            pu8Src[l32i] >= pu8Src[l32i-l32DesMapExtStride+1] &&
                            pu8Src[l32i] >= pu8Src[l32i+l32DesMapExtStride] &&
                            pu8Src[l32i] >= pu8Src[l32i+l32DesMapExtStride-1] &&
                            pu8Src[l32i] >= pu8Src[l32i+l32DesMapExtStride+1])
                        {
                            //pu8Dst[l32i] = 255;
                            l32MeetMax = 1;
                        }
                    }
                    //遇到局部最大值,此段内部中点被视为真实局部最大值
                    if(l32MeetMax)
                    {
                        l32LcX = (l32PeakS + l32X)/2;
                        l32LcY = l32Y;

                        if(pu8Src[l32LcX] > 0)
                        {
                            pu8Dst[l32LcX] = 255;

                            //局部最大值点
                            for(l32i1 = 0;l32i1 < sizeof(aLCMaxPos)/sizeof(aLCMaxPos[0]); l32i1++)
                            {
                                if(aLCMaxPos[l32i1].l32Val < pu8Src[l32LcX])
                                {
                                    //插入位置
                                    break;
                                }
                            }
                            if(l32i1 < sizeof(aLCMaxPos)/sizeof(aLCMaxPos[0]))
                            {
                                //移动出空位
                                for(l32i = sizeof(aLCMaxPos)/sizeof(aLCMaxPos[0])-1;l32i > l32i1; l32i--)
                                {
                                    aLCMaxPos[l32i] = aLCMaxPos[l32i-1];
                                }
                                aLCMaxPos[l32i1].l32PosX = l32X;
                                aLCMaxPos[l32i1].l32PosY = l32LcY;
                                aLCMaxPos[l32i1].l32Val = pu8Src[l32LcX];
                            }
                        }
                    }
                }
                l32PeakS = -1;
                l32Val = -1;
            }
            else
            if(pu8Src[l32X] == pu8Src[l32X - 1])
            {
                if(l32PeakS < 0) 
                {
                    l32PeakS = l32X - 1;
                }
            }
            else
            if(pu8Src[l32X] > pu8Src[l32X - 1])
            {
                l32Val = 1; //上升中
                l32PeakS = -1;
            }
        }
    }


    l32RectID = 0;
    memset(ptLocLcMax->pu8EdgeDensMap2, 0, l32Width * l32Height);
    for(l32i = 0;l32i < sizeof(aLCMaxPos)/sizeof(aLCMaxPos[0]); l32i++)
    {
        l32X = aLCMaxPos[l32i].l32PosX;
        l32Y = aLCMaxPos[l32i].l32PosY;

        if(aLCMaxPos[l32i].l32Val == 0)
        {
            break;
        }
        
        l32Val = 10; //经过尝试，此方法稳定性最好
        
        pu8Src = ptLocLcMax->pu8EdgeDensMapMoph + l32X;
        for(l32Ys = aLCMaxPos[l32i].l32PosY; l32Ys > aLCMaxPos[l32i].l32PosY - 3; l32Ys--)
        {
            if(pu8Src[l32Ys * l32DensMapStride] < l32Val) 
            {
                break;
            }
        }
        for(l32Ye = aLCMaxPos[l32i].l32PosY; l32Ye < aLCMaxPos[l32i].l32PosY + 3; l32Ye++)
        {
            if(pu8Src[l32Ye*l32DensMapStride] < l32Val) 
            {
                break;
            }
        }
/*
        if(ptLocLcMax->l32UpsideDown == 0)
        {
            if(l32Y < l32Height/2)
            {
                l32Ys ++;
            }
        }
        else
        {
            // 目前输入yuv图像上面小，下面大
            if(l32Y > l32Height/2)
            {
                l32Ys ++;
            }
        }
*/	
	//only the following will be exectued
            if(l32Y > l32Height/2)
            {
                l32Ys ++;
            }
        pu8Temp = (u8*)ptLocLcMax->pu8Temp;
        for(l32Xs = 0; l32Xs < l32Width; l32Xs++)
        {
            l32Val = 0;
            for(l32Y = l32Ys; l32Y < l32Ye; l32Y++)
            {
                if(l32Val <  ptLocLcMax->pu8EdgeDensMapMoph[l32Y*l32DensMapStride + l32Xs])
                {
                    l32Val = ptLocLcMax->pu8EdgeDensMapMoph[l32Y*l32DensMapStride + l32Xs];
                }
            }
            l32Val = l32Val * 3;
            if(l32Val > 255) l32Val = 255;
            pu8Temp[l32Xs] = l32Val;
        }
        
        l32Val = pu8Temp[l32X] * 40/100;
        pu8Src = pu8Temp;
        for(l32Xs = aLCMaxPos[l32i].l32PosX; l32Xs > aLCMaxPos[l32i].l32PosX - 11; l32Xs--)
        {
            if(pu8Src[l32Xs] <= l32Val) 
            {
                break;
            }
        }
        for(l32Xe = aLCMaxPos[l32i].l32PosX; l32Xe < aLCMaxPos[l32i].l32PosX + 11; l32Xe++)
        {
            if(pu8Src[l32Xe] <= l32Val) 
            {
                break;
            }
        }

        if((l32Ye <= l32Ys) || (l32Xe <= l32Xs))
        {
            continue;
        }
            
        if(l32Xs < 0) l32Xs = 0;
        if(l32Ys < 0) l32Ys = 0;
        if(l32Xe > l32Width-1) l32Xe = l32Width-1;
        if(l32Ye > l32Height-1) l32Ye = l32Height-1;

        l32Xs0 = l32Xs;
        l32Xe0 = l32Xe;
        l32Ys0 = l32Ys;
        l32Ye0 = l32Ye;

        {
            l32Xs *= 8;
            l32Xe *= 8;
            l32Ys *= 4;
            l32Ye *= 4;
        }
        
        l32PlateHeight = (l32Ye-l32Ys);
        l32PlateWidth = (l32Xe-l32Xs);

        if(aLCMaxPos[l32i].l32Val > 0)
        {
            if(l32RectID < ptLocOutput->l32RectCount)
            {
				l32 l32W,l32H;
				l32W = l32Xe - l32Xs;
				l32H = l32Ye - l32Ys;

				ptLocOutput->ptRect[l32RectID].l32left = l32Xs;
				ptLocOutput->ptRect[l32RectID].l32right = l32Xe;
				ptLocOutput->ptRect[l32RectID].l32top = l32Ys;
				ptLocOutput->ptRect[l32RectID].l32bottom = l32Ye;
				ptLocOutput->ptRect[l32RectID].l32Valid = 1;



                l32RectID ++;
            }
        }
    }
    //最终返回区域个数
    ptLocOutput->l32RectCount = l32RectID;
	
    return 0;
}

void *thread_func_cuda(void *arg)
{
	struct _cuda_thread_arg *th_arg = (struct _cuda_thread_arg *)arg;
	TVehLprLcRect tRect[MAX_LOCNUM] = {0};
	myTimer timer2;
	
	ptLocInput0.l32Width = th_arg->l32Width;
	ptLocInput0.l32Height = th_arg->l32Height;
	ptLocInput0.l32Stride = th_arg->l32Stride;
	
	ptImage0.l32Width = ptLocInput0.l32Width;
	ptImage0.l32Height = ptLocInput0.l32Height;
	
	while (1) {
		sem_wait(&sem_full);			
		if (pu8CudaInputSrc == NULL) {
			printf("cuda exit successfully\n");	
			return NULL;
		}
		timer2.start();
		//copy image source
		//memcpy(pu8CudaImgBuf, pu8CudaInputSrc, (ptLocInput0.l32Width * ptLocInput0.l32Height * 3) >> 1);
		checkCudaErrors(cudaMemcpy(pu8CudaImgBuf, pu8CudaInputSrc, (ptLocInput0.l32Width * ptLocInput0.l32Height * 3) >> 1, cudaMemcpyHostToDevice));
		sem_post(&sem_empty);
		timer2.disp("copy in");

		timer2.start();
		//rough locate
		ptLocInput0.pu8SrcImg = pu8CudaImgBuf;
		ptLocOutput0.ptRect = tRect;
		ptLocOutput0.l32RectCount = MAX_LOCNUM;
		VehLprLocMaxProcess_cuda(&ptLocLcMax0, &ptLocInput0, &ptLocOutput0);

		//yuv2rgb
		ptImage0.pu8Y = pu8CudaImgBuf;
		ptImage0.pu8U = pu8CudaImgBuf + ptLocInput0.l32Width * ptLocInput0.l32Height;
		ptImage0.pu8V = pu8CudaImgBuf + (ptLocInput0.l32Width * ptLocInput0.l32Height * 5) / 4;
		YUV2RGB24Roi_cuda(&ptImage0, pu8CudaRGBBuf, ptLocInput0.l32Width * 3);

		//resize
		BilinearZoom_c_DownSample4x4GRAY_cuda(pu8CudaImgBuf, pu8CudaZoomBuf,
				ptLocInput0.l32Width, ptLocInput0.l32Height, ptLocInput0.l32Stride,
				SCALEWIDTH, SCALEHEIGHT, SCALEWIDTH, pu8CudaGrayBuf);
		TGuass2Model *ptGuassModel = (TGuass2Model*)pvCudaBgModel;
		BilinearZoom_c_cuda(pu8CudaGrayBuf,pu8CudaFrameCurBuf,
				SCALEWIDTH/4,SCALEHEIGHT/4,SCALEWIDTH/4,
				ptGuassModel->l32ImageWidth,ptGuassModel->l32ImageHeight,ptGuassModel->l32ImageWidth);
		
		//bgm
		int state = BGMGuassMogProcess_cuda(ptGuassModel,pu8CudaFrameCurBuf,pu8CudaFGFrameBuf);
		if(state!=EStatus_Success){
			printf("BGMGuass ERROR!\n");
			exit(EXIT_FAILURE);
		}
		timer2.disp("cuda process");

		sem_wait(&sem_finish);
		timer2.start();
		//rough output
		memcpy(&atCudaRoughRectsOut, ptLocOutput0.ptRect, sizeof(TVehLprLcRect) * MAX_LOCNUM);
		l32CudaRectCountOut = ptLocOutput0.l32RectCount;
		//yuv2rgb output
		memcpy(pu8CudaRGBOut, pu8CudaRGBBuf, ptLocInput0.l32Width * ptLocInput0.l32Height * 3);
		//resize output
		memcpy(pu8CudaZoomOut, pu8CudaZoomBuf, SCALEWIDTH * SCALEHEIGHT);
		memcpy(pu8CudaGrayOut, pu8CudaGrayBuf, SCALEWIDTH * SCALEHEIGHT);
		memcpy(pu8CudaFrameCurOut, pu8CudaFrameCurBuf, ptGuassModel->l32ImageWidth * ptGuassModel->l32ImageHeight);
		//bgm output
		memcpy(pu8CudaFGFrameOut, pu8CudaFGFrameBuf, ptGuassModel->l32ImageWidth * ptGuassModel->l32ImageHeight);
		timer2.disp("copy out");
		sem_post(&sem_ready);
	}
}

