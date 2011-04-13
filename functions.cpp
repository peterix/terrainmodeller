#include <stdlib.h>
#include "functions.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

//prevadi 2D souradnice na 1D
long int cord (int x, int y, int width)
{
	return (x+y*width);
}

//prevod bile 255 na WHITECONV
void convertWhite (float * input, int width, int height)
{
	int i, j;
	#pragma omp parallel shared (height, width, input)
	#pragma omp for private (i,j)
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (input[cord(j, i, width)] == 255) input[cord(j, i, width)] = WHITECONV;
		}
	}
}


//prevede body vrstevnice na nejvyssi hodnotu v 8mi okoli
void convertContourLines (float * input, int width, int height)
{	
	cout << "Extracting countour lines: ..." << endl;
	int i, j;
	float max;
	#pragma omp parallel shared (height, width, input)
	#pragma omp for private (i, j, max)
	for (i = 1; i < height-1; i++)
	{
		for (j = 1; j < width-1; j++)
		{
			if (input[cord(j, i, width)] == CONTOURLINE)
			{
				max = 0;
				if (input[cord(j-1, i-1, width)] > max) max = input[cord(j-1, i-1, width)];
				if (input[cord(j-1, i, width)] > max) max = input[cord(j-1, i, width)];
				if (input[cord(j-1, i+1, width)] > max) max = input[cord(j-1, i+1, width)];
				if (input[cord(j, i-1, width)] > max) max = input[cord(j, i-1, width)];
				if (input[cord(j, i+1, width)] > max) max = input[cord(j, i+1, width)];
				if (input[cord(j+1, i-1, width)] > max) max = input[cord(j+1, i-1, width)];
				if (input[cord(j+1, i, width)] > max) max = input[cord(j+1, i, width)];
				if (input[cord(j+1, i+1, width)] > max) max = input[cord(j+1, i+1, width)];
				input[cord(j, i, width)] = max;
			}
		}
	}
}

void aproximateUnknownPixels (float * input, int width, int height)
{	
	//sjednoceni nedefinovanych pixelu
	int k,l;
	#pragma omp parallel shared (height, width, input)
	#pragma omp for private (k,l)
	for (k = 0; k < height; k++)
		for (l = 0; l < width; l++)
			if (input[cord(l, k, width)] == CONTOURLINE || input[cord(l, k, width)] == WHITECONV)
				input[cord(l, k, width)] = UNDEFINED;

	int unknownCnt;
	for (;;)
	{
		unknownCnt = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (input[cord(j, i, width)] == UNDEFINED)
				{
					if ((j > 0) && (i > 0) && (input[cord(j-1, i-1, width)]!= UNDEFINED))				input[cord(j, i, width)] = input[cord(j-1, i-1, width)];
					else if ((j > 0) && (input[cord(j-1, i  , width)] != UNDEFINED)						)input[cord(j, i, width)]= input[cord(j-1, i  , width)];
					else if ((j > 0) && (i < width-1) && (input[cord(j-1, i+1, width)] != UNDEFINED)	)input[cord(j, i, width)]= input[cord(j-1, i+1, width)];
					else if ((j < height-1) && (i > 0) && (input[cord(j+1, i-1, width)] != UNDEFINED)		)input[cord(j, i, width)]= input[cord(j+1, i-1, width)];
					else if ((j < height-1) && (input[cord(j+1, i  , width)] != UNDEFINED)					)input[cord(j, i, width)]= input[cord(j+1, i  , width)];
					else if ((j < height-1) && (i < width-1) && (input[cord(j+1, i+1, width)] != UNDEFINED)	)input[cord(j, i, width)]= input[cord(j+1, i+1, width)];
					else if ((i > 0) && (input[cord(j  , i-1, width)] != UNDEFINED)							)input[cord(j, i, width)]= input[cord(j  , i-1, width)];
					else if ((i < width-1) && (input[cord(j  , i+1, width)] != UNDEFINED)					)input[cord(j, i, width)]= input[cord(j  , i+1, width)];
					else unknownCnt++;
				}
			}
		}
		if (unknownCnt == 0) break;
	}
}

//hleda doteky a splynuti vrstevnic
t_seed validateResolution (unsigned char * input, int width, int height)
{
	t_seed seed;
	seed.seedCnt = 0;

	long int faultyPixels = 0;
	int neighbours = 0;
	int i, j;
	#pragma omp parallel shared (height, width, input, faultyPixels, seed)
	#pragma omp for private (i,j,neighbours) reduction (+ : faultyPixels)
	for (i = 1; i < height-1; i++)				//indexujeme od 1 a do h-1 protoze krajni body nas nezajimaji
	{												//nemusi splnovat, ze v 8mi okoli se musi nachazet nejmene 2 pixely
		for (j = 1; j < width-1; j++)			
		{
			if (input[cord(j, i, width)] < UNDEFINED)		//cerny pixel
			{	
				neighbours = 0;
				if (input[cord(j-1, i-1, width)]< WHITECONV) neighbours++;
				if (input[cord(j, i-1, width)]	< WHITECONV) neighbours++;
				if (input[cord(j+1, i-1, width)]< WHITECONV) neighbours++;
				if (input[cord(j-1, i, width)]	< WHITECONV) neighbours++;
				if (input[cord(j+1, i, width)]	< WHITECONV) neighbours++;
				if (input[cord(j-1, i+1, width)]< WHITECONV) neighbours++;
				if (input[cord(j, i+1, width)]	< WHITECONV) neighbours++;
				if (input[cord(j+1, i+1, width)]< WHITECONV) neighbours++;
				
				if (neighbours != 2)
				{
					input[cord(j, i, width)] = UNDEFINED;
					faultyPixels ++;
				}
			}
			
			if (input[cord(j, i, width)] > BASECOLOR && input[cord(j, i, width)] < 255)	//seed nejnizsi oblasti
			{
				if (seed.seedCnt < MAXSEEDS)
				{
					seed.seeds[seed.seedCnt] = cord(j, i, width);
				}
				seed.seedCnt++;
				input[cord(j, i, width)] = WHITECONV;
			}
		}
	}
	
	long double imagePixels = width*height;
	long double faultPercentage = 100 * faultyPixels / imagePixels ;
	
	cout.precision(3);
	cout << "Low resolution faults: " << faultPercentage << "%" << endl;
	cout << "Found seeds: " << seed.seedCnt << endl;
	return seed;
}

/*	Vyplni sousedni bile oblasti oblasti specifikovane hodnoty
*/

int fillNextLvl (CvMat * src, float * input, int actualLevel)
{
	int areasCnt = 0;
	for (int i = 3; i < src->rows-3; i++)				//indexujeme od 1 a do h-1 protoze krajni body nas nezajimaji
	{													//nemusi splnovat, ze v 8mi okoli se musi nachazet nejmene 2 pixely
		for (int j = 3; j < src->cols-3; j++)
		{
			if (input[cord(j, i, src->cols)] == actualLevel)
			{
				if (input[cord(j-2, i, src->cols)] == WHITECONV && input[cord(j-3, i, src->cols)] == WHITECONV && input[cord(j-1, i, src->cols)] == CONTOURLINE)
				{
					cvFloodFill (src, cvPoint((j-2), (i)), cvScalarAll (actualLevel+1), cvScalarAll(0), cvScalarAll(0),NULL, 4, NULL);
					areasCnt++;
				}
				else if (input[cord(j, i-2, src->cols)] == WHITECONV && input[cord(j, i-3, src->cols)] == WHITECONV && input[cord(j, i-1, src->cols)] == CONTOURLINE)
				{
					cvFloodFill (src, cvPoint((j), (i-2)), cvScalarAll (actualLevel+1), cvScalarAll(0), cvScalarAll(0), NULL, 4, NULL);
					areasCnt++;
				}
				else if (input[cord(j, i+2, src->cols)] == WHITECONV && input[cord(j, i+3, src->cols)] == WHITECONV && input[cord(j, i+1, src->cols)] == CONTOURLINE)
				{
					cvFloodFill (src, cvPoint((j), (i+2)), cvScalarAll (actualLevel+1), cvScalarAll(0), cvScalarAll(0), NULL, 4, NULL);
					areasCnt++;
				}
				else if (input[cord(j+2, i, src->cols)] == WHITECONV && input[cord(j+3, i, src->cols)] == WHITECONV && input[cord(j+1, i, src->cols)] == CONTOURLINE)
				{
					cvFloodFill (src, cvPoint((j+2), (i)), cvScalarAll (actualLevel+1), cvScalarAll(0), cvScalarAll(0), NULL, 4, NULL);
					areasCnt++;
				}
			}
		}
	}
	return areasCnt;
}

//aproximacni vyhlazeni vyskove mapy - TADY BY SE HODILO TROSK OPTIMALIZOVAT
void smooth (float * src, float * tmp, unsigned char * lines, int width, int height, int iterrations)
{
	double elapsed_time = omp_get_wtime();
	
	cout << "Smoothing terrain ...";

	#pragma omp parallel shared (width, height, lines)
	for(;iterrations >=0; iterrations--)
	{
		// vnitrni cast prvni beh
		int i, j;
		#pragma omp for private (i,j) firstprivate (src)
		for (i = 1; i < height-1; i++)
		{
			for (j = 1; j < width-1; j++)
			{
				if (lines [cord(j, i, width)] != CONTOURLINE)
				{
					tmp [cord(j, i, width)] =(	
												src[cord(j-1, i, width)] +
												src[cord(j+1, i, width)] +
												src[cord(j, i-1, width)] +
												src[cord(j, i+1, width)] +
												
												src[cord(j-1, i-1, width)] +
												src[cord(j+1, i+1, width)] +
												src[cord(j-1, i+1, width)] +
												src[cord(j+1, i-1, width)] ) / 8;
												 //mame 2^3 prvku, protoze mocniny 2 to proste deli nejrychlej
				}
			}
		}
		
		//okrajove casti prvni beh
		//-------------------------------------------------------------------------------

		//horizontalni okraje
		#pragma omp for
		for (int m = 1; m < width -1; m++)
		{		
				if (lines [cord(m, 0, width)] != CONTOURLINE)
					tmp[cord(m, 0, width)] = (src[cord(m, 1, width)] + src[cord(m-1, 1, width)] + src[cord(m+1, 1, width)]) / 3;
				if (lines [cord(m, height-1, width)] != CONTOURLINE)
					tmp[cord(m, height-1, width)] = (src[cord(m, height-2, width)] + src[cord(m-1, height-2, width)] + src[cord(m+1, height-2, width)]) / 3;
		}
		//vertikalni okraje
		#pragma omp for
		for (int n = 1; n < height -1; n++)
		{
				if (lines [cord(0, n, width)] != CONTOURLINE)
					tmp[cord(0, n, width)] = (src[cord(1, n, width)] + src[cord(1, n-1, width)] + src[cord(1, n+1, width)]) / 3;
				if (lines [cord(width-1, n, width)] != CONTOURLINE)
					tmp[cord(width-1, n, width)] = (src[cord(width-2, n, width)] + src[cord(width-2, n-1, width)] + src[cord(width-2, n+1, width)]) / 3;
		}
	
		//rohy
		tmp[0] = src[width];
		tmp[cord(width-1, 0, width)] = src[cord(width-2, 1, width)];
		tmp[cord(0, height-1, width)] = src[cord(1, height-2, width)];
		tmp[cord(width-1, height-1, width)] = src[cord(width-2, height-2, width)];
		//-------------------------------------------------------------------------------

		//vnitrni cast reverzni beh
		#pragma omp for private (i,j) firstprivate (tmp)
		for (i = 1; i < height-1; i++)
		{
			for (j = 1; j < width-1; j++)
			{
				if (lines [cord(j, i, width)] != CONTOURLINE)
				{
					src [cord(j, i, width)] =(	tmp[cord(j-1, i, width)] +
												tmp[cord(j+1, i, width)] +
												tmp[cord(j, i-1, width)] +
												tmp[cord(j, i+1, width)] +
													
												tmp[cord(j-1, i-1, width)] +
												tmp[cord(j+1, i+1, width)] +
												tmp[cord(j-1, i+1, width)] +
												tmp[cord(j+1, i-1, width)] ) / 8;
												
				}
			}
		}
		//okrajove casti reverzni beh
		//-------------------------------------------------------------------------------
		//horizontalni okraje
		#pragma omp for
		for (int m = 1; m < width -1; m++)
		{		
				if (lines [cord(m, 0, width)] != CONTOURLINE)
					src[cord(m, 0, width)] = (tmp[cord(m, 1, width)] + tmp[cord(m-1, 1, width)] + tmp[cord(m+1, 1, width)]) / 3;
				if (lines [cord(m, height-1, width)] != CONTOURLINE)
					src[cord(m, height-1, width)] = (tmp[cord(m, height-2, width)] + tmp[cord(m-1, height-2, width)] + tmp[cord(m+1, height-2, width)]) / 3;
		}
	
		//vertikalni okraje
		#pragma omp for
		for (int n = 1; n < height -1; n++)
		{
				if (lines [cord(0, n, width)] != CONTOURLINE)
					src[cord(0, n, width)] = (tmp[cord(1, n, width)] + tmp[cord(1, n-1, width)] + tmp[cord(1, n+1, width)]) / 3;
				if (lines [cord(width-1, n, width)] != CONTOURLINE)
					src[cord(width-1, n, width)] = (tmp[cord(width-2, n, width)] + tmp[cord(width-2, n-1, width)] + tmp[cord(width-2, n+1, width)]) / 3;
		}
	
		//rohy
		src[0] = tmp[width];
		src[cord(width-1, 0, width)] = tmp[cord(width-2, 1, width)];
		src[cord(0, height-1, width)] = tmp[cord(1, height-2, width)];
		src[cord(width-1, height-1, width)] = tmp[cord(width-2, height-2, width)];

		//-------------------------------------------------------------------------------
		cout << "\rSmoothing terrain : " << iterrations;
	}
	cout << "\rSmoothing execution time: " <<  omp_get_wtime() - elapsed_time << "s" <<endl;
}

void extraSmooth (float * src, float * tmp, int width, int height, int iterrations)
{
	double elapsed_time = omp_get_wtime();
	
	cout << "Extra-smoothing terrain ... ";

	#pragma omp parallel shared (width, height)
	for(;iterrations >=0; iterrations--)
	{
		// vnitrni cast prvni beh

		int i, j;
		#pragma omp for private (i,j) firstprivate (src)
		for (i = 1; i < height-1; i++)
		{
			for (j = 1; j < width-1; j++)
			{
					tmp [cord(j, i, width)] =(	
												src[cord(j-1, i, width)] +
												src[cord(j+1, i, width)] +
												src[cord(j, i-1, width)] +
												src[cord(j, i+1, width)] +
												
												src[cord(j-1, i-1, width)] +
												src[cord(j+1, i+1, width)] +
												src[cord(j-1, i+1, width)] +
												src[cord(j+1, i-1, width)] ) / 8;
												 //mame 2^3 prvku, protoze mocniny 2 to proste deli nejrychlej
			}
		}
		
		//okrajove casti prvni beh
		//-------------------------------------------------------------------------------

		//horizontalni okraje
		#pragma omp for
		for (int m = 1; m < width -1; m++)
		{		
			tmp[cord(m, 0, width)] = (src[cord(m, 1, width)] + src[cord(m-1, 1, width)] + src[cord(m+1, 1, width)]) / 3;
			tmp[cord(m, height-1, width)] = (src[cord(m, height-2, width)] + src[cord(m-1, height-2, width)] + src[cord(m+1, height-2, width)]) / 3;
		}
		//vertikalni okraje
		#pragma omp for
		for (int n = 1; n < height -1; n++)
		{
			tmp[cord(0, n, width)] = (src[cord(1, n, width)] + src[cord(1, n-1, width)] + src[cord(1, n+1, width)]) / 3;
			tmp[cord(width-1, n, width)] = (src[cord(width-2, n, width)] + src[cord(width-2, n-1, width)] + src[cord(width-2, n+1, width)]) / 3;
		}
	
		//rohy
		tmp[0] = src[width];
		tmp[cord(width-1, 0, width)] = src[cord(width-2, 1, width)];
		tmp[cord(0, height-1, width)] = src[cord(1, height-2, width)];
		tmp[cord(width-1, height-1, width)] = src[cord(width-2, height-2, width)];
		//-------------------------------------------------------------------------------

		//vnitrni cast reverzni beh
		#pragma omp for private (i,j) firstprivate (tmp)
		for (i = 1; i < height-1; i++)
		{
			for (j = 1; j < width-1; j++)
			{
					src [cord(j, i, width)] =(	tmp[cord(j-1, i, width)] +
												tmp[cord(j+1, i, width)] +
												tmp[cord(j, i-1, width)] +
												tmp[cord(j, i+1, width)] +
													
												tmp[cord(j-1, i-1, width)] +
												tmp[cord(j+1, i+1, width)] +
												tmp[cord(j-1, i+1, width)] +
												tmp[cord(j+1, i-1, width)] ) / 8;
			}
		}
		//okrajove casti reverzni beh
		//-------------------------------------------------------------------------------
		//horizontalni okraje
		#pragma omp for
		for (int m = 1; m < width -1; m++)
		{		
			src[cord(m, 0, width)] = (tmp[cord(m, 1, width)] + tmp[cord(m-1, 1, width)] + tmp[cord(m+1, 1, width)]) / 3;
			src[cord(m, height-1, width)] = (tmp[cord(m, height-2, width)] + tmp[cord(m-1, height-2, width)] + tmp[cord(m+1, height-2, width)]) / 3;
		}
	
		//vertikalni okraje
		#pragma omp for
		for (int n = 1; n < height -1; n++)
		{
			src[cord(0, n, width)] = (tmp[cord(1, n, width)] + tmp[cord(1, n-1, width)] + tmp[cord(1, n+1, width)]) / 3;
			src[cord(width-1, n, width)] = (tmp[cord(width-2, n, width)] + tmp[cord(width-2, n-1, width)] + tmp[cord(width-2, n+1, width)]) / 3;
		}
	
		//rohy
		src[0] = tmp[width];
		src[cord(width-1, 0, width)] = tmp[cord(width-2, 1, width)];
		src[cord(0, height-1, width)] = tmp[cord(1, height-2, width)];
		src[cord(width-1, height-1, width)] = tmp[cord(width-2, height-2, width)];

		//-------------------------------------------------------------------------------
		cout << "\rExtra-smoothing terrain: " << iterrations;
	}
	cout << "\rExtra-smoothing execution time: " <<  omp_get_wtime() - elapsed_time << "s" <<endl;
}

void exportToOBJ (float * map, int width, int height, int level)
{
	cout << "Exporting to object file ..." << endl;
	
	ofstream ofile;
	ofile.open ("terrain.obj");

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
				ofile << "v " << j << " " << ((map[cord(j, i, width)] - BASECOLOR) * level) << " " << i << endl;
		}
	}	

	for (long int i = 1; i < (width-1) * (height-1); i++)
	{
		if (i%width!= 0)	
		{
			ofile << "f " << i << " " << (i+width) << " " << (i+1) << endl << "f " << (i+1) << " " << (i+width) << " " << (i+1+width) << endl;
		}
	}
	cout << endl;
	ofile.close();
}