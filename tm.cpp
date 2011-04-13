/********************************************
 * ZPO 2011 - Projekt 2 - Terrain modeller 	*
 *										   	*
 * authors: Miroslav Dvoøák (xdvora11)		*
 *			Petr Mrázek		(xmraze03)		*
 ********************************************/

//standard
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cstdlib>

//openCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

//openMP
#include <omp.h>

//user defined
#include "functions.h"

using namespace std;

//vypis napovedy
void printHelp ()
{
	cout << endl <<
			"Terrain modeller 2011 by Miroslav Dvorak & Petr Mrazek" << endl <<
			"===============================================================================" << endl << endl <<
			" Mandatory parameters:" << endl <<
			" ---------------------" << endl <<
			"-input [file_name] specifies the input raster file(bmp), f.e. plotered from CAD" << endl <<
			"-scale [int]	   specifies how many pixels reprent 1m in real" << endl <<
			"-lvl   [int]	   specifies elevation between 2 contour lines in m" << endl << endl <<
			" Optional parameters:" << endl <<
			" --------------------" << endl <<
			"-iter  [int]  specifies no of iterrations for smoothing - default 1" << endl <<
			"-extra [int]  specifies no of iterrations for extra smoothing -default 0" << endl <<
			"-t     [int]  pocet vlaken pracujicich na vypoctu -default 2 -best no.of cores" << endl <<
			"-help         prints this help" << endl <<
			"===============================================================================" << endl << endl;
}

//vypis chybovych hlasek dle kodu
void errorHandler (int errCode)
{
	switch (errCode)
	{
		case 0:	cout << "Nenalezena znacka nejnizsi oblasti!";
				exit (1);
		case 1: cout << "Prilis mnoho bodu odpovida znacce pro nejnizsi oblast!";
				exit (1);
		default : cout << "Random Error, HF :-)";
				exit (1);
	}
}

int main(int argc, char* argv[])
{
	int scale, lvl, threads = 2, smoothIter = 0, extraIter = 0;
	double elapsed_time;
	char * srcName = NULL, * outName = "out.bmp";
	CvMat * src = NULL, * lines = NULL, * tmp = NULL;

	CV_FUNCNAME("Terrain modeller");
	__CV_BEGIN__


    // zpracovani specifickych argumentu
    for (int i = 1; i < argc; i++) {
		if      (strcmp(argv[i],"-input")   == 0 && i < argc+1 )  { srcName  = argv[++i]; }			//jmeno vstupniho souboru
		else if (strcmp(argv[i],"-scale")   == 0 && i < argc+1 )  { scale = atoi(argv[++i]); }		//meritko (kolik pixelu predstavuje 1m)
		else if (strcmp(argv[i],"-lvl") == 0 && i < argc+1)  { lvl = atoi(argv[++i]); }				//vyskovy rozdil mezi vrstevnicemi v metrech
		else if (strcmp(argv[i],"-iter") == 0 && i < argc+1)  { smoothIter = atoi(argv[++i]); }		//pocet iteraci pro vyhlazovani - defaultne 0
		else if (strcmp(argv[i],"-extra") == 0 && i < argc+1)  { extraIter = atoi(argv[++i]); }		//pocet iteraci TURBO:) dohlazovani
		else if (strcmp(argv[i],"-t") == 0 && i < argc+1)  { threads = atoi(argv[++i]); }			//pocet vlaken - defaultne 2
		else if (strcmp(argv[i],"-help")  == 0 && argc == 2 )  { printHelp(); return 0; }			//vypis napovedy
    }
	
	//nezadan nektery z parametru jmeno vstupiho souboru, kolik pix je 1m ve skutecnosti ,vyska vrstevnice -> vypis napovedy
	if( !srcName || !scale || !lvl ) { printHelp(); return 1;}
	
	//nastaveni poctu vlaken
	omp_set_num_threads(threads);
	
	//iterativni "konstanty" musi byt delitelne poctem vlaken
	smoothIter += (threads - (smoothIter % threads));
	extraIter += (threads - (extraIter % threads));
	
	cout << endl << 
			"-------------------------------------------------" << endl <<
			"Terrain modeller by Miroslav Dvorak & Petr Mrazek" << endl <<
			"-------------------------------------------------" << endl << endl;


	//start mereni casu behu aplikace
	elapsed_time = omp_get_wtime();

	//nacteni vstupnich dat
	src = cvLoadImageM(srcName, 0);
	
	//ve zpracovavanych datech nalezneme pocatecni seedy v nejnizsich oblastech a spocitam miru rastrove chyby
	t_seed seed = validateResolution (src->data.ptr, src->cols, src->rows);
	
	if (seed.seedCnt == 0) errorHandler (0);				//nenalezena znacka nejnizsi oblasi
	else if (seed.seedCnt > MAXSEEDS) errorHandler (1);		//nalezeno prilis mnoho znacek nejnizsi oblasti
	
	lines = cvCloneMat (src);//cvCreateMat((src->rows/scale), (src->cols/scale), CV_8UC1 ); //mapa vrstevnic pro potreby aproximace
	//cvResize (src, lines, CV_INTER_AREA);
	//cvSaveImage( outName, lines, 0 );

	//prevede vstupni data unsigned char na 32bit float
	CvMat * map = cvCreateMat( src->rows, src->cols, CV_32FC1 );
	cvConvertScale( src, map, 1, 0 );
	
	//prevede bile plochy na hodnotu WHITECONV - nutne, pokud by mela mapa vice naz 255 urovni
	convertWhite (map->data.fl, map->cols, map->rows);

	//zafillovani nejnizsich oblasti
	int fillColor = BASECOLOR;
	for (int i = 0; i < seed.seedCnt; i++)
		cvFloodFill (map, cvPoint((seed.seeds[i]%(map->cols)-1), seed.seeds[i]/( map->cols)), cvScalarAll(fillColor), cvScalarAll(0), cvScalarAll(0), NULL, 4, NULL);
	
	//najdeme vsechny urovne a rasterove je "obarvime"
	int areasCnt = 0, areas;
	cout << "levels: 0\tareas: 0";
	for (int i = fillColor;; i++)//dokud nachazi nove urovne
	{
		areas = fillNextLvl (map, map->data.fl, i);
		if (areas == 0)
			break;
		else
		{
			areasCnt += areas;
			cout << "\rlevels: " << i << "\tareas: " << areasCnt;
		}
	}
	cout << endl;
	//odstranime vrstevnice u urovni - dame jim hodnotu vyssi z urovni (vstevnice oznacuje pocatek vyssi urovne)
	convertContourLines (map->data.fl, map->cols, map->rows);
	
	//sjednotime nezname body a dame jim hodnotu nejblizsiho znameho okoli - pozustatky vrstevnic, uzly vrstevnic,
	//ktere nemaji dostatecne rozliseni, vyskove plochy nezpracovane kvuli nizkemu rozliseni
	aproximateUnknownPixels (map->data.fl, map->cols, map->rows);
	
	//pro vyuzití paralelizace je nutné pracovat s pomocným polem
	tmp = cvCloneMat(map);		//mapa vrstevnic
	smooth (map->data.fl, tmp->data.fl, lines->data.ptr, map->cols, map->rows, smoothIter);

	//obrazkovy vystup do output.bmp
	//cvConvertScale( map, src, 1., 0 );
	//cvSaveImage( outName, src, 0 );
	
	//vystup do obj souboru - v pomeru zadanem pomoci parametru scale
	CvMat * output = cvCreateMat((src->rows/scale), (src->cols/scale), CV_32FC1 );
	cvResize (map, output, CV_INTER_AREA);
	
	if (extraIter > 0)
	{	
		tmp = cvCloneMat(output);
		extraSmooth (output->data.fl, tmp->data.fl, output->cols, output->rows, extraIter);
	}
	//zapis dat do vystupniho objektoveho souboru
	exportToOBJ (output->data.fl, output->cols, output->rows, lvl);
	__CV_END__
	
	//uvolneni zbytku matic
	if(src) cvReleaseMat(&src);
	if(tmp) cvReleaseMat (&tmp);
	if(lines) cvReleaseMat (&lines);
	
	//vypis doby behu aplikace   
	cout << "Overall execution time: " <<  omp_get_wtime() - elapsed_time << "s"<< endl;
	
	return 0;
}