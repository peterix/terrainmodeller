#include <stdlib.h>
#include <opencv2/opencv.hpp>


#define BASECOLOR 3		//hodnota nejnizsi urovne
#define WHITECONV 2		//hodnota, na kterou je konvertovana bila barva
#define CONTOURLINE 0	//hodnota vrstevnice
#define UNDEFINED 1		//hodnota nedefinovaneho pixelu

#define MAXSEEDS 10		//maximalni pocet vstupnich seedu


/*struktura pro vstupni seed pointy
  seeds - pole souradic vstupnich bidu
  seedCnt - pocet vstupnich bodu
*/
typedef struct seed
{
	int seedCnt;
	long int seeds[MAXSEEDS];
} t_seed;

/*	Prevede 2D souradnice na 1D index pole
	**Vstup: souradnice x, souradnice y, sirka, vyska (obrazku)
	**Vystup: 1D souradnice
*/
long int cord (int x, int y, int width);


/*	Funkce prohleda vstupni data a oznace chybne body, ktere vznikly,
	nepresnosti rasterizace.
	Zadne 2 vrstevnice se nsmi sbihat, ani dotykat jedna druhe!
	**Vstup: pole pixelu vstupniho obrazku, sirka obrazku, vyska obrazku
	**Vystup: 1-pokud jsou data v dostecnem tozliseni. 0-pokud je nutne
			  data dodat ve vyssim rozliseni
*/
t_seed validateResolution (unsigned char * input, int width, int height);


/*	Prohledava data input, a pokud nalezne pixel s hodnotou actualLevel,
	ktery je na pokraji oblasti, obarvi oblast na druhe strane vrstevnice
	hodnotou actualLevel+1
	Vstupuje CvMat i float * input, protoze jako vstup cvFloodFill musi byt
	prave cvMat
*/
int fillNextLvl (CvMat * src, float * input, int actualLevel);

/*
	"Obarvi" vrstevnice na barvu oblasti, kterou ohranicuji
	(te vyssi, ktere se dotykaji)
*/
void convertContourLines (float * input, int width, int height);

/*	Nalezne zbytky vrstevnic v datech a chybnych bodu vzniklych rasterizaci
	a prideli jim validni hodnotu z nejblizsiho okoli
*/
void aproximateUnknownPixels (float * input, int width, int height);

/*	Prevede body nesouci bilou podkladovou barvu na hodnotu WHITECONV
	-vsechny hodnoty nesouci interni informace musi byt v souvisle oblasti0-3
*/
void convertWhite (float * input, int width, int height);

/*
	Aproximuje hodnotu bodu dle jeho nejblizsiho okoli
*/
void smooth (float * src, float * tmp, unsigned char * lines, int width, int height, int iterrations);

/*
	kady ctverec metr x metr prevede na dva trojuhelnikove polygony
	a ulozi dle obj specifikace do souboru
*/
void exportToOBJ (float * map, int width, int height, int level);

/*
	zaverecne vyhlazeni - dohladi stopy vrstevnic - rychlost za cenu presnosti
*/
void extraSmooth (float * src, float * tmp, int width, int height, int iterrations);