#ifndef _FFTW_LGFCONVOLUTION_H_
#define _FFTW_LGFCONVOLUTION_H_
# include <iostream>
# include <vector>
# include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <complex>
#include <algorithm>
#include <iomanip>
//#include "Proto.H"
#include "fftw3.h"
//using namespace Proto;
class LGFConvolution
{
  public:
  LGFConvolution(std::string a_lgffile,double a_h,int a_N);
  void convolveWithG(std::vector<double>& a_dataIn,
                     std::vector<double>& a_dataOut);

private:
  std::vector<std::complex<double> > m_Ghat;
  int m_N;
  double m_h;
};
#include "FFTWLGFConvolutionImplem.H"
#endif
