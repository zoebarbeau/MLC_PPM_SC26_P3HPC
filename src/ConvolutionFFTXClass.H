#ifndef _FFTX_CONVOLUTION_H_
#define _FFTX_CONVOLUTION_H_
#include <ExaMPM_MLC_Interp.hpp>
#include <ExaMPM_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <ExaMPM_GreensFunction.hpp>
#include <ExaMPM_GridManager.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <Kokkos_Complex.hpp>
#include "fftx3.hpp"
#include "interface.hpp"
#include "mdprdftObj.hpp"
#include "imdprdftObj.hpp"
#include "rconvObj.hpp"
# include <vector>
#include <complex>
#include <algorithm>
//#include "Proto.H"

//using namespace Proto;
class FFTXConvolution
{
  public:
  FFTXConvolution(std::string a_lgffile,double a_h,int a_N);
  double extent;
  double domaindouble_x;  
  double domaindouble_y;
  double domaindouble_z;
  void convolveWithG(std::vector<double>& a_dataIn,
                     std::vector<double>& a_dataOut);

private:
  std::vector<std::complex<double> > m_Ghat;
  int m_N;
  double m_h;
};
#include "FFTXConvolutionImplem.H"
#endif
