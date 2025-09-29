#ifndef _FFTX_LGFCONVOLUTION_H_
#define _FFTX_LGFCONVOLUTION_H_
#include <Kokkos_Core.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <Kokkos_Complex.hpp>
#include <ExaMPM_MLC_Interp.hpp>
#include <ExaMPM_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <ExaMPM_GreensFunction.hpp>
#include <ExaMPM_GridManager.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>
#include "fftx.hpp"
#include "fftxinterface.hpp"
#include "fftxmdprdftObj.hpp"
#include "fftximdprdftObj.hpp"
#include "fftxrconvObj.hpp"

template <typename ExecutionSpace>
class ConvolutionFFTX {
public:
  using memory_space = typename ExecutionSpace::memory_space;
  using Complex = Kokkos::complex<double>;

private:
  int extent;
  double center;
  double h;

  int domaindouble_x;
  int domaindouble_y;
  int domaindouble_z;

  Kokkos::View<double*,  memory_space> F1D;
  Kokkos::View<double*,  memory_space> F1D_domaindouble;
  Kokkos::View<Complex*, memory_space> symbol;
  Kokkos::View<double*,  memory_space> dummy1;
  Kokkos::View<double*,  memory_space> dev_LGF;
  Kokkos::View<Complex*, memory_space> F_dft;
  Kokkos::View<double*,  memory_space> dummy2;
  Kokkos::View<Complex*, memory_space> pointwise_mul;
  Kokkos::View<double*,  memory_space> out_idft;
  Kokkos::View<double*,  memory_space> dummy3;
  Kokkos::View<double*,  memory_space> out_normalize;
  Kokkos::View<double*,  memory_space> conv_output;

  std::vector<double> lgf_values;
//  std::unique_ptr<MDPRDFTProblem>;

public:
  ConvolutionFFTX(int extent_, double center_, double h_)
      : extent(extent_), center(center_), h(h_),
        domaindouble_x(2 * extent_), domaindouble_y(2 * extent_), domaindouble_z(2 * extent_),
        F1D("Fvector", extent * extent * extent),
        F1D_domaindouble("Fdomaindouble", domaindouble_x * domaindouble_y * domaindouble_z),
        symbol("symbol_view", domaindouble_x * domaindouble_y * ((domaindouble_z / 2) + 1)),
        dev_LGF("d_view", domaindouble_x * domaindouble_y * domaindouble_z),
        dummy1("dummy1_view", domaindouble_x * domaindouble_y * domaindouble_z),
        F_dft("fwd_dft_view", domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)),
        dummy2("dummy2_view", domaindouble_x * domaindouble_y * domaindouble_z),
        pointwise_mul("fwd_dft_view", domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)),
        out_idft("inv_dft_view", domaindouble_x * domaindouble_y * domaindouble_z),
        dummy3("dummy3_view", domaindouble_x * domaindouble_y * domaindouble_z),
        out_normalize("norm_output_view", domaindouble_x * domaindouble_y * domaindouble_z),
        conv_output("Final exatracted output", extent * extent * extent),
        lgf_values(domaindouble_x * domaindouble_y * domaindouble_z, 0.0)
  {}

  void read_LGF_file(const std::string& filename);
  template <class ProblemManagerType>
  void compute_convolution(const ExecutionSpace& exec_space, ProblemManagerType& pm);
//  void write_view_to_txt(const Kokkos::View<double*, Kokkos::HostSpace>& view,
//                       const std::string& filename);
};

#include "FFTXLGFConvolutionImplem.H"
#endif
