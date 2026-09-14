/* This file is written by the authors specifically for MLC */
//
// c2c (complex-to-complex) variant of MLC_ConvolutionGPU.hpp, which runs
// successfully on H100. Kept on the SAME fftx3.hpp/interface.hpp API and the
// SAME args-construction idiom (static locals, address-of in the args
// vector) as the working file, since that idiom is what this older API
// actually requires (it takes void**, not void*). Only the transform kind
// changed from mdprdft/imdprdft (real<->complex) to mddft/imddft
// (complex<->complex), with the AMD MI300 file's complex Views/copies
// merged in to feed it.
//
// ASSUMPTION TO VERIFY: this assumes the older API exposes mddft/imddft via
// headers named "mddftObj.hpp"/"imddftObj.hpp" (mirroring
// mdprdftObj.hpp/imdprdftObj.hpp) with classes MDDFTProblem/IMDDFTProblem.
// If your FFTX tree names these differently (or doesn't have a c2c problem
// under the old API at all), swap the include/class names below accordingly
// -- that in itself would be useful diagnostic info.
//
#ifndef MLC_CONVOLUTIONGPU_C2C_HPP
#define MLC_CONVOLUTIONGPU_C2C_HPP

#include <MLC_MLC_Interp.hpp>
#include <MLC_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <MLC_GreensFunction.hpp>
#include <MLC_GridManager.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <Kokkos_Complex.hpp>
#include "fftx.hpp"
#include "fftxinterface.hpp"
#include "fftxmdprdftObj.hpp"
// #include "fftximdprdftObj.hpp"
#include "fftxmddftObj.hpp"
#include "fftximddftObj.hpp"
#include "fftximdprdftObj.hpp"

namespace MLC
{
namespace ConvolutionGPU
{
//---------------------------------------------------------------------------//
// Particle-to-grid.
template <class ExecutionSpace, class ProblemManagerType>
void Conv_fftx_c2c(const ExecutionSpace& exec_space, const ProblemManagerType& pm, const int extent,
                    const double center, const double h)
{
  using Complex = Kokkos::complex<double>;

  auto F = pm.get( Location::Node(),Field::F() );
  auto velx = pm.get( Location::Node(),Field::velx() );
  Kokkos::deep_copy( velx, 0.0);
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> F1D("Fvector", extent*extent*extent);

  // Dims for double domain
  int domaindouble_x = 2*extent;
  int domaindouble_y = 2*extent;
  int domaindouble_z = 2*extent;

  // GPU domain double F1D (real) and its complex copy for the c2c transform
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> F1D_domaindouble("Fdomaindouble", domaindouble_x*domaindouble_y*domaindouble_z);
  Kokkos::View<Complex*, Kokkos::DefaultExecutionSpace::memory_space> F1D_dd_cmplx("Fdomaindouble_cmplx", domaindouble_x*domaindouble_y*domaindouble_z);

  Kokkos::Timer timer;
  // Lattice Green's function
  std::ifstream infileLGF("/home/zbarbeau/Hudson_P3M_H100/MLC_PPM/LatticeGreensFunction/exec/G_128_Octant");
  std::vector<double> lgf_values(domaindouble_x * domaindouble_y * domaindouble_z);
  if(infileLGF.is_open()){
    std::string line;
    int xdir;
    int ydir;
    int zdir;
    double lgf;

    while(getline(infileLGF, line)){

      for(char& c : line){
        if (c == '(' || c == ')'){
          c = ' ';
        }
      }

       // Remove extra spaces around commas
      size_t pos = line.find(", ");
      while (pos != std::string::npos) {
        line.erase(pos, 1);  // Erase space after comma
        pos = line.find(", ", pos);
      }

      // Remove leading and trailing spaces
      size_t first = line.find_first_not_of(" \t");
      size_t last = line.find_last_not_of(" \t");

      if (first != std::string::npos && last != std::string::npos) {
        line = line.substr(first, last - first + 1);
      }
      std::stringstream ss(line);

      ss >> xdir;
      ss.ignore(1, ',');
      ss >> ydir;
      ss.ignore(1, ',');
      ss >> zdir;
      ss.ignore(1, ',');
      ss >> lgf;

      if(xdir >= 0 && xdir <= extent && ydir >= 0 && ydir <= extent && zdir >= 0 && zdir <= extent)
        {
          for (int isign = -1; isign < 2; isign+=2)
            {
              for (int jsign = -1; jsign < 2; jsign+=2)
                {
                  for (int ksign = -1; ksign < 2; ksign+=2)
                    {
                      int zdirsigned = zdir*ksign;
                      int ydirsigned = ydir*jsign;
                      int xdirsigned = xdir*isign;
                      if ((xdirsigned != extent) &&
                          (ydirsigned != extent) &&
                          (zdirsigned != extent))
                        {
                          int index = zdirsigned + extent +
                            2*extent*(ydirsigned + extent) +
                            4*extent*extent*(xdirsigned + extent);

                          lgf_values[index] = lgf/h;
                        }
                    }
                }
            }
        }

    }
    infileLGF.close();
  }
  else{
    std::cout << "Unable to open file" << std::endl;
  }

  // Creating a host view for the LGF values
  Kokkos::View<double*, Kokkos::HostSpace> host_LGF("h_view", domaindouble_x * domaindouble_y * domaindouble_z);
  // Copying/storing lgf_values into the host view
  for(int i = 0; i < (domaindouble_x * domaindouble_y * domaindouble_z); i++){
    host_LGF[i] = lgf_values.data()[i];
  }
  // Creating a device view to deep copy host_LGF values
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> dev_LGF("d_view", domaindouble_x * domaindouble_y * domaindouble_z);
  // deepcopy host to device
  Kokkos::deep_copy(dev_LGF, host_LGF);

  Kokkos::View<Complex*, Kokkos::DefaultExecutionSpace::memory_space> dev_LGF_cmplx("d_view_cmplx", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::deep_copy(dev_LGF_cmplx, Complex(0.0, 0.0));
  Kokkos::parallel_for("Copy dev_LGF to dev_LGF_cmplx", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, domaindouble_x * domaindouble_y * domaindouble_z),
      KOKKOS_LAMBDA(const int i) {
          dev_LGF_cmplx[i] = Complex(dev_LGF[i], 0.0);
      });


  Kokkos::View<Complex*, Kokkos::DefaultExecutionSpace::memory_space> symbol("symbol_view", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> dummy1("dummy1_view", domaindouble_x * domaindouble_y * domaindouble_z);

/*  std::vector<void*> args1 = [&]() {
      static auto symbol_data = symbol.data();
      static auto lgf_data = dev_LGF_cmplx.data();
      static auto dummy1_data = dummy1.data();
      return std::vector<void*>{&symbol_data, &lgf_data, &dummy1_data};
  }();

*/

  void* symbol_data = symbol.data();
  void* lgf_data = dev_LGF_cmplx.data();
  void* dummy1_data = dummy1.data();
  std::vector<void*> args1{&symbol_data, &lgf_data, &dummy1_data};
  // Calculate the symbol i.e. forward DFT of the lattice Green's function
  std::vector<int> sizes{domaindouble_x, domaindouble_y, domaindouble_z};
  MDDFTProblem c2cdft1{args1, sizes, "mddft"};
  c2cdft1.transform();

  //**************** Using Individual FFTX functions for Convolution *******//
  timer.reset();
  Kokkos::View<Complex*,Kokkos::DefaultExecutionSpace::memory_space> F_dft("fwd_dft_view", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> dummy2("dummy2_view", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::View<Complex*,Kokkos::DefaultExecutionSpace::memory_space> pointwise_mul("ptwise_view", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::View<Complex*, Kokkos::DefaultExecutionSpace::memory_space> out_idft("inv_dft_view", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> out_idft_real("inv_dft_real_view", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> dummy3("dummy3_view", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> out_normalize("norm_output_view", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> conv_output("Final exatracted output", extent * extent * extent);

  for(int d = 0; d < 3; d++){
    timer.reset();
    Kokkos::parallel_for("Copy 4D to 1D", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(exec_space,{0, 0, 0}, {extent, extent, extent}),
          KOKKOS_LAMBDA(const int i, const int j, const int k) {
              int index = i * extent * extent + j * extent + k;
              F1D(index) = F(i, j, k, d);
          });

    Kokkos::parallel_for("Place F1D in doubledomain", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(exec_space,{0, 0, 0}, {extent, extent, extent}),
             KOKKOS_LAMBDA(const int k, const int j, const int i) {
              int index_dd = (k + extent)*4*extent*extent + (j + extent)*2*extent + i + extent;
              int index_1d = k * extent * extent + j * extent + i;
              F1D_domaindouble[index_dd] = F1D[index_1d];
          });

    // Copying F1D_domaindouble (real) to F1D_dd_cmplx (complex input for mddft)
    Kokkos::deep_copy(F1D_dd_cmplx, Complex(0.0, 0.0));
    Kokkos::parallel_for("Copy F1D_domaindouble to F1D_dd_cmplx", Kokkos::RangePolicy<ExecutionSpace>(exec_space,0,domaindouble_x * domaindouble_y * domaindouble_z),
          KOKKOS_LAMBDA(const int i) {
              F1D_dd_cmplx[i] = Complex(F1D_domaindouble[i], 0.0);
          });

/*    std::vector<void*> args2 = [&]() {
       static auto Fdft_data = F_dft.data();
        static auto F1D_data = F1D_dd_cmplx.data();
        static auto dummy2_data = dummy2.data();
        return std::vector<void*>{&Fdft_data, &F1D_data, &dummy2_data};
    }(); */
    void* Fdft_data = F_dft.data();
    void* F1D_data = F1D_dd_cmplx.data();
    void* dummy2_data = dummy2.data();
    std::vector<void*> args2{&Fdft_data, &F1D_data, &dummy2_data};
    // Calculate the forward DFT of the second input F1D_dd_cmplx
    std::vector<int> sizes2{domaindouble_x, domaindouble_y, domaindouble_z};
    MDDFTProblem c2cdft2{args2, sizes2, "mddft"};
    c2cdft2.transform();

    // Pointwise Multiply (full domain-doubled range, not half+1)
    Kokkos::parallel_for("Pointwise_multiply", Kokkos::RangePolicy<ExecutionSpace>(exec_space,0,domaindouble_x * domaindouble_y * domaindouble_z),
            KOKKOS_LAMBDA(const int i) {
                double a = symbol(i).real();
                double b = symbol(i).imag();
                double c = F_dft(i).real();
                double d = F_dft(i).imag();

                //Pointwise multiply symbol and Fpc_dft
                double real_pw = a * c - b * d;
                double img_pw = a * d + b * c;

                pointwise_mul[i] = Complex(real_pw, img_pw);
            });

    // Calculate the inverse dft to compute the final convolution value
/*  std::vector<void*> args3 = [&]() {
        static auto out_data = out_idft.data();
        static auto pwise_data = pointwise_mul.data();
        static auto dummy3_data = dummy3.data();
        return std::vector<void*>{&out_data, &pwise_data, &dummy3_data};
    }(); */

    void* out_data = out_idft.data();
    void* pwise_data = pointwise_mul.data();
    void* dummy3_data = dummy3.data();
    std::vector<void*> args3{&out_data, &pwise_data, &dummy3_data};
    std::vector<int> sizes3{domaindouble_x, domaindouble_y, domaindouble_z};
    IMDDFTProblem c2cidft{args3, sizes3, "imddft"};
    c2cidft.transform();

    // Extracting the real part of the (complex) inverse DFT output
    Kokkos::parallel_for("Extract real part of inverse DFT", Kokkos::RangePolicy<ExecutionSpace>(exec_space,0,domaindouble_x * domaindouble_y * domaindouble_z),
            KOKKOS_LAMBDA(const int i) {
                out_idft_real[i] = out_idft[i].real();
            });

    // Normalizing the output with h^3/dd^3
    double norm_factor = pow(h,3)/(domaindouble_x * domaindouble_y * domaindouble_z);
    Kokkos::parallel_for("Normalize output", Kokkos::RangePolicy<ExecutionSpace>(exec_space,0,domaindouble_x * domaindouble_y * domaindouble_z),
            KOKKOS_LAMBDA(const int i) {
                out_normalize[i] = norm_factor * out_idft_real[i];
            });

    Kokkos::parallel_for("Normalize output", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(exec_space, {0, 0, 0}, {extent, extent, extent}),
            KOKKOS_LAMBDA(const int k, const int j, const int i) {
                int out_dd_index = k * domaindouble_y * domaindouble_x + j * domaindouble_x + i;
                int out_original_index = k * extent * extent + j * extent + i;
                conv_output[out_original_index] = out_normalize[out_dd_index];
            });
    auto velocity_g = pm.get(Location::Node(), Field::Velocity());

    int N = extent;
    double U = 1.0;

    Kokkos::fence();
    double timerFFTX = timer.seconds();
    std::cout << timerFFTX << " FFTX " << std::endl;

    Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N,N,N}),
          KOKKOS_LAMBDA(const int i, const int j, const int k) {
            int index_f = i * N * N + j * N + k;
            double x[3] = {i*h-0.5,j*h-0.5,k*h-0.5};
            double v_exact[3],v_error[3];
            double r =  sqrt(x[1]*x[1] + x[2]*x[2]+x[0]*x[0]);
            double R = 0.25;
            double R2 = 0.25*0.25;
            double U = 1;
            double R3 = 0.25*0.25*0.25;
            double xz = x[0]*x[2];
            double yz = x[1]*x[2];

            if(d == 2 ){
              velocity_g(i,j,k,d) = U + conv_output[index_f] ;
              velx(i,j,k,0)  = U + (conv_output[index_f]);
            }else if(d == 1){
              velocity_g(i,j,k,d) = conv_output[index_f];
              velx(i,j,k,0)  = conv_output[index_f];
            }else if(d == 0){
              velocity_g(i,j,k,d) = conv_output[index_f];
              velx(i,j,k,0) =  conv_output[index_f];
            }
     });

  }
}
}
}
#endif
