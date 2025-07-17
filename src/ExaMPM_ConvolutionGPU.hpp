/****************************************************************************
 * Copyright (c) 2018-2020 by the ExaMPM authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ExaMPM library. ExaMPM is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef EXAMPM_CONVOLUTIONGPU_HPP
#define EXAMPM_CONVOLUTIONGPU_HPP

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
// #include "fftw3.h"
namespace ExaMPM
{
namespace ConvolutionGPU
{
//---------------------------------------------------------------------------//
// Particle-to-grid.
//


template <class ExecutionSpace, class ProblemManagerType>
void Test_F(const ExecutionSpace& exec_space, const ProblemManagerType& pm, const double extent, 
	    const double center, const double cellsize)
{
       auto F = pm.get( Location::Node(),Field::F() );

       for(int i = 0; i < extent; i++)
        for(int j = 0; j < extent; j++)
          for( int k = 0; k < extent; k++)
          {
         
            //   std::cout << " Fx = " << F(i,j,k,0) << " Fy = " <<
		        //  F(i,j,k,1) << " Fz = " << F(i,j,k,2) << std::endl;
             std::cout <<"i = " << i << " Fx = " << F(i,j,k,0) << " Fy = " <<
		         F(i,j,k,1) << " Fz = " << F(i,j,k,2) << std::endl;



          }		 

      pm.save_F( "Test_F",1,0); 

}
 
template <class ExecutionSpace, class ProblemManagerType>
void Conv_fftx(const ExecutionSpace& exec_space, const ProblemManagerType& pm, const int extent, const double center, const double h)
{
  printf("INSIDE CONVOLUTION::CONV_FFTX!!!!\n");
  auto F = pm.get( Location::Node(),Field::F() );
  auto velx = pm.get( Location::Node(),Field::velx() ); 
  Kokkos::deep_copy( velx, 0.0);
  // Kokkos::View<double*> F1D("Fvector", extent*extent*extent);
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> F1D("Fvector", extent*extent*extent);
  
  // Copy data from 4D to 1D using a parallel loop
  //  Kokkos::parallel_for("Copy 4D to 1D", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {extent, extent, extent, 1}), 
  //       KOKKOS_LAMBDA(const int i, const int j, const int k, const int m) {
  //           int index = i * extent * extent + j * extent + k;
  //           F1D(index) = F(i, j, k, 0);
  //       });

printf("access to f1d before");
  double *F1D_vec = F1D.data();
printf("access F1D after parallel");
// Dims for double domain
  int domaindouble_x = 2*extent;
  int domaindouble_y = 2*extent;
  int domaindouble_z = 2*extent;

// // Domain doubling the Velocity vector F1D so that it can be used as input for FFTX convolution function
// // The data cube is placed in the upper right corner of the doubled domain cube
//  double* F1D_domaindouble = new double[domaindouble_x*domaindouble_y*domaindouble_z];

//  for(int i = 0; i < domaindouble_x; i++){
//    for(int j = 0; j < domaindouble_y; j++){
//      for(int k = 0; k < domaindouble_z; k++){

//         int index_dd = i*domaindouble_y*domaindouble_z + j*domaindouble_z + k;

//         if((i < extent) && (j >= extent && j < domaindouble_y) && (k < extent)){
//           int index_lgf = i * extent * extent + ((j-extent) * extent) + k;
//           F1D_domaindouble[index_dd] = F1D_vec[index_lgf];     
//         }
//         else{
//           F1D_domaindouble[index_dd] = 0.0;     
//         }
//       //  std::cout << "index_dd=" << index_dd << "\t" << "Fpc_1D_dd = " <<F1D_domaindouble[index_dd] << std::endl;
//      }
//    }
//   }

// for (int i = 0; i < extent ; i++)
//     {
//       for (int j = 0; j < extent; j++)
//         {
//           for (int k = 0;  k < extent; k++)
//             {
//               int index_dd = (i + extent)*4*extent*extent + (j + extent)*2*extent + k + extent;
//               int index_1d = i * extent * extent + j * extent + k;
//               F1D_domaindouble[index_dd] = F1D_vec[index_1d];              
//             }
//         }
//     }

// GPU domain double F1D
Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> F1D_domaindouble("Fdomaindouble", domaindouble_x*domaindouble_y*domaindouble_z);

Kokkos::Timer timer;
  // Lattice Green's function
  std::ifstream infileLGF("/g/g16/barbeau2/CPU/MLC_PPM/LatticeGreensFunction/exec/G_128_Octant");
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
      // std::cout << line << std::endl;
      std::stringstream ss(line);
      
      ss >> xdir;
      ss.ignore(1, ',');
      ss >> ydir;
      ss.ignore(1, ',');
      ss >> zdir;
      ss.ignore(1, ',');
      ss >> lgf;

      // if((xdir >= (-extent) && xdir < (extent)) && (ydir >= (-extent) && ydir < (extent)) && (zdir >= (-extent) && zdir < (extent))){
      //     int index_lgf = zdir + extent + 2*extent*(ydir + extent) + 4*extent*extent*(xdir+extent);
      //     lgf_values[index_lgf] = lgf;
      //     // lgf_values.push_back(lgf);
      // }
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
    // for(double lgfv : lgf_values){
    //   std::cout << std::setprecision(10) << lgfv << std::endl;
    // }
    infileLGF.close();
  }
  else{
    std::cout << "Unable to open file" << std::endl;
  }
  std::cout << "Size of LGF Vector=" << lgf_values.size()<<std::endl;
  // for(int i = 0; i < lgf_values.size(); i++){
  //   lgf_values.data()[i] = 1.0;

  // }

  // Creating a host view for the LGF values
  Kokkos::View<double*, Kokkos::HostSpace> host_LGF("h_view", domaindouble_x * domaindouble_y * domaindouble_z);
  // Copying/storing lgf_values into the host view
  for(int i = 0; i < (domaindouble_x * domaindouble_y * domaindouble_z); i++){
    host_LGF[i] = lgf_values.data()[i];
    // std::cout<<"i = "<< i<<",\t host_LGF = " << host_LGF[i] <<",\t" << lgf_values.data()[i] << std::endl;
  }
  // Creating a device view to deep copy host_LGF values 
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> dev_LGF("d_view", domaindouble_x * domaindouble_y * domaindouble_z);
  // deepcopy host to device
  Kokkos::deep_copy(dev_LGF, host_LGF);

  double timerReadIn = timer.seconds();
  std::cout << " time to readIn = " << timerReadIn << std::endl;
  // Kokkos::parallel_for("Print symbol", Kokkos::RangePolicy<Kokkos::Cuda>(0, domaindouble_x * domaindouble_y * domaindouble_z), 
  //       KOKKOS_LAMBDA(const int i) {
  //           printf("device lgf[%d] = %f\n", i, dev_LGF[i]);
  //       });


//Kokkos::Timer timer();
  // Defining data vectors required for forward DFT in FFTX as Kokkos views
  using Complex = Kokkos::complex<double>;
  Kokkos::View<Complex*, Kokkos::DefaultExecutionSpace::memory_space> symbol("symbol_view", domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> dummy1("dummy1_view", domaindouble_x * domaindouble_y * domaindouble_z);
 
  std::vector<void*> args1 = [&]() {
      static auto symbol_data = symbol.data();
      static auto lgf_data = dev_LGF.data();
      static auto dummy1_data = dummy1.data();
      return std::vector<void*>{&symbol_data, &lgf_data, &dummy1_data};
  }();
// Calculate the symbol i.e. forward DFT of the lattice Green's function
    // std::complex<double> *symbol = new std::complex<double>[domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)];
    // double *dummy = new double[domaindouble_x * domaindouble_y * domaindouble_z];
    //MDDFT class
    // std::vector<void*> args{symbol, lgf_values.data(), dummy};
    std::vector<int> sizes{domaindouble_x, domaindouble_y, domaindouble_z};
    MDPRDFTProblem r2cdft{args1, sizes, "mdprdft"};
    r2cdft.transform();

//  for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i0++){      
//         std::cout << " i0 = ," << i0 << " " << symbol[i0] << std::endl;        
//     }

//  Kokkos::parallel_for("Print symbol", Kokkos::RangePolicy<Kokkos::Cuda>(0, domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)), 
//         KOKKOS_LAMBDA(const int i) {
//             // printf("i = %d", i);
//             printf("symbol[%d] = (%f,%f)\n", i, symbol(i).real(), symbol(i).imag());
//         });

//--------------------------------------------------------------//
//--------------------------------------------------------------//
// ************* Using RConv function in FFTX *****************//
//  // Using RConv function in FFTX to compute convolution between symbol and F1D_domaindouble
/*
  Kokkos::View<double*, Kokkos::CudaSpace> out_idft("Rconv output", domaindouble_x * domaindouble_y * domaindouble_z);
  std::vector<void*> args4 = [&]() {
       static auto output_data = out_idft.data();
       static auto F1D_data = F1D_domaindouble.data();
       static auto symbol1_data = symbol.data();
       return std::vector<void*>{&output_data, &F1D_data, &symbol1_data};
   }();
   std::vector<int> sizes4{domaindouble_x, domaindouble_y, domaindouble_z};
//   //rconv class
     RCONVProblem conv{args4, sizes4, "rconv"};

//     // // Run the transform
     conv.transform();
*/
//**************** END of of RCONV ****************************//
//--------------------------------------------------------------//
//--------------------------------------------------------------//

//**************** Using Individual FFTX functions for Convolution *******//
timer.reset();
 // Defining data vectors required for forward DFT of the input2 ie F1D in FFTX as Kokkos views
  Kokkos::View<Complex*,Kokkos::DefaultExecutionSpace::memory_space> F_dft("fwd_dft_view", domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> dummy2("dummy2_view", domaindouble_x * domaindouble_y * domaindouble_z);
  Kokkos::View<Complex*,Kokkos::DefaultExecutionSpace::memory_space> pointwise_mul("fwd_dft_view", domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> out_idft("inv_dft_view", domaindouble_x * domaindouble_y * domaindouble_z);
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

  std::vector<void*> args2 = [&]() {
      static auto Fdft_data = F_dft.data();
      static auto F1D_data = F1D_domaindouble.data();
      static auto dummy2_data = dummy2.data();
      return std::vector<void*>{&Fdft_data, &F1D_data, &dummy2_data};
  }();
//  // Calculate the forward DFT of the second input F1D_domaindouble
//     std::complex<double> *F_dft = new std::complex<double>[domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)];
//     double *dummy2 = new double[domaindouble_x * domaindouble_y * domaindouble_z];
//     //MDDFT class
    // std::vector<void*> args2{F_dft, F1D_domaindouble, dummy2};
    std::vector<int> sizes2{domaindouble_x, domaindouble_y, domaindouble_z};
    MDPRDFTProblem r2cdft2{args2, sizes2, "mdprdft"};
    r2cdft2.transform();
//  for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i0++){      
//         std::cout << " i0 = " << i0 << ",\tFpc_dft =  " << F_dft[i0] << std::endl;        
//     }
// Kokkos::parallel_for("Print F1DDFT", Kokkos::RangePolicy<Kokkos::Cuda>(0, domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)), 
//         KOKKOS_LAMBDA(const int i) {
//             // printf("i = %d", i);
//             printf("F1D_DFT[%d] = (%f,%f)\n", i, F_dft(i).real(), F_dft(i).imag());
//         });

// Pointwise Mulitply

Kokkos::parallel_for("Pointwise_multiply", Kokkos::RangePolicy<ExecutionSpace>(exec_space,0,domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)), 
        KOKKOS_LAMBDA(const int i) {
            double a = symbol(i).real();
            double b = symbol(i).imag();
            double c = F_dft(i).real();
            double d = F_dft(i).imag();

            //Pointwise multiply symbol and Fpc_dft
            double real_pw = a * c - b * d;
            double img_pw = a * d + b * c;

            pointwise_mul[i] = Complex(real_pw, img_pw);
            // printf("pwise[%d] = (%f,%f)\n", i, pointwise_mul(i).real(), pointwise_mul(i).imag());
        });

// Calculate the inverse dft to compute the final convolution value
std::vector<void*> args3 = [&]() {
      static auto out_data = out_idft.data();
      static auto pwise_data = pointwise_mul.data();
      static auto dummy3_data = dummy3.data();
      return std::vector<void*>{&out_data, &pwise_data, &dummy3_data};
  }();
//     double *out_idft = new double[domaindouble_x * domaindouble_y * domaindouble_z];
//     double *dummy3 = new double[domaindouble_x * domaindouble_y * domaindouble_z];
//     //MDDFT class
    //  std::vector<void*> args3{out_idft, pointwise_mul, dummy3};
    std::vector<int> sizes3{domaindouble_x, domaindouble_y, domaindouble_z};
    IMDPRDFTProblem c2rdft{args3, sizes3, "imdprdft"};
    c2rdft.transform();
    // Kokkos::parallel_for("Print inv output", Kokkos::RangePolicy<Kokkos::Cuda>(0, domaindouble_x * domaindouble_y * domaindouble_z), 
    //     KOKKOS_LAMBDA(const int i) {
    //         // printf("i = %d", i);
    //         // printf("inv_output[%d] = %f\n", i, out_idft(i));
    //     });
  
//     // for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * domaindouble_z; i0++){      
//     //     std::cout << " i0 = " << i0 << ",\tConvolution_dd =  " << out_idft[i0] << std::endl;        
//     // }

 //**************** END of INDIVIDUAL FUNCTIONS ****************************//
//--------------------------------------------------------------//
//--------------------------------------------------------------//
//  // Normalizing the output with h^3/dd^3
// Normalization Factor
double norm_factor = pow(h,3)/(domaindouble_x * domaindouble_y * domaindouble_z);
// printf("norm factor = %f\n", norm_factor);
Kokkos::parallel_for("Normalize output", Kokkos::RangePolicy<ExecutionSpace>(exec_space,0,domaindouble_x * domaindouble_y * domaindouble_z), 
        KOKKOS_LAMBDA(const int i) {

            out_normalize[i] = norm_factor * out_idft[i];
            // Kokkos::printf("out_normalize[%d] = %f\n", i, out_normalize[i]);
        });

//double timeConv = timer.seconds();
Kokkos::parallel_for("Normalize output", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(exec_space, {0, 0, 0}, {extent, extent, extent}), 
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            int out_dd_index = k * domaindouble_y * domaindouble_x + j * domaindouble_x + i;
            int out_original_index = k * extent * extent + j * extent + i;
            conv_output[out_original_index] = out_normalize[out_dd_index];
//            printf("conv_output[%d] = %f\n", out_original_index, conv_output[out_original_index]);
        });
auto velocity_g = pm.get(Location::Node(), Field::Velocity());

int N = extent;
double U = 1.0;


/*    if(d == 2){

Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N,N,N}),
     KOKKOS_LAMBDA(const int i, const int j, const int k) {

              int index_f = i * N * N + j * N + k;
              velocity_g(i,j,k,d) = 1.0;
              velx(i,j,k,0)       = 1.0;

     });
}
*/

 Kokkos::fence();
 double timerFFTX = timer.seconds();
 std::cout << timerFFTX << " FFTX " << std::endl;

Kokkos::printf("velocity added");
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
/*          if( r < (0.25-1e-10) ){

            v_exact[0] = -1.5*xz/R2*U;
            v_exact[1] = -1.5*yz/R2*U;
            v_exact[2] = -1.5*( 1.0 + x[2]*x[2]/R2 -  2.0*r*r/R2  ) *U;
          }else{

            v_exact[0] = -1.5 * (  xz / pow( r, 5.0) )*R3*U;
            v_exact[1] = -1.5 * (  yz / pow( r, 5.0) )*R3*U;
            v_exact[2] = U*( 1.0 + R3 / (2.0*pow( r, 3.0 ) ) ) - U*1.5*R3 / (  pow( r, 5.0 ) ) * x[2]*x[2] ;

         }
*/

          if(d == 2 ){
            velocity_g(i,j,k,d) = conv_output[index_f] ;
            velx(i,j,k,0)  = (conv_output[index_f]);
//            velocity_g(i,j,k,d) += 1.0;
//            velx(i,j,k,0)  += 1.0;
          }else if(d == 1){

            velocity_g(i,j,k,d) = conv_output[index_f];
            velx(i,j,k,0)  = conv_output[index_f];
          }else if(d == 0){

            velocity_g(i,j,k,d) = conv_output[index_f];
            velx(i,j,k,0) =  conv_output[index_f];
          }

//          Kokkos::printf(" conv output %f d %d \n", conv_output[index_f], d);

   });

//   Kokkos::printf("error");
//   std::stringstream ss;
//   ss << d << "_Velocity";
//   pm.save_v( ss.str(),1,0.0);
//   pm.save_v("velocity_0",1,0.0);


  }






  //**CPU implementation **// 
  // // // std::complex<double> *pointwise_mul = new std::complex<double>[domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)];
// // // for(int i = 0; i < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i++){
// // //   //std::cout << "Real = " << symbol[i].real() << std::endl;
// // //   double a = symbol[i].real();
// // //   double b = symbol[i].imag();
// // //   double c = F_dft[i].real();
// // //   double d = F_dft[i].imag();

// // //   //Pointwise multiply symbol and Fpc_dft
// // //   double real_pw = a * c - b * d;
// // //   double img_pw = a * d + b * c;

// // //   pointwise_mul[i] = std::complex<double>(real_pw, img_pw);

// // //   //std::cout << "i = " << i << "\t Pointwise_mul = " << pointwise_mul[i]<<std::endl;
// // // }   
// //  double *out_normalize = new double[domaindouble_x * domaindouble_y * domaindouble_z];
// //     double norm_factor = pow(h,3)/(domaindouble_x * domaindouble_y * domaindouble_z);
// //     for(int i = 0; i < domaindouble_x*domaindouble_y*domaindouble_z; i++){
// //        out_normalize[i] = norm_factor * out_idft[i];
// //     }
// //   double *conv_output = new double[extent * extent * extent];
// //   // Extracting the output for the orginal size from the above domain doubled output
// //   for(int i = 0; i < extent; i++){
// //     for(int j = 0; j < extent; j++){
// //       for(int k = 0; k < extent; k++){
// //         //Calculate the index in the domain doubled output vector
// //             int out_dd_index = i * domaindouble_y * domaindouble_z + j * domaindouble_z + k;
// //         // Calculate the index in the smaller output of the orginal domain size 
// //             int out_original_index = i * extent * extent + j * extent + k;
// //         // Copying the values from larger to smaller output vector
// //             conv_output[out_original_index] = out_normalize[out_dd_index];
// //       }
// //     }
// //   }
// printf("END OF CONVOLUTION!!! \n");
//   //  for(int i0 = 0; i0 < extent*extent*extent; i0++){      
//   //       std::cout << "i =  " << i0 << "\t Extracted Conv output = "<< conv_output[i0] << std::endl;        
//   //   }





 // Convolution F*G
  
  //   double *output = new double[domaindouble_x * domaindouble_y * domaindouble_z];
  //   // //Vector of void pointers
  //   args.clear();
  //   args.push_back(output);
  //   args.push_back(F1D_domaindouble);
  //   args.push_back(symbol);

  //   sizes.clear();
  //   sizes.push_back(domaindouble_x);
  //   sizes.push_back(domaindouble_y);
  //   sizes.push_back(domaindouble_z);

  //   //rconv class
  //   RCONVProblem conv{args, sizes, "rconv"};

  //   // // Run the transform
  //   conv.transform();

  //   // for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * domaindouble_z ; i0++){      
  //   //     std::cout << " " << output[i0] << std::endl;        
  //   // }

  // double *conv_output = new double[extent * extent * extent];
  // // Extracting the output for the orginal size from the above domain doubled output
  // for(int i = 0; i < extent; i++){
  //   for(int j = 0; j < extent; j++){
  //     for(int k = 0; k < extent; k++){
  //       //Calculate the index in the domain doubled output vector
  //           int out_dd_index = i * domaindouble_y * domaindouble_z + j * domaindouble_z + k;
  //       // Calculate the index in the smaller output of the orginal domain size 
  //           int out_original_index = i * extent * extent + j * extent + k;
  //       // Copying the values from larger to smaller output vector
  //           conv_output[out_original_index] = output[out_dd_index];
  //     }
  //   }
  // }
  // Kokkos::printf("This is the EXTRACTED OUTPUT!!! \n");
  // for(int i0 = 0; i0 < extent*extent*extent; i0++){      
  //       std::cout << " " << conv_output[i0] << std::endl;        
  //   }

  // // Converting the extracted convolution output from FFTX into a 3D Kokkos View
  // Kokkos::printf("KOKKOS VIEW FROM 1D to 3D!!!!\n");
  // using View3D = Kokkos::View<double***, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
  // View3D view(conv_output, extent, extent, extent);
  // // Parallel loop using Kokkos to modify the view
  //       Kokkos::parallel_for("ModifyView", extent, KOKKOS_LAMBDA(int i) {
  //           for (int j = 0; j < extent; ++j) {
  //               for (int k = 0; k < extent; ++k) {
  //                   // double x[3] = { i*h - center, j*h - center, k*h - center };
  //                   // double r    = pow( pow(x[0], 2.0) + pow(x[1],2.0) + pow(x[2],2.0),0.5);

  //                   view(i, j, k) = conv_output[i*extent*extent +  j*extent + k];  
  //                   // velx(i,j,k,0) = view(i,j,k); // 1.0 ; //abs(view(i,j,k) - x[2]/(4.0*Kokkos::numbers::pi*pow(r, 3.0) ));


  //               }
  //           }
  //       });

  // for (int i = 0; i < extent; ++i) {
  //           for (int j = 0; j < extent; ++j) {
  //               for (int k = 0; k < extent; ++k) {

  //                   double x[3] = { i*h - center, j*h - center, k*h - center };
  //                   double r    = pow( pow(x[0], 2.0) + pow(x[1],2.0) + pow(x[2],2.0),0.5);
  //                   std::cout << "view(" << i << "," << j << "," << k << ") = "
  //                             << view(i, j, k) << " exact " << 1.0/(4.0*Kokkos::numbers::pi*pow(r, 3.0) )  <<"\n";
  //               }
  //           }
  //       }

  //        pm.save_v( "convolution_error",1,0);
// delete[] conv_output;

// //****************************//
// //         FFTW test
// // ***************************//
// // Input 1 = lgf_values -> domain doubled symbol
// // Input 2 = F1D_domaindouble

// std::vector<std::complex<double>> temp(domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
// std::vector<std::complex<double>> out1(domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
// std::vector<std::complex<double>> out2(domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
// std::vector<double> out_final(domaindouble_x * domaindouble_y * domaindouble_z);

// //FFTW call to compute r2c dft
// fftw_plan p1 = fftw_plan_dft_r2c_3d(domaindouble_x, domaindouble_y, domaindouble_z, lgf_values.data(),
//                   (fftw_complex*)out1.data(), FFTW_ESTIMATE);
// fftw_execute(p1);                  
// // for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i0++){      
// //          std::cout << " i0 =" << i0 << "\t" << out1[i0] << " , symbol - fftw_out1 = "<< symbol[i0] - out1[i0]<< std::endl;
// // }
// fftw_plan p2 = fftw_plan_dft_r2c_3d(domaindouble_x, domaindouble_y, domaindouble_z, F1D_domaindouble,
//                   (fftw_complex*)out2.data(), FFTW_ESTIMATE);
// fftw_execute(p2);

// // Pointwise Mulitply

// auto complex_multiply = std::multiplies<
//                             std::complex<double>>{}; 
//     std::transform(out2.begin(), //start location 
//                 out2.end(), //end location
//                 out1.data(), //2nd input
//                 temp.begin(), //output 
//                 complex_multiply); //operator

// // for(int i = 0; i < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i++){
// //   //temp[i] = out2[i] * out1[i];
// //   std::cout << "i = "<< i << " temp =  " << temp[i] << std::endl;
// // }

// // Inverse FFT using c2r
// fftw_plan p3 = fftw_plan_dft_c2r_3d(domaindouble_x, domaindouble_y, domaindouble_z, (fftw_complex*)temp.data(),
//                   out_final.data(), FFTW_ESTIMATE);

// fftw_execute(p3);

// // for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * domaindouble_z; i0++){      
// //          std::cout << " i0 = ," << i0 << " " << out_final[i0] << "SPIRAL output = " << output[i0]<<" , SPIRAL - fftw = "<< output[i0] - out_final[i0]<< std::endl;
// // }

// double *extract_output_fftw = new double[extent * extent * extent];
//   // Extracting the output for the orginal size from the above domain doubled output
//   for(int i = 0; i < extent; i++){
//     for(int j = 0; j < extent; j++){
//       for(int k = 0; k < extent; k++){
//         //Calculate the index in the domain doubled output vector
//             int dd_index_fftw = i * domaindouble_y * domaindouble_z + j * domaindouble_z + k;
//         // Calculate the index in the smaller output of the orginal domain size 
//             int original_index_fftw = i * extent * extent + j * extent + k;
//         // Copying the values from larger to smaller output vector
//           extract_output_fftw[original_index_fftw] = out_final[dd_index_fftw];
//       }
//     }
//   }
// for(int i0 = 0; i0 < extent * extent * extent; i0++){      
//          std::cout << " i0 = " << i0 << "\t"<< extract_output_fftw[i0] << std::endl;
// }
//   for(int i0 = 0; i0 < extent * extent * extent; i0++){      
//          std::cout << " i0 = " << i0 << "\t"<< extract_output_fftw[i0] << "\t SPIRAL output = " << conv_output[i0]<<" , SPIRAL - fftw = "<< conv_output[i0] - extract_output_fftw[i0]<< std::endl;
// }
// std::vector<double> input(32*32*32, 1.0);
//     std::vector<std::complex<double>> outputf(32*32*32, 0);
//     // fftw_plan f = fftw_plan_dft_c2r_3d(32,32,32, input.data(), (fftw_complex*)output.data(), FFTW_ESTIMATE);
//     fftw_plan p = fftw_plan_dft_r2c_3d(32, 32, 32, input.data(),
//                   (fftw_complex*)outputf.data(), FFTW_ESTIMATE);
//     fftw_execute(p); 
//     std::cout << "This is FFTW!! "<<outputf[0] << std::endl;
//  int half_extent = extent/2;

//  for(int i = 0; i < domaindouble_x; i++){
//   for(int j = 0; j < domaindouble_y; j++){
//     for(int k = 0; k < domaindouble_z; k++){

//       int index_dd = i*domaindouble_y*domaindouble_z + j*domaindouble_z + k;
//       //std::cout<< index_dd << std::endl;
//       if((i >= half_extent && i < ((half_extent) + extent)) && 
//          (j >= half_extent && j < ((half_extent) + extent)) &&
//          (k >= half_extent && k < ((half_extent) + extent))){
        
//         int index_lgf = (i-half_extent) * extent * extent + (j-half_extent) * extent + (k-half_extent);
//         latticeGreensfunc[index_dd] = lgf_values[index_lgf];
//       }
//       else{
//         latticeGreensfunc[index_dd] = 0.0;
//       }
//       // std::cout << "index_dd = " << index_dd << "LGF = " << latticeGreensfunc[index_dd] << std::endl;
//     }
//   }
//  }

  
  // double *GreensFunc_K = new double[(extent-1)*(extent-1)*(extent-1)];
//   double scal_GreensFunc;
  
// //extent = no. cells
// // actual no. of points  = extent + 1
//   //iterate over D
//      for(int i = 0; i < extent+1; i++){
//        for(int j = 0; j < extent+1; j++){
//          for(int k = 0; k < extent+1; k++){
            
//             //xp
//             double xg[3] = { i*h - center, j*h - center, k*h - center };
            
//             // Iterate over D0
//            //for(int kk = 0; kk<9; kk++){
//               for(int i0 = 1; i0 < extent; i0++){
//                 for(int j0 = 1; j0 < extent; j0++){
//                   for(int k0 = 1; k0 < extent; k0++){
                  
//                   // xq
//                   double xg0[3] = { i0*h - center, j0*h - center, k0*h - center };
                
//                   double K[9];
                  
//                   // Computing the K matrix from the Almgren paper
//                   int index_K = (i0-1) * (extent-1) * (extent-1) + (j0-1) * (extent-1) + (k0-1);
//                   //GreensFunction::CalculateK(xg, xg0, K);
//                   GreensFunction::Calculate_scalarK(xg, xg0, &scal_GreensFunc);
//                   // for(int kk = 0; kk<9; kk++){
//                     GreensFunc_K[index_K] = scal_GreensFunc; //1.0; //K[kk];
//                    // printf("GreensFunc_K[%d] = %f \n", index_K, GreensFunc_K[index_K]);
//                   // }
//                   }  
                  
//                 }
//               }
//            //}

           
//          }
//        }
//      }

    //  // Calculate the symbol i.e. forward DFT of the kernel K=GreensFunc_K
    // std::complex<double> *symbol = new std::complex<double>[(extent-2)*(extent-2)*(extent-2)];
    // double *dummy = new double[(extent-2)*(extent-2)*(extent-2)];
    // //MDDFT class
    // std::vector<void*> args{symbol, GreensFunc_K, dummy};
    // std::vector<int> sizes{extent-2,extent-2,extent-2};
    // MDPRDFTProblem r2cdft{args, sizes, "mdprdft"};
    // r2cdft.transform();
    // // for(int i0 = 0; i0 < (extent-2)*(extent-2)*(extent-2); i0++){      
    // //     std::cout << " " << symbol[i0] << std::endl;        
    // // }
    
    // // Convolution F*G
    // double *output = new double[extent*extent*extent];
  
    // // //Vector of void pointers
    // args.clear();
    // args.push_back(output);
    // args.push_back(F1D_vec);
    // args.push_back(symbol);

    // sizes.clear();
    // sizes.push_back(extent);
    // sizes.push_back(extent);
    // sizes.push_back(extent);

    // //rconv class
    // RCONVProblem conv{args, sizes, "rconv"};

    // // // Run the transform
    // conv.transform();

    // // for(int i0 = 0; i0 < (extent)*(extent)*(extent); i0++){      
    // //     std::cout << " " << output[i0] << std::endl;        
    // // }

}
}
}
#endif
