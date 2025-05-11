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

#ifndef EXAMPM_CONVOLUTION_FFTX_HPP
#define EXAMPM_CONVOLUTION_FFTX_HPP

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
#include "fftx3.hpp"
#include "interface.hpp"
#include "mdprdftObj.hpp"
#include "imdprdftObj.hpp"
#include "rconvObj.hpp"
// #include "fftw3.h"
namespace ExaMPM
{
namespace ConvolutionFFTX
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
void Conv(const ExecutionSpace& exec_space, const ProblemManagerType& pm, const int extent, const double center, const double h, const int d)
{
  auto F = pm.get( Location::Node(),Field::F() );
  Kokkos::View<double*> F1D("Fvector", extent*extent*extent);

  // Copy data from 4D to 1D using a parallel loop
   Kokkos::parallel_for("Copy 4D to 1D", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {extent, extent, extent, 1}), 
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int m) {
            int index = i * extent * extent + j * extent + k;
            F1D(index) = F(i, j, k, d);
        //    std::cout << "F1D " << F1D(index) << std::endl;
        });
  double *F1D_vec = F1D.data();

// Dims for double domain
  int domaindouble_x = 2*extent;
  int domaindouble_y = 2*extent;
  int domaindouble_z = 2*extent;

// Domain doubling the Velocity vector F1D so that it can be used as input for FFTX convolution function
// The data cube is placed in the upper right corner of the doubled domain cube
 double* F1D_domaindouble = new double[domaindouble_x*domaindouble_y*domaindouble_z];

 for(int i = 0; i < domaindouble_x; i++){
   for(int j = 0; j < domaindouble_y; j++){
     for(int k = 0; k < domaindouble_z; k++){

        int index_dd = i*domaindouble_y*domaindouble_z + j*domaindouble_z + k;

/*        if((i < extent) && (j >= extent && j < domaindouble_y) && (k < extent)){
          int index_lgf = i * extent * extent + ((j-extent) * extent) + k;
          F1D_domaindouble[index_dd] = F1D_vec[index_lgf];     
        }
        else{ */
          F1D_domaindouble[index_dd] = 0.0;     
      //  }
       // std::cout << "index_dd=" << index_dd << "\t" << "Fpc_1D_dd = " <<F1D_domaindouble[index_dd] << std::endl;
     }
   }
  }


  
  for (int i = 0; i < extent ; i++)
    {
      for (int j = 0; j < extent; j++)
        {
          for (int k = 0;  k < extent; k++)
            {
              int index_dd = (i + extent)*4*extent*extent + (j + extent)*2*extent + k + extent;
              int index_1d = i * extent * extent + j * extent + k;
              F1D_domaindouble[index_dd] = F1D_vec[index_1d];
            }
        }
    }


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
  
// Calculate the symbol i.e. forward DFT of the lattice Green's function
    std::complex<double> *symbol = new std::complex<double>[domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)];
    double *dummy = new double[domaindouble_x * domaindouble_y * domaindouble_z];
    //MDDFT class
    std::vector<void*> args{symbol, lgf_values.data(), dummy};
    std::vector<int> sizes{domaindouble_x, domaindouble_y, domaindouble_z};
    MDPRDFTProblem r2cdft{args, sizes, "mdprdft"};
    r2cdft.transform();
//  for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i0++){      
//         std::cout << " i0 = ," << i0 << " " << symbol[i0] << std::endl;        
//     }
 
 // Calculate the forward DFT of the second input F1D_domaindouble
    std::complex<double> *F_dft = new std::complex<double>[domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)];
    double *dummy2 = new double[domaindouble_x * domaindouble_y * domaindouble_z];
    //MDDFT class
    //std::vector<void*> args{symbol, greens_func.data(), dummy};
     std::vector<void*> args2{F_dft, F1D_domaindouble, dummy2};
    std::vector<int> sizes2{domaindouble_x, domaindouble_y, domaindouble_z};
    MDPRDFTProblem r2cdft2{args2, sizes2, "mdprdft"};
    r2cdft2.transform();
//  for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i0++){      
//         std::cout << " i0 = " << i0 << ",\tFpc_dft =  " << F_dft[i0] << std::endl;        
//     }

// Pointwise Mulitply
std::complex<double> *pointwise_mul = new std::complex<double>[domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)];
for(int i = 0; i < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i++){
  //std::cout << "Real = " << symbol[i].real() << std::endl;
  double a = symbol[i].real();
  double b = symbol[i].imag();
  double c = F_dft[i].real();
  double d = F_dft[i].imag();

  //Pointwise multiply symbol and Fpc_dft
  double real_pw = a * c - b * d;
  double img_pw = a * d + b * c;

  pointwise_mul[i] = std::complex<double>(real_pw, img_pw);

  //std::cout << "i = " << i << "\t Pointwise_mul = " << pointwise_mul[i]<<std::endl;
}

// Calculate the inverse dft to compute the final convolution value
    double *out_idft = new double[domaindouble_x * domaindouble_y * domaindouble_z];
    double *dummy3 = new double[domaindouble_x * domaindouble_y * domaindouble_z];
    //MDDFT class
    //std::vector<void*> args{symbol, greens_func.data(), dummy};
     std::vector<void*> args3{out_idft, pointwise_mul, dummy3};
    std::vector<int> sizes3{domaindouble_x, domaindouble_y, domaindouble_z};
    IMDPRDFTProblem c2rdft{args3, sizes3, "imdprdft"};
    c2rdft.transform();
  
    for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * domaindouble_z; i0++){      
  //    std::cout << " i0 = " << i0 << ",\tConvolution_dd =  " << out_idft[i0] << std::endl;        
    }
 
 // Normalizing the output with h^3/dd^3
 double *out_normalize = new double[domaindouble_x * domaindouble_y * domaindouble_z];
    double norm_factor = pow(h,3)/(domaindouble_x * domaindouble_y * domaindouble_z);
    for(int i = 0; i < domaindouble_x*domaindouble_y*domaindouble_z; i++){
       out_normalize[i] = norm_factor * out_idft[i];
    }
  double *conv_output = new double[extent * extent * extent];
  // Extracting the output for the orginal size from the above domain doubled output
  for(int i = 0; i < extent; i++){
    for(int j = 0; j < extent; j++){
      for(int k = 0; k < extent; k++){
        //Calculate the index in the domain doubled output vector
            int out_dd_index = i * domaindouble_y * domaindouble_z + j * domaindouble_z + k;
        // Calculate the index in the smaller output of the orginal domain size 
            int out_original_index = i * extent * extent + j * extent + k;
        // Copying the values from larger to smaller output vector
            conv_output[out_original_index] = out_normalize[out_dd_index];
      }
    }
  }

   for(int i0 = 0; i0 < extent*extent*extent; i0++){      
  //      std::cout << "i =  " << i0 << "\t Extracted Conv output = "<< conv_output[i0] << std::endl;        
    }

     auto velx = pm.get(Location::Node(), Field::velx());
     auto velocity_g = pm.get(Location::Node(), Field::Velocity());
     
     int N = extent;
     double U = 1.0;
     if(d == 2){

        Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N+1,N+1,N+1}),
           KOKKOS_LAMBDA(const int i, const int j, const int k) {

              int index_f = i * N * N + j * N + k;
              velocity_g(i,j,k,d) = U;
              velx(i,j,k,0)       = U;

        });

     }



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
          if( r < (0.25-1e-10) ){

            v_exact[0] = -1.5*xz/R2*U;
            v_exact[1] = -1.5*yz/R2*U;
            v_exact[2] = -1.5*( 1.0 + x[2]*x[2]/R2 -  2.0*r*r/R2  ) *U;
          }else{

            v_exact[0] = -1.5 * (  xz / pow( r, 5.0) )*R3*U;
            v_exact[1] = -1.5 * (  yz / pow( r, 5.0) )*R3*U;
            v_exact[2] = U*( 1.0 + R3 / (2.0*pow( r, 3.0 ) ) ) - U*1.5*R3 / (  pow( r, 5.0 ) ) * x[2]*x[2] ;

         }
 
         
          if(d == 2 ){
            velocity_g(i,j,k,d) += conv_output[index_f];
            velx(i,j,k,0)  += (conv_output[index_f]);
          }else if(d == 1){

            velocity_g(i,j,k,d) = conv_output[index_f];
            velx(i,j,k,0)  = conv_output[index_f];
          }else if(d == 0){

            velocity_g(i,j,k,d) = conv_output[index_f];
            velx(i,j,k,0) =  conv_output[index_f];
          }

//          Kokkos::printf("Velocity Error %f %f %f \n ",velocity_g(i,j,k,d),v_exact[d],velocity_g(i,j,k,d)-v_exact[d]);
   });

   std::stringstream ss;
   ss << d << "_Velocity";
   pm.save_v( ss.str(),1,0.0);



}
}
}
#endif
