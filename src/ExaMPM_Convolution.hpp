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

#ifndef EXAMPM_CONVOLUTION_HPP
#define EXAMPM_CONVOLUTION_HPP

#include <ExaMPM_MLC_Interp.hpp>
#include <ExaMPM_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <ExaMPM_GreensFunction.hpp>
#include <ExaMPM_GridManager.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include "interface.hpp"
#include "mdprdftObj.hpp"
namespace ExaMPM
{
namespace Convolution
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
  auto F = pm.get( Location::Node(),Field::F() );
  
  Kokkos::View<double*> F1D("Fvector", extent*extent*extent);

  // Copy data from 3D to 1D using a parallel loop
    Kokkos::parallel_for("Copy 3D to 1D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {extent, extent, extent}), 
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            int index = i * extent * extent + j * extent + k;
            F1D(index) = F(i, j, k,0);
           // Kokkos::printf("F1D = %f, index = %d \n", F1D(index), index);
        });
  //  Kokkos::parallel_for("Copy 3D to 1D", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {extent, extent, extent, 3}), 
  //       KOKKOS_LAMBDA(const int i, const int j, const int k, const int m) {
  //           int index = i * extent * extent + j * extent + k;
  //           F1D(index) = F(i, j, k,0);
  //       });

  // Green's function
  //Kokkos::View<double*> GreensFunc_K("Fvector", (extent-2)*(extent-2)*(extent-2));
  double *GreensFunc_K = new double[(extent-2)*(extent-2)*(extent-2)];

  //iterate over D
     for(int i = 0; i < extent; i++){
       for(int j = 0; j < extent; j++){
         for(int k = 0; k < extent; k++){
            
            //xp
            double xg[3] = { i*h - center, j*h - center, k*h - center };
            
            // Iterate over D0
           for(int i0 = 0; i0 < extent-2; i0++){
              for(int j0 = 0; j0 < extent-2; j0++){
                for(int k0 = 0; k0 < extent-2; k0++){
                  
                  // xq
                  double xg0[3] = { i0*h - center, j0*h - center, k0*h - center };
                
                  double K[9];

                  // Computing the K matrix from the Almgren paper
                  int index_K = i0 * (extent-2) * (extent-2) + j0 * (extent-2) + k0;
                  //GreensFunc_K[index_K] = GreensFunction::CalculateK(xg, xg0, K);
                  GreensFunction::CalculateK(xg, xg0, K);
                  for(int kk = 0; kk<9; kk++){
                    GreensFunc_K[index_K] = 1.0; //K[kk];
                    //printf("GreensFunc_K[%d] = %f \n", index_K, GreensFunc_K[index_K]);
                  }
                  
                  
                }
              }
           }

           
         }
       }
     }

     // Calculate the symbol i.e. forward DFT of the kernel K=GreensFunc_K
    std::complex<double> *symbol = new std::complex<double>[(extent-2)*(extent-2)*(extent-2)];
    double *dummy = new double[(extent-2)*(extent-2)*(extent-2)];
    //MDDFT class
    std::vector<void*> args{symbol, GreensFunc_K, dummy};
    std::vector<int> sizes{extent-2,extent-2,extent-2};
    MDPRDFTProblem r2cdft{args, sizes, "mdprdft"};
    r2cdft.transform();
    for(int i0 = 0; i0 < (extent-2)*(extent-2)*(extent-2); i0++){      
        std::cout << " " << symbol[i0] << std::endl;        
    }

    

  // Convolution
    // Computing the Symbol = FFT(F1D)
    //std::complex<double> *symbol = new std::complex<double>[10*10*10];
    //MDDFT class
    //MDDFTProblem r2cdft{args, sizes, "mddft"};
    // double *input = new double[extent*extent*extent];
    // double *phi = new double[extent*extent*extent];
    // std::complex<double> *symbol = new std::complex<double>[10*10*10];
    //Vector of void pointers
    // std::vector<void*> args{output, F1D.data(), symbol};
    // std::vector<int> sizes{10,10,10};

}
}
}
#endif
