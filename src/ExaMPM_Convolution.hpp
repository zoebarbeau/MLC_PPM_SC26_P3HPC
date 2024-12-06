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
#include "rconvObj.hpp"
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
    // Kokkos::parallel_for("Copy 4D to 1D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {extent, extent, extent}), 
    //     KOKKOS_LAMBDA(const int i, const int j, const int k) {
    //         int index = i * extent * extent + j * extent + k;
    //         F1D(index) = F(i, j, k,0);
    //        // Kokkos::printf("F1D = %f, index = %d \n", F1D(index), index);
    //     });
   Kokkos::parallel_for("Copy 4D to 1D", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {extent, extent, extent, 1}), 
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int m) {
            int index = i * extent * extent + j * extent + k;
            F1D(index) = F(i, j, k,0);
            //Kokkos::printf("F1D = %f, index = %d \n", F1D(index), index);
        });
  double *F1D_vec = F1D.data();
  // Green's function
  //Kokkos::View<double*> GreensFunc_K("Fvector", (extent-2)*(extent-2)*(extent-2));
  double *GreensFunc_K = new double[(extent-1)*(extent-1)*(extent-1)];
  double scal_GreensFunc;
  
//extent = no. cells
// actual no. of points  = extent + 1
  //iterate over D
     for(int i = 0; i < extent+1; i++){
       for(int j = 0; j < extent+1; j++){
         for(int k = 0; k < extent+1; k++){
            
            //xp
            double xg[3] = { i*h - center, j*h - center, k*h - center };
            
            // Iterate over D0
           //for(int kk = 0; kk<9; kk++){
              for(int i0 = 1; i0 < extent; i0++){
                for(int j0 = 1; j0 < extent; j0++){
                  for(int k0 = 1; k0 < extent; k0++){
                  
                  // xq
                  double xg0[3] = { i0*h - center, j0*h - center, k0*h - center };
                
                  double K[9];
                  
                  // Computing the K matrix from the Almgren paper
                  int index_K = (i0-1) * (extent-1) * (extent-1) + (j0-1) * (extent-1) + (k0-1);
                  //GreensFunction::CalculateK(xg, xg0, K);
                  GreensFunction::Calculate_scalarK(xg, xg0, &scal_GreensFunc);
                  // for(int kk = 0; kk<9; kk++){
                    GreensFunc_K[index_K] = scal_GreensFunc; //1.0; //K[kk];
                    printf("GreensFunc_K[%d] = %f \n", index_K, GreensFunc_K[index_K]);
                  // }
                  }  
                  
                }
              }
           //}

           
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
    
    // Convolution F*G
    double *output = new double[extent*extent*extent];
  
    // //Vector of void pointers
    args.clear();
    args.push_back(output);
    args.push_back(F1D_vec);
    args.push_back(symbol);

    sizes.clear();
    sizes.push_back(extent);
    sizes.push_back(extent);
    sizes.push_back(extent);

    //rconv class
    RCONVProblem conv{args, sizes, "rconv"};

    // // Run the transform
    conv.transform();

    for(int i0 = 0; i0 < (extent)*(extent)*(extent); i0++){      
        std::cout << " " << output[i0] << std::endl;        
    }

}
}
}
#endif
