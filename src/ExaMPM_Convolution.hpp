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
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
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
   Kokkos::parallel_for("Copy 4D to 1D", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {extent, extent, extent, 1}), 
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int m) {
            int index = i * extent * extent + j * extent + k;
            F1D(index) = F(i, j, k,0);
            //Kokkos::printf("F1D = %f, index = %d \n", F1D(index), index);
        });
  double *F1D_vec = F1D.data();

 
  // Lattice Green's function
  std::ifstream infileLGF("/home/h82/Documents/Bluestone/P3M/MLC_PPM/LatticeGreensFunction/exec/phiTrimmed");
  std::vector<double> lgf_values;
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

      if((xdir >= (-extent/2) && xdir < (extent/2)) && (ydir >= (-extent/2) && ydir < (extent/2)) && (zdir >= (-extent/2) && zdir < (extent/2))){
          lgf_values.push_back(lgf);
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

 int domaindouble_x = 2*extent;
 int domaindouble_y = 2*extent;
 int domaindouble_z = 2*extent;
 double* latticeGreensfunc = new double[domaindouble_x*domaindouble_y*domaindouble_z];
 double* F1D_domaindouble = new double[domaindouble_x*domaindouble_y*domaindouble_z];

 for(int i = 0; i < domaindouble_x; i++){
   for(int j = 0; j < domaindouble_y; j++){
     for(int k = 0; k < domaindouble_z; k++){

        int index_dd = i*domaindouble_y*domaindouble_z + j*domaindouble_z + k;
        if(i < extent && j < extent && k < extent){
          int index_lgf = i * extent * extent + j * extent + k;
          latticeGreensfunc[index_dd] = lgf_values[index_lgf];
          F1D_domaindouble[index_dd] = F1D_vec[index_lgf];     
        }
        else{
          latticeGreensfunc[index_dd] = 0.0;
          F1D_domaindouble[index_dd] = 0.0;     
        }
     }
   }
  }
// for(int i = 0; i < domaindouble_x*domaindouble_y*domaindouble_z; i++){
//   std::cout<<F1D_domaindouble[i] << std::endl;
// }

// Calculate the symbol i.e. forward DFT of the lattice Green's function
    std::complex<double> *symbol = new std::complex<double>[domaindouble_x * domaindouble_y * domaindouble_z];
    double *dummy = new double[domaindouble_x * domaindouble_y * domaindouble_z];
    //MDDFT class
    std::vector<void*> args{symbol, latticeGreensfunc, dummy};
    std::vector<int> sizes{domaindouble_x, domaindouble_y, domaindouble_z};
    MDPRDFTProblem r2cdft{args, sizes, "mdprdft"};
    r2cdft.transform();
    // for(int i0 = 0; i0 < domaindouble_x*domaindouble_y*domaindouble_z; i0++){      
    //     std::cout << " " << symbol[i0] << std::endl;        
    // }
 // Convolution F*G
    double *output = new double[domaindouble_x * domaindouble_y * domaindouble_z];
  
    // //Vector of void pointers
    args.clear();
    args.push_back(output);
    args.push_back(F1D_domaindouble);
    args.push_back(symbol);

    sizes.clear();
    sizes.push_back(domaindouble_x);
    sizes.push_back(domaindouble_y);
    sizes.push_back(domaindouble_z);

    //rconv class
    RCONVProblem conv{args, sizes, "rconv"};

    // // Run the transform
    conv.transform();

    for(int i0 = 0; i0 < (domaindouble_x)*(domaindouble_y)*(domaindouble_z); i0++){      
        std::cout << " " << output[i0] << std::endl;        
    }

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
