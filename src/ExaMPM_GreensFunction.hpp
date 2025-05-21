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

#ifndef EXAMPM_GREENSFUNCTION_HPP
#define EXAMPM_GREENSFUNCTION_HPP

#include <ExaMPM_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <ExaMPM_DenseLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

namespace ExaMPM
{
namespace GreensFunction
{
//---------------------------------------------------------------------------//
// Particle-to-grid.
//
KOKKOS_INLINE_FUNCTION
void Calculate_qK(const double xp[3],const double xq[3], const double up[3], double K[3],const double h, const int corr_radius)
{
   double r = pow( pow( xp[0] - xq[0], 2) + pow( xp[1] - xq[1], 2) + pow( xp[2] - xq[2], 2), 0.5);
   double K_M[3][3] = { { 0, (xp[2] - xq[2]), -1*(xp[1] - xq[1])},
                        { -1*(xp[2] - xq[2]), 0, (xp[0] - xq[0])},
                        { (xp[1] - xq[1]), -1*(xp[0] - xq[0]), 0} };
   double delta = 0.5*h;
   if( r < (delta - 1e-10)) 
   {
   

       
      for(int d0 = 0; d0 < 3; d0++){
         for(int d1 = 0; d1 < 3; d1++){


           K_M[d0][d1] *= 1.0/8.0 * ( -12.0*(r*r / (delta*delta) ) + 20 ) / (delta*delta*delta) * 1.0/(4.0*Kokkos::numbers::pi); //1.0/(4.0*Kokkos::numbers::pi)*(-3.0*pow(r/delta, 4.0) + 10.0*pow(r/delta,2.0) - 7.0 ) / 60.0; //(4-3*r/pow(delta, 3.0));
        }

      }

       
      
      DenseLinearAlgebra::matVecMultiply(K_M, up, K);


   }else
   {
      for(int d0 = 0; d0 < 3; d0++){
         for(int d1 = 0; d1 < 3; d1++){

	 
           K_M[d0][d1] *= 1.0/(4.0*Kokkos::numbers::pi*pow(r, 3.0) );
        }
 
      }

      DenseLinearAlgebra::matVecMultiply(K_M, up, K);
   }

/*   Kokkos::printf("delta %f KM11 %f KM12 %f KM 13 %f KM21 %f KM22 %f KM23 %f KM31 %f KM32 %f KM33 %f \n", delta, K_M[0][0], K_M[0][1],
                   K_M[0][2], K_M[1][0], K_M[1][1], K_M[1][2],K_M[2][0],  K_M[2][1], K_M[2][2]);

   Kokkos::printf(" K1 %f K2 %f K3 %f \n", K[0],K[1],K[2]);

*/
}
KOKKOS_INLINE_FUNCTION
void CalculateK(const double xp[3],const double xq[3], double K[9])
{
   double r = pow( pow( xp[0] - xq[0], 2.0) + pow( xp[1] - xq[1], 2.0) + pow( xp[2] - xq[2], 2.0), 0.5);
   double K_M[3][3] = { { 0, (xp[2] - xq[2]), -1*(xp[1] - xq[1])},
                        { -1*(xp[2] - xq[2]), 0, (xp[0] - xq[0])},
                        { (xp[1] - xq[1]), -1*(xp[0] - xq[0]), 0} };

      for(int d0 = 0; d0 < 3; d0++){
         for(int d1 = 0; d1 < 3; d1++){

            if(r < 1.0e-9){
               K[d0*3 + d1] = 0.0;
            }
            else{
               K[d0*3 + d1] = K_M[d0][d1] * 1.0/(4.0*Kokkos::numbers::pi*pow(r, 3.0) );
               // printf("K[%d] = %f \n", d0*3+d1, K[d0*3+d1]);
            }
	          
        }
      }

}
KOKKOS_INLINE_FUNCTION
void Calculate_scalarK(const double xp[3],const double xq[3], double* scal_K)
{
   double r = pow( pow( xp[0] - xq[0], 2.0) + pow( xp[1] - xq[1], 2.0) + pow( xp[2] - xq[2], 2.0), 0.5);
   if(r < 1.0e-9){
      *scal_K = 0.0;
   }
   else{
      *scal_K = 1.0/(4.0*Kokkos::numbers::pi*pow(r, 3.0) );
      // printf("K = %f \n", *scal_K);
   }
}

} // end namespace GREENS FUNCTION
} // end namespace ExaMPM

#endif // EXAMPM_GREENSFUNCTION_HPP

