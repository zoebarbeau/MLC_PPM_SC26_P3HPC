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
/*KOKKOS_INLINE_FUNCTION
__attribute__((always_inline))
void Calculate_qK(const double xp[3],const double xq[3], const double up[3], double K[3],const double h, const int corr_radius)
{
   double r = pow( pow( xp[0] - xq[0], 2) + pow( xp[1] - xq[1], 2) + pow( xp[2] - xq[2], 2), 0.5);
   double K_M[3][3] = { { 0, (xp[2] - xq[2]), -1*(xp[1] - xq[1])},
                        { -1*(xp[2] - xq[2]), 0, (xp[0] - xq[0])},
                        { (xp[1] - xq[1]), -1*(xp[0] - xq[0]), 0} };
   double delta = pow(2,0.5)*h/2;
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


}
*/
KOKKOS_INLINE_FUNCTION
void Calculate_qK( const double xp[3], const double xq[3],
                   const double up[3], double K[3],
                   const double h, const int corr_radius )
{

	
    double dx = xp[0] - xq[0];
    double dy = xp[1] - xq[1];
    double dz = xp[2] - xq[2];

    double r = sqrt(dx*dx + dy*dy + dz*dz);

    double K_M[3][3] =
    {
        { 0.0,  dz,  -dy },
        { -dz, 0.0,  dx },
        { dy,  -dx, 0.0 }
    };
    
    double delta = 0.5*h; 
    const double inv_4pi = 1.0/(4.0*Kokkos::numbers::pi);
    const double r3 = r*r*r;
    if ( r < delta && r > 1e-12 )
    {

	const double delta2 = delta*delta;
        const double delta3_inv = 1.0/(delta*delta*delta);
        const double near_field_constant = 0.125 * inv_4pi * delta3_inv;
        const double r2 = r*r;	
        double c = (-12.0*r2/delta2 + 20.0)*near_field_constant;

        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++)
                K_M[i][j] *= c;
    }
    else if ( r >= delta )
    {
        double c = inv_4pi/r3; //1.0 / (4.0*Kokkos::numbers::pi*r*r*r);
        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++)
                K_M[i][j] *= c;
    }

    DenseLinearAlgebra::matVecMultiply( K_M, up, K );
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

