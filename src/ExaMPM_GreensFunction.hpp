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
KOKKOS_INLINE_FUNCTION void CalculateK(const double xp[3],const double xq[3], const double up[3], double K[3])
{
   double r = pow( pow( xp[0] - xq[0], 2.0) + pow( xp[1] - xq[1], 2.0) + pow( xp[2] - xq[2], 2.0), 0.5);
   double K_M[3][3] = { { 0, (xp[2] - xq[2]), -1*(xp[1] - xq[1])},
                        { -1*(xp[2] - xq[2]), 0, (xp[0] - xq[0])},
                        { (xp[1] - xq[1]), -1*(xp[0] - xq[0]), 0} };

   if( r < pow(10.0,-9.0) )
   {
         K[0] = 0; K[1] = 0; K[2] = 0;
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
} // end namespace GREENS FUNCTION
} // end namespace ExaMPM

#endif // EXAMPM_GREENSFUNCTION_HPP

