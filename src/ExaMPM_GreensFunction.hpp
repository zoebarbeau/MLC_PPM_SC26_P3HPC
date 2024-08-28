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
void CalculateK(const double xp[3],const double xq[3], const double up[3], double K[3])
{
   double r = pow( pow( xp[0] - xq[0], 2.0) + pow( xp[1] - xq[1], 2.0) + pow( xp[2] - xq[2], 2.0), 0.5);
   double K_M[3][3] = { { 0, (xp[2] - xq[2]), -1*(xp[1] - xq[1])},
                        { -1*(xp[2] - xq[2]), 0, (xp[0] - xq[0])},
                        { (xp[1] - xq[1]), -1*(xp[0] - xq[0]), 0} };

   for(int d0 = 0; d0 < 3; d0++){
     for(int d1 = 0; d1 < 3; d1++){

	     
        K_M[d0][d1] *= 1/(4*Kokkos::numbers::pi*pow(r, 3.0) );
     }

   }

   std::cout << " xp = " << xp[0] << " yp = " << xp[1] << " zp = " << xp[2] << std::endl;
   std::cout << " xq = " << xq[0] << " yq = " << xq[1] << " zq = " << xq[2] << std::endl;
   std::cout << " r = " << r << std::endl;
   std::cout << K_M[0][0] << " " << K_M[1][0] << " " 
	     << K_M[0][1] << " " << K_M[1][1] << " "
	     << K_M[0][2] << " " << K_M[1][2] << " "
	     << K_M[2][0] << " " << K_M[2][1] << " "
	     << K_M[2][2] << std::endl;
   DenseLinearAlgebra::matVecMultiply(K_M, up, K);

   if( abs(r) < pow(10,-9.0) )
	 K[0] = 0; K[1] = 0; K[2] = 0;  

}
} // end namespace GREENS FUNCTION
} // end namespace ExaMPM

#endif // EXAMPM_GREENSFUNCTION_HPP

