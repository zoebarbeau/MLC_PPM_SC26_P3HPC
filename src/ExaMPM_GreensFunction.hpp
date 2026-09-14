/* This file is written by the authors specifically for MLC */
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
void Calculate_qK_MatVec_Fused(
    const double xp[3], const double xq[3],
    const double up[3], double K[3],
    const double h )
{
    // Pre-computed constants
    const double inv_4pi = 0.07957747154594767;  // 1/(4π)
    const double delta = 0.5 * h;
    const double delta2 = delta * delta;

    // Compute distance components
    const double dx = xp[0] - xq[0];
    const double dy = xp[1] - xq[1];
    const double dz = xp[2] - xq[2];
    const double r2 = dx*dx + dy*dy + dz*dz;

    // FIXED: Near-field FIRST (matches original logic)
    if ( r2 < delta2 && r2 > 1e-24 )  // ← FIX #1: Changed >= to <
    {
        const double r = Kokkos::sqrt(r2);
        const double delta3_inv = 1.0 / (delta * delta2);
        const double near_const = 0.125 * inv_4pi * delta3_inv;
        const double c = (-12.0 * r2 / delta2 + 20.0) * near_const;

        K[0] = c * (dz*up[1] - dy*up[2]);
        K[1] = c * (dx*up[2] - dz*up[0]);
        K[2] = c * (dy*up[0] - dx*up[1]);
    }
    else if ( r2 >= delta2 )  // ← FIX #2: Far-field SECOND
    {
        const double r_inv = 1.0 / Kokkos::sqrt(r2);
        const double r3_inv = r_inv * r_inv * r_inv;
        const double c = inv_4pi * r3_inv;

        K[0] = c * (dz*up[1] - dy*up[2]);
        K[1] = c * (dx*up[2] - dz*up[0]);
        K[2] = c * (dy*up[0] - dx*up[1]);
    }
    else  // Singularity: 0 <= r2 <= 1e-24
    {
        K[0] = 0.0;
        K[1] = 0.0;
        K[2] = 0.0;
    }
}

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

