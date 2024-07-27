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

#ifndef EXAMPM_VINTERPOLATION_HPP
#define EXAMPM_VINTERPOLATION_HPP

#include <ExaMPM_Types.hpp>

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cmath>
#include <type_traits>

namespace ExaMPM
{
namespace VPIC
{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Interpolate particle momentum to the nodes. (Second and Third order
// splines)
template <class SplineDataType, class MomentumView>
KOKKOS_INLINE_FUNCTION void
p2g( const typename MomentumView::original_value_type vort_p[3],
     const SplineDataType& sd, const MomentumView& node_vort,
     typename std::enable_if<
         ( Cabana::Grid::isNode<typename SplineDataType::entity_type>::value &&
           ( SplineDataType::order == 2 || SplineDataType::order == 3 ) ),
         void*>::type = 0 )
{
    static_assert( Cabana::Grid::P2G::is_scatter_view<MomentumView>::value,
                   "P2G requires a Kokkos::ScatterView" ); 
    auto vort_access = node_vort.access();

    using value_type = typename MomentumView::original_value_type;

    value_type vort_g[3];
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {


                // Interpolate particle momentum to the entity.
                for ( int d = 0; d < 3; ++d ){
                    vort_access( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                     sd.s[Dim::K][k], d ) +=
                        sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k] * vort_p[d];
		}
            }

}

//---------------------------------------------------------------------------//
// Interpolate particle momentum to the nodes. (First order splines)
template <class SplineDataType, class MomentumView>
KOKKOS_INLINE_FUNCTION void
p2g( const typename MomentumView::original_value_type vort_p[3],
     const SplineDataType& sd, const MomentumView& node_vort,
     typename std::enable_if<
         ( Cabana::Grid::isNode<typename SplineDataType::entity_type>::value &&
           ( SplineDataType::order == 1 ) ),
         void*>::type = 0 )
{
  static_assert( Cabana::Grid::P2G::is_scatter_view<MomentumView>::value,
                   "P2G requires a Kokkos::ScatterView" ); 
       	  auto vort_access = node_vort.access();

    using value_type = typename MomentumView::original_value_type;

    // Project momentum.
    value_type vort_g[3];
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
            
	          // Interpolate particle momentum to the entity.
                for ( int d = 0; d < 3; ++d )
                    vort_access( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                     sd.s[Dim::K][k], d ) +=
                        sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k] * vort_g[d];
	    	    
	    }
}

//---------------------------------------------------------------------------//
// Interpolate grid node velocity to the particle.
template <class SplineDataType, class VelocityView>
KOKKOS_INLINE_FUNCTION void
g2p( const VelocityView& node_vorticity, const SplineDataType& sd,
     typename VelocityView::value_type vort_p[3],
     typename std::enable_if<
         Cabana::Grid::isNode<typename SplineDataType::entity_type>::value,
         void*>::type = 0 )
{
    using value_type = typename VelocityView::value_type;

    for ( int d = 0; d < 3; ++d )
        vort_p[d] = 0.0;


    value_type w_ip;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Projection weight.
                w_ip = sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];

                // Update velocity.
                for ( int d = 0; d < 3; ++d )
                    vort_p[d] +=
                        w_ip * node_vorticity( sd.s[Dim::I][i], sd.s[Dim::J][j],sd.s[Dim::K][k], d );

            }
}

//---------------------------------------------------------------------------//

} // end namespace VPIC
} // end namespace ExaMPM

#endif // end EXAMPM_VINTERPOLATION_HPP
