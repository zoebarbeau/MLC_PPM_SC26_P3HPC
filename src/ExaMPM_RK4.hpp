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

#ifndef EXAMPM_RK4_HPP
#define EXAMPM_RK4_HPP

#include <ExaMPM_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
namespace ExaMPM
{
namespace Time
{

template <class ProblemManagerType, class ExecutionSpace>
void RK4( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
                        const double center, const double cell_size, const double hp, const int corr_radius, const double time, const double dt)
{
    double sixth = 1, third=1, half = 1;
    sixth/=6; third/=3; half/=2;

    auto u_p = pm.get( Location::Particle(), Field::Velocity() );
    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );
    auto advect_vorticity = pm.get( Location::Particle(), Field::Vorticity_Advect() );

      Kokkos::parallel_for(
        "Corrections",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int p ) {

             double tmp1[3] = { u_p(p,0), u_p(p,1), u_p(p,2) };
             double tmp2[3] = {0.0,0.0,0.0};

             





        });
    //Time Advance dxp/dt = up
    m_delta.init(a_state);
    m_k.init(a_state);                  // init must allocate stroage, and initialize it to zero.
    m_f(m_k, a_time, a_dt, a_state);    // compute k1
    m_delta.increment(sixth, m_k);
    m_k*=half;
    m_f(m_k, a_time+half*a_dt, a_dt, a_state);  // compute k2
    m_delta.increment(third, m_k);
    m_k*=half;
    m_f(m_k, a_time+half*a_dt, a_dt, a_state);  // conpute k3
    m_delta.increment(third, m_k);
    m_f(m_k, a_time+a_dt, a_dt, a_state); // compute k4
    m_delta.increment(sixth, m_k);
    a_state.increment(m_delta);
}


} // end namespace LocalCorrection
} // end namespace ExaMPM

#endif // EXAMPM_LocalCorrection_HPP
