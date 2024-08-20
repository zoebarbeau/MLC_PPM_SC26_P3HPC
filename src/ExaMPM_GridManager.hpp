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

#ifndef EXAMPM_GRIDMANAGER_HPP
#define EXAMPM_GRIDMANAGER_HPP

#include <ExaMPM_Mesh.hpp>
#include <ExaMPM_ParticleInit.hpp>
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <memory>

namespace ExaMPM
{
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
template <class MemorySpace>
class GridManager
{
  public:
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    using grid_members = Cabana::MemberTypes<int[3]>;
    using grid_list = Cabana::AoSoA<grid_members, MemorySpace>;

    template <class ExecutionSpace>
    GridManager(const ExecutionSpace& exec_space,const int num_D0, const int extent)
             : _num_D0( num_D0),
	       _extent( extent ),
	       _gridp( "A", _num_D0)
    {

        int particle_counter = 0;
        auto index = Cabana::slice<0>( _gridp );
        for ( int i = 1; i < _extent; ++i )
          for ( int j = 1; j < _extent; ++j )
                for ( int k = 1; k < _extent; ++k, ++particle_counter )
                {
                    index( particle_counter, 0 ) = i;
                    index( particle_counter, 1 ) = j;
                    index( particle_counter, 2 ) = k;
                }
    }


    typename grid_list::template member_slice_type<0>
    get(  ) const
    {
        return Cabana::slice<0>( _gridp, "index" );
    }


  private:
    int _num_D0;
    int _extent;
    grid_list _gridp;
};

//---------------------------------------------------------------------------//

} // end namespace ExaMPM

#endif // end EXAMPM_PROBLEMMANAGER2_HPP
