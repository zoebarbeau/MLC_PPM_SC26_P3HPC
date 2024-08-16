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

#ifndef EXAMPM_PROBLEMMANAGER2_HPP
#define EXAMPM_PROBLEMMANAGER2_HPP

#include <ExaMPM_Mesh.hpp>
#include <ExaMPM_ParticleInit.hpp>
#include <ExaMPM_GreensFunction.hpp>
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <memory>

namespace ExaMPM
{
//---------------------------------------------------------------------------//
// Field locations
namespace Location
{
struct Cell
{
};
struct Node
{
};
struct Particle
{
};
} // end namespace Location

//---------------------------------------------------------------------------//
// Fields.
namespace Field
{
struct Velocity
{
};
struct Position
{
};
struct Vorticity
{
};
struct Velocity_Correction
{
};
struct Velocity_Nbody
{
};
struct Vortx
{
};
} // end namespace Field.

//---------------------------------------------------------------------------//
template <class MemorySpace>
class ProblemManager
{
  public:
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    using particle_members =
        Cabana::MemberTypes<double[3], double[3], double[3],double,double[3], double[3]>;
    using particle_list = Cabana::AoSoA<particle_members, MemorySpace>;
    using particle_type = typename particle_list::tuple_type;

    using particle_grid_members =
        Cabana::MemberTypes<double[3], double[3], double[3],double,double[3], double[3],int[3], int>;
    using particle_grid_list = Cabana::AoSoA<particle_grid_members, MemorySpace>;
    using particle_grid_type = typename particle_grid_list::tuple_type;

    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node,
                            Cabana::Grid::UniformMesh<double>, MemorySpace>;

    using cell_array =
        Cabana::Grid::Array<double, Cabana::Grid::Cell,
                            Cabana::Grid::UniformMesh<double>, MemorySpace>;

    using halo = Cabana::Grid::Halo<MemorySpace>;
    using mesh_type = Mesh<MemorySpace>;

    template <class InitFunc, class ExecutionSpace>
    ProblemManager( const ExecutionSpace& exec_space,
                    const std::shared_ptr<mesh_type>& mesh,
                    const InitFunc& create_functor,
                    const int particles_per_cell, const double cell_size)
        : _mesh( mesh )
        , _cell_size( cell_size )
        , _particles( "particles" )
    {
        initializeParticles( exec_space, *( _mesh->localGrid() ),
                             particles_per_cell, create_functor, _particles );
        auto node_vector_layout = Cabana::Grid::createArrayLayout(
            _mesh->localGrid(), 3, Cabana::Grid::Node() );
        auto node_scalar_layout = Cabana::Grid::createArrayLayout(
            _mesh->localGrid(), 1, Cabana::Grid::Node() );
        auto cell_scalar_layout = Cabana::Grid::createArrayLayout(
            _mesh->localGrid(), 1, Cabana::Grid::Cell() );

        _velocity = Cabana::Grid::createArray<double, MemorySpace>(
            "velocity", node_vector_layout );
        _vorticity = Cabana::Grid::createArray<double, MemorySpace>(
            "vorticity", node_vector_layout );
        _vortx = Cabana::Grid::createArray<double, MemorySpace>(
            "vortx", node_scalar_layout );

	_velocity_correction = Cabana::Grid::createArray<double, MemorySpace>(
            "velocity_correction", node_vector_layout );

	_velocity_nbody = Cabana::Grid::createArray<double, MemorySpace>(
            "velocity_nbody", node_vector_layout );

        _node_scatter_halo =
           Cabana::Grid::createHalo( Cabana::Grid::NodeHaloPattern<3>(), -1,
                                      *_vorticity, *_velocity, *_vortx );
    //
        _node_gather_halo = Cabana::Grid::createHalo(
            Cabana::Grid::NodeHaloPattern<3>(), -1, *_velocity,*_vorticity, *_vortx );
    }

    std::size_t numParticle() const { return _particles.size(); }

    const std::shared_ptr<mesh_type>& mesh() const { return _mesh; }

    typename particle_list::template member_slice_type<1>
    get( Location::Particle, Field::Velocity ) const
    {
        return Cabana::slice<1>( _particles, "velocity" );
    }

    typename particle_list::template member_slice_type<2>
    get( Location::Particle, Field::Position ) const
    {
        return Cabana::slice<2>( _particles, "position" );
    }

    typename particle_list::template member_slice_type<0>
    get( Location::Particle, Field::Vorticity ) const
    {
        return Cabana::slice<0>( _particles, "vorticity" );
    }


    typename particle_list::template member_slice_type<3>
    get( Location::Particle, Field::Vortx ) const
    {
        return Cabana::slice<3>( _particles, "vortx" );
    }

    typename particle_list::template member_slice_type<4>
    get( Location::Particle, Field::Velocity_Correction ) const
    {
        return Cabana::slice<4>( _particles, "velocity_correction" );
    }

    typename particle_list::template member_slice_type<5>
    get( Location::Particle, Field::Velocity_Nbody ) const
    {
        return Cabana::slice<5>( _particles, "velocity_nbody" );
    }

    typename node_array::view_type get( Location::Node, Field::Vortx ) const
    {
        return _vortx->view();
    }


    typename node_array::view_type get( Location::Node, Field::Vorticity ) const
    {
        return _vorticity->view();
    }


    typename node_array::view_type get( Location::Node, Field::Velocity ) const
    {
        return _velocity->view();
    }

    typename node_array::view_type get( Location::Node, Field::Velocity_Nbody ) const
    {
        return _velocity_nbody->view();
    }


    typename node_array::view_type get( Location::Node, Field::Velocity_Correction ) const
    {
        return _velocity_correction->view();
    }


    // WHAT IS SCATTER FOR
 /*   void scatter( Location::Cell ) const
    {
        _cell_halo->scatter( execution_space(),
                             Cabana::Grid::ScatterReduce::Sum(), *_vorticity );
    }
*/
    //Changed Scatter operation to replace instead of sum
    void scatter( Location::Node ) const
    {
        _node_scatter_halo->scatter( execution_space(),Cabana::Grid::ScatterReduce::Replace(),
                                     *_vorticity, *_velocity, *_vortx, *_velocity_nbody, *_velocity_correction );
    }

    void gather( Location::Node ) const
    {
        _node_gather_halo->gather( execution_space(), *_velocity, *_vorticity, *_vortx, *_velocity_nbody, *_velocity_correction );
    }

    void communicateParticles( const int minimum_halo_width )
    {
        auto positions = get( Location::Particle(), Field::Position() );
        Cabana::Grid::particleGridMigrate( *( _mesh->localGrid() ), positions,
                                           _particles, minimum_halo_width );
    }


  private:
    std::shared_ptr<mesh_type> _mesh;
    double _amp, _cell_size;
    particle_list _particles;
    particle_grid_list _grid_particles;
    std::shared_ptr<node_array> _vorticity;
    std::shared_ptr<node_array> _velocity;
    std::shared_ptr<node_array> _velocity_correction;
    std::shared_ptr<node_array> _velocity_nbody;
    std::shared_ptr<node_array> _vortx;
    std::shared_ptr<halo> _node_scatter_halo;
    std::shared_ptr<halo> _node_gather_halo;
    std::shared_ptr<halo> _node_correction_halo;
    std::shared_ptr<halo> _cell_halo;
};

//---------------------------------------------------------------------------//

} // end namespace ExaMPM

#endif // end EXAMPM_PROBLEMMANAGER2_HPP
