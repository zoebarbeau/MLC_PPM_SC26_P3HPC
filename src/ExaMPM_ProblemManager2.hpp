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
struct F
{
};
struct Vorticity_hp
{
};
struct Fx
{
};
struct velx
{
};
struct Velocity_Corr
{
};
struct Vorticity_Advect
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
        Cabana::MemberTypes<double[3], double[3], double[3], double[3]>;
    using particle_list = Cabana::AoSoA<particle_members, MemorySpace>;
    using particle_type = typename particle_list::tuple_type;

    using node_array =
        Cabana::Grid::Array<double, Cabana::Grid::Node,
                            Cabana::Grid::UniformMesh<double>, MemorySpace>;

    using cell_array =
        Cabana::Grid::Array<double, Cabana::Grid::Cell,
                            Cabana::Grid::UniformMesh<double>, MemorySpace>;

    using halo = Cabana::Grid::Halo<MemorySpace>;
    using mesh_type = Mesh<MemorySpace>;

    std::shared_ptr<mesh_type> _pmesh;

    template <class InitFunc, class ExecutionSpace>
    ProblemManager( const ExecutionSpace& exec_space,
                    const std::shared_ptr<mesh_type>& mesh,
		    const std::shared_ptr<mesh_type>& pmesh,
                    const InitFunc& create_functor,
                    const int particles_per_cell, const double cell_size, 
		    const double center, const double hp, const double extent, const double extentp)
        : _mesh( mesh )
	, _pmesh( pmesh )  
        , _cell_size( cell_size )
        , _particles( "particles" )
	, _ppc( particles_per_cell)
        , _center(center)
	, _hp(hp)
        , _extent(extent)
        , _extentp(extentp)
    {
        initializeParticles( exec_space, *( _pmesh->localGrid() ),
                             particles_per_cell, create_functor, _particles,
		             _center, _hp,_extentp);

	std::cout << " hp particle " << hp << std::endl;
	std::cout << " _center particle " << _center << std::endl;

	// Grid Layout
        auto node_vector_layout = Cabana::Grid::createArrayLayout(
            _mesh->localGrid(), 3, Cabana::Grid::Node() );
        auto node_scalar_layout = Cabana::Grid::createArrayLayout(
            _mesh->localGrid(), 1, Cabana::Grid::Node() );
        auto cell_scalar_layout = Cabana::Grid::createArrayLayout(
            _mesh->localGrid(), 1, Cabana::Grid::Cell() );

	auto pnode_vector_layout = Cabana::Grid::createArrayLayout(
            _pmesh->localGrid(), 3, Cabana::Grid::Node() );


        _velocity = Cabana::Grid::createArray<double, MemorySpace>(
            "velocity", node_vector_layout );
        _vorticity = Cabana::Grid::createArray<double, MemorySpace>(
            "vorticity", node_vector_layout );
	_F = Cabana::Grid::createArray<double, MemorySpace>(
            "F", node_vector_layout );

        _vorticity_hp = Cabana::Grid::createArray<double, MemorySpace>(
            "vorticity_hp", pnode_vector_layout );

	_velx = Cabana::Grid::createArray<double, MemorySpace>(
            "velx", node_scalar_layout );

        _Fx = Cabana::Grid::createArray<double, MemorySpace>(
            "Fx", node_scalar_layout );

        _velocity_corr = Cabana::Grid::createArray<double, MemorySpace>(
            "velocity_corr", node_vector_layout );

        _node_scatter_halo =
           Cabana::Grid::createHalo( Cabana::Grid::NodeHaloPattern<3>(), -1,
                                      *_vorticity, *_velocity );
        _node_gather_halo = Cabana::Grid::createHalo(
            Cabana::Grid::NodeHaloPattern<3>(), -1, *_velocity,*_vorticity );

        std::array<std::string, 4> names;
        names[0] = "F"; names[1] = "lap_u";
        names[2] = "pre_corr_v"; names[3] = "post_corr_v";
        // create an array and store the name of each variable:

	// Particle Deposition Grid Layout
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
    get( Location::Particle, Field::Vorticity_Advect ) const
    {
        return Cabana::slice<3>( _particles, "vorticity_advect" );
    }


    typename node_array::view_type get( Location::Node, Field::Vorticity ) const
    {
        return _vorticity->view();
    }


    typename node_array::view_type get( Location::Node, Field::Velocity ) const
    {
        return _velocity->view();
    }

    typename node_array::view_type get( Location::Node, Field::Velocity_Corr ) const
    {
        return _velocity_corr->view();
    }


    typename node_array::view_type get( Location::Node, Field::F ) const
    {
        return _F->view();
    }

    typename node_array::view_type get( Location::Node, Field::Vorticity_hp ) const
    {
        return _vorticity_hp->view();
    }

    typename node_array::view_type get( Location::Node, Field::velx ) const
    {
        return _velx->view();
    }

    typename node_array::view_type get( Location::Node, Field::Fx ) const
    {
        return _Fx->view();
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
                                     *_vorticity, *_velocity );
    }

    void gather( Location::Node ) const
    {
        _node_gather_halo->gather( execution_space(), *_velocity, *_vorticity );
    }

    void communicateParticles( const int minimum_halo_width )
    {
        auto positions = get( Location::Particle(), Field::Position() );
        Cabana::Grid::particleGridMigrate( *( _mesh->localGrid() ), positions,
                                           _particles, minimum_halo_width );
    }

    template <class InitFunc, class ExecutionSpace>
    void Resize_Remap(const ExecutionSpace& exec_space, 
		      const InitFunc& create_functor ) 
    {

       auto vorticity_g = get( Location::Node(), Field::Vorticity_hp());
       _particles.resize( 0 );
    //   _particles.shrinkToFit();
       remapParticles( exec_space, *( _pmesh->localGrid() ),
                             _ppc, create_functor, _vorticity_hp, _particles,
		             _center, _hp, _extentp);


    }	    

    void save_F(std::string run_name, const int timesteps_done, const double time) const
    {   std::stringstream name;
        name << run_name << "_" << _Fx->label();
        const std::string prefix = name.str();
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(prefix,timesteps_done, time, *_Fx);
    }

    void save_v(std::string run_name, const int timesteps_done, const double time) const
    {   std::stringstream name;
        name << run_name << "_" << _velx->label();
        const std::string prefix = name.str();
        Cabana::Grid::Experimental::BovWriter::writeTimeStep(prefix,timesteps_done, time, *_velx);
    }







  private:
    double _amp, _cell_size,_hp, _center, _extent, _extentp;
    int _ppc;
    particle_list _particles;
    std::shared_ptr<node_array> _velx,_Fx;
    std::shared_ptr<node_array> _vorticity, _F;
    std::shared_ptr<node_array> _velocity,_velocity_corr,_vorticity_hp;
    std::shared_ptr<halo> _node_scatter_halo;
    std::shared_ptr<halo> _node_gather_halo;
    std::shared_ptr<halo> _node_correction_halo;
    std::shared_ptr<halo> _cell_halo;
    std::shared_ptr<mesh_type> _mesh;

};

//---------------------------------------------------------------------------//

} // end namespace ExaMPM

#endif // end EXAMPM_PROBLEMMANAGER2_HPP
