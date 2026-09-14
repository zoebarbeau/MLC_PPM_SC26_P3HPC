/* This file is written by the authors specifically for MLC */
#ifndef MLC_GRIDMANAGER_HPP
#define MLC_GRIDMANAGER_HPP

#include <MLC_Mesh.hpp>
#include <MLC_ParticleInit.hpp>
#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <memory>

namespace MLC
{


namespace Grid
{
struct Index
{
};
struct Position
{
};
struct Id
{
};
} 

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
template <class MemorySpace>
class GridManager
{
  public:
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    using grid_members = Cabana::MemberTypes<int[3], double[3], int>;
    using grid_list = Cabana::AoSoA<grid_members, MemorySpace>;

    int _num_D0, _num_p;
    double _h, _center;
    int _extent;
    grid_list _gridp;

    template <class ExecutionSpace, class ViewType, class LocalGridType>
    GridManager(const ExecutionSpace& exec_space, const LocalGridType& cgrid,
		       const ViewType& x, const int num_p
		      ,const int num_D0, const int extent, const double h, const double center)
               : _num_D0( num_D0),
	       _num_p( num_p ),
	       _extent( extent ),
	       _h( h),
               _center( center ),
	       _gridp( "A", _num_D0+_num_p)
    {

    }

    std::size_t numParticle() const { return _gridp.size(); }

    //Functions to access index, position, and particle ID
    typename grid_list::template member_slice_type<0>
    get( Grid::Index  ) const
    {
        return Cabana::slice<0>( _gridp, "index" );
    }

    typename grid_list::template member_slice_type<1>
    get( Grid::Position  ) const
    {
        return Cabana::slice<1>( _gridp, "position" );
    }

    typename grid_list::template member_slice_type<2>
    get( Grid::Id  ) const
    {
        return Cabana::slice<2>( _gridp, "ID" );
    }

};

//---------------------------------------------------------------------------//

} // end namespace MLC

#endif // end MLC_PROBLEMMANAGER2_HPP
