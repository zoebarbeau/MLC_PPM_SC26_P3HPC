// counters.hpp
#pragma once
#include <Kokkos_Core.hpp>
#include <cstdint>

struct PerfCounters
{
    // Host-visible names (index mapping)
    enum IDX : int {
      TOTAL_PARTICLES = 0,
      CELLS_WITH_PARTICLES,
      PARTICLES_INTERACTED,     // real particles inspected/used
      PARTICLES_TESTED,         // real particles tested (including those rejected)
      GRID_POINTS_TOUCHED,      // grid nodes written/read
      INTERACTIONS_NBODY,       // neighbor interactions processed (p,q pairs)
      CELLS_TOUCHED,            // cells visited in stencil loops
      FLOPS,                    // integer count of floating point operations
      PARTICLES_PER_CELL_TOTAL, // running sum for average
      LAP_GRID_POINTS,
      INTERP_GRID_POINTS,
      NCOUNTERS
    };

    using view_t = Kokkos::View<uint64_t*>;

    view_t view;

    PerfCounters(const std::string& name="perf_counters")
    {
      view = view_t(name, NCOUNTERS);
      // initialize
      Kokkos::deep_copy(view, (uint64_t)0);
      // initialize min to large value
    }

    // convenience host accessor (copy to host before reading)
    std::array<uint64_t, NCOUNTERS> copy_to_host() const {
      std::array<uint64_t, NCOUNTERS> out;
      auto host = Kokkos::create_mirror_view(view);
      Kokkos::deep_copy(host, view);
      for(int i=0;i<NCOUNTERS;i++) out[i] = host(i);
      return out;
    }

    // atomically add
    KOKKOS_INLINE_FUNCTION
    static void add(view_t v, int idx, uint64_t val){
      Kokkos::atomic_fetch_add(&v(idx), val);
    }

};
