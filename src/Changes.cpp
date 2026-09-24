template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
void Deposition_TeamPolicy_optimized(
    const ExecutionSpace& exec_space,
    const ProblemManagerType& pm,
    const NeighborListType& Pi_list,
    const NeighborListType& Ci_list,
    const GridManager& gridp,
    const int num_grid,
    const int extent,
    const double center,
    const double h,
    const double hp,
    const int corr_radius)
{
    auto index       = gridp.get(Grid::Index());
    auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
    auto positions   = pm.get(Location::Particle(), Field::Position());
    auto F           = pm.get(Location::Node(), Field::F());

    Kokkos::deep_copy(F, 0.0);

    const int sz = 2 * corr_radius + 1;
    const double h2_inv = 1.0 / (h * h);

    const double w_center = -128.0 / 30.0;
    const double w_corner =  1.0 / 30.0;
    const double w_edge   =  1.0 / 10.0;
    const double w_face   =  7.0 / 15.0;

    using team_policy = Kokkos::TeamPolicy<ExecutionSpace>;
    using member_type = typename team_policy::member_type;

    // One team per grid cell
    team_policy policy(exec_space, num_grid, Kokkos::AUTO());

    Kokkos::parallel_for(
        "Deposition_TeamPolicy",
        policy,
        KOKKOS_LAMBDA(const member_type& team)
        {
            const int i = team.league_rank();

            const int ii = index(i, 0);
            const int jj = index(i, 1);
            const int kk = index(i, 2);

            const auto offset = Pi_list.binOffset(ii, jj, kk);
            const auto size   = Pi_list.binSize(ii, jj, kk);
            if (size == 0) return;

            int imin, imax, jmin, jmax, kmin, kmax;
            Ci_list.getStencilCells(
                Ci_list.getParticleBin(i),
                imin, imax, jmin, jmax, kmin, kmax);

            const int nx = imax - imin;
            const int ny = jmax - jmin;
            const int nz = kmax - kmin;

            // Guard if runtime stencil and corr_radius disagree
            if (nx > sz || ny > sz || nz > sz) return;

            // Only interior nodes can deposit a 27-point Laplacian
            if (nx < 3 || ny < 3 || nz < 3) return;

            // Number of interior stencil nodes
            const int n_interior = (nx - 2) * (ny - 2) * (nz - 2);

            // Parallelize over interior stencil nodes
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, n_interior),
                [&](const int flat)
                {
                    // Map flat -> local interior index (li, lj, lk)
                    const int li = 1 + flat / ((ny - 2) * (nz - 2));
                    const int rem = flat % ((ny - 2) * (nz - 2));
                    const int lj = 1 + rem / (nz - 2);
                    const int lk = 1 + rem % (nz - 2);

                    const int c0i = imin + li;
                    const int c0j = jmin + lj;
                    const int c0k = kmin + lk;

                    // Accumulate Laplacian pieces directly, component by component.
                    double center_val[3] = {0.0, 0.0, 0.0};
                    double u_face[3]     = {0.0, 0.0, 0.0};
                    double u_edge[3]     = {0.0, 0.0, 0.0};
                    double u_corner[3]   = {0.0, 0.0, 0.0};

                    // Loop over 3x3x3 neighborhood around current interior node
                    for (int di = -1; di <= 1; ++di) {
                        const int ci = c0i + di;
                        for (int dj = -1; dj <= 1; ++dj) {
                            const int cj = c0j + dj;
                            for (int dk = -1; dk <= 1; ++dk) {
                                const int ck = c0k + dk;

                                const int m = (di == 0 ? 0 : 1)
                                                    + (dj == 0 ? 0 : 1)
                                                    + (dk == 0 ? 0 : 1);

                                const double xg_arr[3] = {
                                    ci * h - center,
                                    cj * h - center,
                                    ck * h - center
                                };

                                double accum[3] = {0.0, 0.0, 0.0};

                                // Sum Green's-function contributions from all particles in this cell
                                for (std::size_t r = offset; r < offset + size; ++r) {
                                    const auto p = Pi_list.getParticle(r);

                                    const double xp_arr[3] = {
                                        positions(p, 0), positions(p, 1), positions(p, 2)
                                    };
                                    const double vortp_arr[3] = {
                                        vorticity_p(p, 0), vorticity_p(p, 1), vorticity_p(p, 2)
                                    };

                                    double K[3];
                                    GreensFunction::Calculate_qK_MatVec_Fused(
                                        xg_arr, xp_arr, vortp_arr, K, hp);

                                    accum[0] += K[0];
                                    accum[1] += K[1];
                                    accum[2] += K[2];
                                }

                                if (m == 0) {
                                    center_val[0] = accum[0];
                                    center_val[1] = accum[1];
                                    center_val[2] = accum[2];
                                } else if (m == 1) {
                                    u_face[0] += accum[0];
                                    u_face[1] += accum[1];
                                    u_face[2] += accum[2];
                                } else if (m == 2) {
                                    u_edge[0] += accum[0];
                                    u_edge[1] += accum[1];
                                    u_edge[2] += accum[2];
                                } else { // m == 3
                                    u_corner[0] += accum[0];
                                    u_corner[1] += accum[1];
                                    u_corner[2] += accum[2];
                                }
                            }
                        }
                    }

                    for (int d = 0; d < 3; ++d) {
                        const double result =
                            (center_val[d] * w_center +
                             u_corner[d]   * w_corner +
                             u_edge[d]     * w_edge +
                             u_face[d]     * w_face) * h2_inv;

                        Kokkos::atomic_add(&F(c0i, c0j, c0k, d), result);
                    }
                });
        });
}





/*
template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void TestConvolution( const ExecutionSpace& exec_space, const ProblemManagerType& pm, const NeighborListType& Ci_list,
                        const GridManager& gridp, const int num_grid, const int extent, const double center, const double h)
{

//     auto velocity_g = pm.get(Location::Node(), Field::Velocity());
//     auto positions  = pm.get(Location::Particle(), Field::Position());
//     auto F          = pm.get(Location::Node(), Field::F() );
//     auto velx       = pm.get(Location::Node(), Field::velx() );
//     auto Fx         = pm.get(Location::Node(), Field::Fx() );
//     Kokkos::deep_copy(velocity_g, 0.0);
//     //Get relevant interpolation quantities 
//     MLC_Interp::GridData<3> g( h, center);

//     int i,j,k;
//     double delta = 2*h;
//        i = floor((0.875-center)/h); j = i; k = i;
// //     Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {extent+1, extent+1, extent+1}),
// //        KOKKOS_LAMBDA(const int i, const int j, const int k) {

// 	      double xg[3] = { i*h - center, j*h - center, k*h - center };	  
//               double loc = pow( pow(xg[0]-0.875, 2.0) + pow(xg[1]-0.875,2.0) + pow(xg[2]-0.875,2.0) , 0.5);
//               if( loc < pow(10,-6) ){
// 	      //iterate over D0	  
//               for(int i0 = 1; i0 <= extent; i0++)
//                 for(int j0 = 1; j0 <= extent; j0++)
//                    for( int k0 = 1; k0 <= extent; k0++)
//                    {

//                          double x0[3] = { i0*h - center, j0*h - center, k0*h - center };
// 			 double r = pow( pow( x0[0]-xg[0], 2.0) + pow( x0[1]-xg[1], 2.0) + pow( x0[2]-xg[2], 2.0), 0.5 );
//                 //         double r = pow( pow( xg[0], 2.0) + pow( xg[1], 2.0) + pow( xg[2], 2.0), 0.5 );
// 			 if( r < delta )
// 		         {
// 		             for(int d = 0; d < 3; d++)		 
// 			        velocity_g(i,j,k,d) += F(i0,j0,k0,d)*(1.0/(32.0*Kokkos::numbers::pi*delta )*(-3.0*pow(r/delta, 4.0) + 10.0*pow(r/delta,2.0) - 7.0) - 1.0/(4.0*Kokkos::numbers::pi*delta) );
//                          }else{

//                              for(int d = 0; d < 3; d++)
// 			        velocity_g(i,j,k,d) -= F(i0,j0,k0,d)*1.0/(4.0*Kokkos::numbers::pi*r);
//                                 velx(i,j,k,0) = velocity_g(i,j,k,0);  
// 			       if ( std::abs( F(i0,j0,k0,0)*1.0/(4.0*Kokkos::numbers::pi*r) ) > 0 
// 			            || std::abs( F(i0,j0,k0,1)*1.0/(4.0*Kokkos::numbers::pi*r) ) > 0
// 				    || std::abs( F(i0,j0,k0,2)*1.0/(4.0*Kokkos::numbers::pi*r) ) > 0  ){

//  //                                   Kokkos::printf(" r %f x %f y %f z %f F %f i %d j %d k %d \n", r, x0[0], x0[1],x0[2],F(i0,j0,k0,0),i0,j0,k0);
//                                 } 
// 			 }
 
//                     }

//                     double r = pow( pow( xg[0]-0.5-0.15*h, 2.0) + pow( xg[1]-0.5-0.15*h, 2.0) + pow( xg[2]-0.5-0.15*h, 2.0), 0.5 );
//                     Kokkos::printf( " x %f y %f z %f velx %f exact %f \n", xg[0],xg[1],xg[2],velocity_g(i,j,k,0),(xg[2] - 0.5-0.15*h) / (4*Kokkos::numbers::pi*pow(r, 3.0) ) );

//                 }

// //         });

//              pm.save_v( "Convolution_V",1,0);
//              pm.save_F( "Laplacian_V",1,0);

}
*/

 // (3) Full optimized kernel
template <class ExecutionSpace, class NeighborListType, class VortSlice,
          class PosSlice, class USlice, class AdvectVortSlice>
void Interaction_NBody_Split(
    const ExecutionSpace& exec_space,
    PosSlice x_p,
    USlice u_p,
    VortSlice vort_p,
    AdvectVortSlice advect_vorticity,
    NeighborListType& neigh_list,
    const double hp,
    const int numP)
{
    const double inv_4pi    = 0.07957747154594767;
    const double delta      = 0.5 * hp;
    const double delta2     = delta * delta;
    const double delta3inv  = 1.0 / (delta * delta2);
    const double near_const = 0.125 * inv_4pi * delta3inv;
    const double hp_inv     = 1.0 / hp;
    const double half_hp    = 0.5 * hp;

    auto eval_biot_savart = KOKKOS_LAMBDA(
        const double dx, const double dy, const double dz,
        const double uq0, const double uq1, const double uq2,
        const double delta2_in,
        const double inv_4pi_in,
        const double near_const_in,
        double& out0, double& out1, double& out2)
    {
        const double r2 = dx * dx + dy * dy + dz * dz;

        if (r2 >= delta2_in)
        {
            const double rinv  = 1.0 / Kokkos::sqrt(r2);
            const double r3inv = rinv * rinv * rinv;
            const double c     = inv_4pi_in * r3inv;

            out0 = c * (dz * uq1 - dy * uq2);
            out1 = c * (dx * uq2 - dz * uq0);
            out2 = c * (dy * uq0 - dx * uq1);
        }
        else if (r2 > 1e-24)
        {
            const double c = (-12.0 * r2 / delta2_in + 20.0) * near_const_in;

            out0 = c * (dz * uq1 - dy * uq2);
            out1 = c * (dx * uq2 - dz * uq0);
            out2 = c * (dy * uq0 - dx * uq1);
        }
        else
        {
            out0 = 0.0;
            out1 = 0.0;
            out2 = 0.0;
        }
    };

    // ============================================================
    // KERNEL 1: base evaluation only -> updates u_p
    // ============================================================
    auto interaction_u = KOKKOS_LAMBDA(const int p, const int q)
    {
        const double xp0 = x_p(p, 0);
        const double xp1 = x_p(p, 1);
        const double xp2 = x_p(p, 2);

        const double xq0 = x_p(q, 0);
        const double xq1 = x_p(q, 1);
        const double xq2 = x_p(q, 2);

        const double uq0 = vort_p(q, 0);
        const double uq1 = vort_p(q, 1);
        const double uq2 = vort_p(q, 2);

        double b0, b1, b2;
        eval_biot_savart(
            xp0 - xq0, xp1 - xq1, xp2 - xq2,
            uq0, uq1, uq2,
            delta2, inv_4pi, near_const,
            b0, b1, b2);

        u_p(p, 0) += b0;
        u_p(p, 1) += b1;
        u_p(p, 2) += b2;
    };

    Cabana::neighbor_parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, numP),
        interaction_u,
        neigh_list,
        Cabana::FirstNeighborsTag(),
        Cabana::SerialOpTag(),
        "Interaction_NBody_u_only");

    // Optional while profiling/debugging:
    // Kokkos::fence();

    // ============================================================
    // KERNEL 2: plus/minus evaluation only -> updates advect_vorticity
    // ============================================================
    auto interaction_adv = KOKKOS_LAMBDA(const int p, const int q)
    {
        const double xp0 = x_p(p, 0);
        const double xp1 = x_p(p, 1);
        const double xp2 = x_p(p, 2);

        const double vp0 = vort_p(p, 0);
        const double vp1 = vort_p(p, 1);
        const double vp2 = vort_p(p, 2);

        const double xpp0 = xp0 + half_hp * vp0;
        const double xpp1 = xp1 + half_hp * vp1;
        const double xpp2 = xp2 + half_hp * vp2;

        const double xpm0 = xp0 - half_hp * vp0;
        const double xpm1 = xp1 - half_hp * vp1;
        const double xpm2 = xp2 - half_hp * vp2;

        const double xq0 = x_p(q, 0);
        const double xq1 = x_p(q, 1);
        const double xq2 = x_p(q, 2);

        const double uq0 = vort_p(q, 0);
        const double uq1 = vort_p(q, 1);
        const double uq2 = vort_p(q, 2);

        double ap0, ap1, ap2;
        eval_biot_savart(
            xpp0 - xq0, xpp1 - xq1, xpp2 - xq2,
            uq0, uq1, uq2,
            delta2, inv_4pi, near_const,
            ap0, ap1, ap2);

        double am0, am1, am2;
        eval_biot_savart(
            xpm0 - xq0, xpm1 - xq1, xpm2 - xq2,
            uq0, uq1, uq2,
            delta2, inv_4pi, near_const,
            am0, am1, am2);

        advect_vorticity(p, 0) += (ap0 - am0) * hp_inv;
        advect_vorticity(p, 1) += (ap1 - am1) * hp_inv;
        advect_vorticity(p, 2) += (ap2 - am2) * hp_inv;
    };

    Cabana::neighbor_parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, numP),
        interaction_adv,
        neigh_list,
        Cabana::FirstNeighborsTag(),
        Cabana::SerialOpTag(),
        "Interaction_NBody_adv_only");
}


template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
void Corrections_Split(
    const ExecutionSpace& exec_space,
    const ProblemManagerType& pm,
    const NeighborListType& Ci_list,
    const NeighborListType& Pi_list,
    const NeighborListType& Neigh_list,
    const GridManager& gridp,
    const int num_grid,
    const int extent,
    const double center,
    const double h,
    const double hp,
    const double corr_radius)
{
    using MemSpace = typename ExecutionSpace::memory_space;

    auto index         = gridp.get(Grid::Index());
    auto vorticity_p   = pm.get(Location::Particle(), Field::Vorticity());
    auto velocity_p    = pm.get(Location::Particle(), Field::Velocity());
    auto velocity_corr = pm.get(Location::Node(), Field::Velocity_Corr());
    auto velocity_g    = pm.get(Location::Node(), Field::Velocity());
    auto positions     = pm.get(Location::Particle(), Field::Position());
    auto advect_vort   = pm.get(Location::Particle(), Field::Vorticity_Advect());

    Kokkos::deep_copy(velocity_corr, 0.0);

    MLC_Interp::GridData<3> g(h, center);

    const double inv_hp  = 1.0 / hp;
    const double half_hp = 0.5 * hp;
    constexpr int stencil = 3;

    // Temporary corrected stencil per grid cell:
    // [grid cell][si][sj][sk][component]
    Kokkos::View<double*****, MemSpace> corr_stencil(
        "corr_stencil", num_grid, stencil, stencil, stencil, 3);

    // ============================================================
    // KERNEL 1: Build corrected local stencil for each grid cell
    // ============================================================
    Kokkos::parallel_for(
        "Corrections_BuildStencil",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_grid),
        KOKKOS_LAMBDA(const int i)
        {
            const int ii = index(i, 0);
            const int jj = index(i, 1);
            const int kk = index(i, 2);

            const auto Pi_offset = Pi_list.binOffset(ii, jj, kk);
            const auto Pi_size   = Pi_list.binSize(ii, jj, kk);

            // No particles in this cell => nothing to interpolate later
            if (Pi_size == 0) return;

            int imin, imax, jmin, jmax, kmin, kmax;
            Ci_list.getStencilCells(Ci_list.getParticleBin(i),
                                    imin, imax, jmin, jmax, kmin, kmax);

            // Precompute stencil node coordinates once
            double gx[stencil], gy[stencil], gz[stencil];
            for (int s = 0; s < stencil; ++s) {
                gx[s] = (ii - 1 + s) * h - center;
                gy[s] = (jj - 1 + s) * h - center;
                gz[s] = (kk - 1 + s) * h - center;
            }

            // Optional short-circuit: check whether any correction particles exist
            bool has_corr_particles = false;
            for (int pi = imin; pi < imax && !has_corr_particles; ++pi) {
                for (int pj = jmin; pj < jmax && !has_corr_particles; ++pj) {
                    for (int pk = kmin; pk < kmax && !has_corr_particles; ++pk) {
                        if (Neigh_list.binSize(pi, pj, pk) > 0)
                            has_corr_particles = true;
                    }
                }
            }

            // Start from base grid velocity
            for (int si = 0; si < stencil; ++si) {
                const int i_idx = ii - 1 + si;
                for (int sj = 0; sj < stencil; ++sj) {
                    const int j_idx = jj - 1 + sj;
                    for (int sk = 0; sk < stencil; ++sk) {
                        const int k_idx = kk - 1 + sk;

                        corr_stencil(i, si, sj, sk, 0) = velocity_g(i_idx, j_idx, k_idx, 0);
                        corr_stencil(i, si, sj, sk, 1) = velocity_g(i_idx, j_idx, k_idx, 1);
                        corr_stencil(i, si, sj, sk, 2) = velocity_g(i_idx, j_idx, k_idx, 2);
                    }
                }
            }

            if (!has_corr_particles) return;

            // Apply Green's function corrections
            for (int pi = imin; pi < imax; ++pi) {
                for (int pj = jmin; pj < jmax; ++pj) {
                    for (int pk = kmin; pk < kmax; ++pk) {

                        const auto Ci_offset = Neigh_list.binOffset(pi, pj, pk);
                        const auto Ci_size   = Neigh_list.binSize(pi, pj, pk);

                        for (std::size_t r = Ci_offset; r < Ci_offset + Ci_size; ++r) {
                            const auto p = Neigh_list.getParticle(r);

                            const double vortp_x = vorticity_p(p, 0);
                            const double vortp_y = vorticity_p(p, 1);
                            const double vortp_z = vorticity_p(p, 2);

                            const double xp_x = positions(p, 0);
                            const double xp_y = positions(p, 1);
                            const double xp_z = positions(p, 2);

                            const double xp_arr[3]    = {xp_x, xp_y, xp_z};
                            const double vortp_arr[3] = {vortp_x, vortp_y, vortp_z};

                            for (int si = 0; si < stencil; ++si) {
                                const double xg_x = gx[si];
                                for (int sj = 0; sj < stencil; ++sj) {
                                    const double xg_y = gy[sj];
                                    for (int sk = 0; sk < stencil; ++sk) {
                                        const double xg_z = gz[sk];

                                        const double xg_arr[3] = {xg_x, xg_y, xg_z};
                                        double K[3];

                                        // Keep this call exactly as-is
                                        GreensFunction::Calculate_qK_MatVec_Fused(
                                            xg_arr, xp_arr, vortp_arr, K, hp);

                                        corr_stencil(i, si, sj, sk, 0) -= K[0];
                                        corr_stencil(i, si, sj, sk, 1) -= K[1];
                                        corr_stencil(i, si, sj, sk, 2) -= K[2];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

    // Kokkos::fence();

    // ============================================================
    // KERNEL 2: Interpolate corrected stencil to particles
    // ============================================================
    Kokkos::parallel_for(
        "Corrections_Interpolate",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_grid),
        KOKKOS_LAMBDA(const int i)
        {
            const int ii = index(i, 0);
            const int jj = index(i, 1);
            const int kk = index(i, 2);

            const auto Pi_offset = Pi_list.binOffset(ii, jj, kk);
            const auto Pi_size   = Pi_list.binSize(ii, jj, kk);

            if (Pi_size == 0) return;

            // Local stencil for interpolation only
            double vel_loc[stencil][stencil][stencil][3];

            for (int si = 0; si < stencil; ++si) {
                for (int sj = 0; sj < stencil; ++sj) {
                    for (int sk = 0; sk < stencil; ++sk) {
                        vel_loc[si][sj][sk][0] = corr_stencil(i, si, sj, sk, 0);
                        vel_loc[si][sj][sk][1] = corr_stencil(i, si, sj, sk, 1);
                        vel_loc[si][sj][sk][2] = corr_stencil(i, si, sj, sk, 2);
                    }
                }
            }

            for (int r = Pi_offset; r < Pi_offset + Pi_size; ++r) {
                const auto p = Pi_list.getParticle(r);

                const double px = positions(p, 0);
                const double py = positions(p, 1);
                const double pz = positions(p, 2);

                const double wx = vorticity_p(p, 0);
                const double wy = vorticity_p(p, 1);
                const double wz = vorticity_p(p, 2);

                double xp[3] = {px, py, pz};
                double x_plus[3] = {
                    px + half_hp * wx,
                    py + half_hp * wy,
                    pz + half_hp * wz
                };
                double x_minus[3] = {
                    px - half_hp * wx,
                    py - half_hp * wy,
                    pz - half_hp * wz
                };

                {
                    double u_temp[3];
                    MLC_Interp::HarmonicValue_local(vel_loc, g, xp, u_temp);
                    velocity_p(p, 0) = u_temp[0];
                    velocity_p(p, 1) = u_temp[1];
                    velocity_p(p, 2) = u_temp[2];
                }

                double u_plus[3];
                MLC_Interp::HarmonicValue_local(vel_loc, g, x_plus, u_plus);

                double u_minus[3];
                MLC_Interp::HarmonicValue_local(vel_loc, g, x_minus, u_minus);

                advect_vort(p, 0) = (u_plus[0] - u_minus[0]) * inv_hp;
                advect_vort(p, 1) = (u_plus[1] - u_minus[1]) * inv_hp;
                advect_vort(p, 2) = (u_plus[2] - u_minus[2]) * inv_hp;
            }
        });
}

KOKKOS_INLINE_FUNCTION
void Calculate_qK_MatVec_Fused(
    const double xp[3], const double xq[3],
    const double up[3], double K[3],
    const double h )
{
    // Pre-computed constants (candidates for SGPR on AMD)
    const double inv_4pi = 0.07957747154594767;  // 1/(4π)
    const double delta = 0.5 * h;
    const double delta2 = delta * delta;
    
    // Compute distance components (3 VGPRs)
    const double dx = xp[0] - xq[0];
    const double dy = xp[1] - xq[1];
    const double dz = xp[2] - xq[2];
    const double r2 = dx*dx + dy*dy + dz*dz;
    
    // Fast path first: Far-field (most common case on AMD)
    // FIXED: Compare r2 against delta2 (both are squared distances)
    if ( r2 >= delta2 ) 
    {
        // Compute coefficient directly without intermediate storage
        const double r_inv = 1.0 / Kokkos::sqrt(r2);
        const double r3_inv = r_inv * r_inv * r_inv;
        const double c = inv_4pi * r3_inv;
        
        // Fused K_M * up directly into output (eliminates K_M[3][3] array)
        K[0] = c * (dz*up[1] - dy*up[2]);
        K[1] = c * (dx*up[2] - dz*up[0]);
        K[2] = c * (dy*up[0] - dx*up[1]);
    }
    else if ( r2 > 1e-24 )  // Near-field (rare)
    {
        const double r = Kokkos::sqrt(r2);
        const double delta3_inv = 1.0 / (delta * delta2);
        
        // Combine constants to reduce operations
        const double near_const = 0.125 * inv_4pi * delta3_inv;
        const double c = (-12.0 * r2 / delta2 + 20.0) * near_const;
        
        // Same fused computation as far-field
        K[0] = c * (dz*up[1] - dy*up[2]);
        K[1] = c * (dx*up[2] - dz*up[0]);
        K[2] = c * (dy*up[0] - dx*up[1]);
    }
    else  // Singularity (very rare)
    {
        K[0] = 0.0;
        K[1] = 0.0;
        K[2] = 0.0;
    }
}

