#ifndef _MULTIGRID_H_
#define _MULTIGRID_H_
#include "Proto.H"
using namespace Proto;
PROTO_KERNEL_START
template<typename T>
void f_greensfcn_F(
                   Point a_pt,
                   Var<T>& a_phi,
                   T a_dx,
                   Point a_ptOffset)
{
  T dist = 0.;
  for (int dir = 0; dir < DIM; dir++)
    {
      T dxdir = a_pt[dir]*a_dx + a_ptOffset[dir]*a_dx;
      dist += dxdir*dxdir;
    }
  dist = max(sqrt(dist),1.e-10);
  if (dist > 1.e-10)
    {
#if DIM==3
  a_phi(0) = -1.0/(4*M_PI*dist);
#else 
  a_phi(0) = (1.0/(2.0*M_PI))*log(dist);
#endif
    }
  else
    {
      a_phi(0) = 0;
    }
}
PROTO_KERNEL_END(f_greensfcn_F,f_greensfcn)
template <typename T>
class Multigrid
{
 public:
  Multigrid(Stencil<T> a_sten,
            T a_diagCoef,
            int a_diamStencil);
  void mgRelax(BoxData<T>& a_phi,
               const BoxData<T>& a_rho,
               T a_h);
  void pointRelax(BoxData<T>& a_phi,
                  const BoxData<T>& a_rho,
                  T a_h,
                  int a_p);
  void residual(BoxData<T>& a_res,
                BoxData<T>& a_phi, 
                const BoxData<T>& a_rho,
                T a_h);
  T resnorm(BoxData<T>& a_phi, 
                const BoxData<T>& a_rho,
                T a_h);
  void average(BoxData<T>& a_rhoCoarse,
               const BoxData<T>& a_rho,
               Box a_bxCoarseValid);
  void interpolate(BoxData<T>& a_phi, 
                   const BoxData<T>& a_phiCoarse);
  void setGhost(BoxData<T>& a_phi,T a_h,int a_stencilDiam = 1);
  void mgSmoother(
                  BoxData<T>& a_phi,
                  BoxData<T>& a_rhs,
                  Stencil<T> a_laplacian,
                  T a_diagCoef,
                  T a_h,
                  int a_diamStencil);
private:
  int m_p;
  int m_pbottom;
  int m_stencilDiam;
  T m_diagCoef;
  Stencil<T> m_laplacian;
};
#include "MultigridImplem.H"
#endif
