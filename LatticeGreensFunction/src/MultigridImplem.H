// #include "Proto_HDF5.H"
template<typename T>
Multigrid<T>::Multigrid(
                     Stencil<T> a_sten,
                     T a_diagCoef,
                     int a_stencilDiam)
{
  m_p = 2*DIM;
  m_pbottom = 10;
  m_laplacian = a_sten;
  m_diagCoef = a_diagCoef;
  m_stencilDiam = a_stencilDiam;
}
template<typename T>
void
Multigrid<T>::mgSmoother(
                      BoxData<T>& a_phi,
                      BoxData<T>& a_rhs,
                      Stencil<T> a_laplacian,
                      T a_diagCoef,
                      T a_h,
                      int a_diamStencil)
{
  PR_TIMERS("Multigrid<T>::mgSmoother");
  BoxData<T> rhsSmoother(a_rhs.box()); 

  setGhost(a_phi,a_h,a_diamStencil);
  BoxData<T> phiSmoother(a_rhs.box().grow(Point::Ones(m_stencilDiam)));
  BoxData<T> phiSmoother0(a_rhs.box().grow(Point::Ones(m_stencilDiam)));
  
  a_phi.copyTo(phiSmoother0);
  phiSmoother0.copyTo(phiSmoother);
  rhsSmoother.setVal(0.);
  rhsSmoother += a_laplacian(a_phi,-1.0/(a_h*a_h));
  rhsSmoother += a_rhs;
  rhsSmoother += m_laplacian(phiSmoother0,1.0/(a_h*a_h));
  mgRelax(phiSmoother,rhsSmoother,a_h);
  phiSmoother -= phiSmoother0;
  a_phi += phiSmoother;

  for (int iter = 0; iter < m_p;iter++)
    {
      setGhost(a_phi,a_h,a_diamStencil);
      rhsSmoother.setVal(0.);
      rhsSmoother += a_rhs;
      rhsSmoother *= -1.0;
      rhsSmoother += a_laplacian(a_phi,1.0/(a_h*a_h));
      rhsSmoother *= a_h*a_h/(a_diagCoef*2);
      a_phi += rhsSmoother;
    }     
}
template<typename T>
void Multigrid<T>::mgRelax(
                        BoxData<T>& a_phi,
                        const BoxData<T>& a_rhs,
                        T a_h)
{
  PR_TIMERS("Multigrid<T>::mgRelax");
  if (a_rhs.box().size() > 1) 
    {
      BoxData<T> res(a_rhs.box());
      Box bxCoarse = a_phi.box().coarsen(Point::Ones(2));      
      BoxData<T> phiCoarse(bxCoarse);
      BoxData<T> phiCoarse0(bxCoarse);
      BoxData<T> resCoarse(bxCoarse.grow(Point::Ones(-1)));
      
      pointRelax(a_phi,a_rhs,a_h,m_p);
      setGhost(a_phi,a_h);    
      average(phiCoarse0,a_phi,a_phi.box());
      setGhost(phiCoarse0,2*a_h);
      phiCoarse0.copyTo(phiCoarse);
      residual(res,a_phi,a_rhs,a_h);
      average(resCoarse,res,a_phi.box());     
      BoxData<T> rhsCoarse = m_laplacian(phiCoarse0,1./(4*a_h*a_h));
      rhsCoarse -= resCoarse;
      mgRelax(phiCoarse,rhsCoarse,2*a_h);
      setGhost(phiCoarse,2*a_h);
      phiCoarse -= phiCoarse0;
      interpolate(a_phi,phiCoarse);
      pointRelax(a_phi,a_rhs,a_h,m_p);
    }
  else
    {
      pointRelax(a_phi,a_rhs,a_h,m_pbottom);
    }
}
template<typename T>
void
Multigrid<T>::residual(
                    BoxData<T>& a_residual,
                    BoxData<T>& a_phi,
                    const BoxData<T>& a_rhs,
                    T a_h)
{
  PR_TIMERS("Multigrid<T>::residual");
  setGhost(a_phi,a_h);
  a_residual |= m_laplacian(a_phi,1./(a_h*a_h));
  a_residual -= a_rhs;  
}
template<typename T>
void
Multigrid<T>::pointRelax(BoxData<T>& a_phi, 
                      const BoxData<T>& a_rhs,
                      T a_h, 
                      int a_numIter)
{
  PR_TIMERS("Multigrid<T>::pointRelax");
  BoxData<T> res(a_rhs.box());
  T rparam = a_h*a_h/(m_diagCoef*2);
  for (int iter =  0; iter < a_numIter; iter++)
    {
      setGhost(a_phi,a_h);
      residual(res,a_phi,a_rhs,a_h);
      //res *= a_h*a_h/(m_diagCoef*4);
      // a_phi += res;
      forallInPlace([] PROTO_LAMBDA(Var<T>& a_phi,
                                    const Var<T>& a_res,T
                                    a_relaxparam)
                    {
                      a_phi(0) += a_res(0)*a_relaxparam;
                    },a_phi,res,rparam);
    }
}
template<typename T>
void Multigrid<T>::setGhost(BoxData<T>& a_phi,
                         T a_h,
                         int a_stencilDiam)
{
  PR_TIMERS("Multigrid<T>::setGhost");
  Box bx = a_phi.box();
  for (int dir = 0; dir < DIM; dir++)
    {
      Box bxdir = bx.grow(Point::Basis(dir,-a_stencilDiam)); 
      Box bxlo = bxdir.adjacent(dir,Side::Lo,a_stencilDiam);
      forallInPlace_p(f_greensfcn,bxlo,a_phi,a_h,Point::Zeros());
      Box bxhi = bxdir.adjacent(dir,Side::Hi,a_stencilDiam);
      forallInPlace_p(f_greensfcn,bxhi,a_phi,a_h,Point::Zeros());
    }
}
template<typename T>
T Multigrid<T>::resnorm(BoxData<T>& a_phi,
                          const BoxData<T>& a_rho,
                          T a_h)
{
  BoxData<T> res(a_rho.box()); 
  residual(res,a_phi,a_rho,a_h);
  T retval = res.absMax();
  return retval;
}
template<typename T>
void Multigrid<T>::average(BoxData<T>& a_bdC,
                        const BoxData<T>& a_bd,
                        Box a_bx)
{
  PR_TIMERS("Multigrid<T>::average");
  // HDF5Handler h5;
  BoxData<T> temp(a_bx);
  a_bd.copyTo(temp);
  BoxData<T> tempNew;
  Box bxPre = a_bx;
  for (int dir = 0;dir < DIM; dir++)
    {
      Box bxPost = bxPre.coarsen(Point::Ones() + Point::Basis(dir,1));
      Box bxPostValid = bxPost.grow(Point::Basis(dir,-1));
      Stencil<T> stencilDir =
        .25*Shift(Point::Basis(dir,1))+
        .25*Shift(Point::Basis(dir,-1))+
        .5*Shift(Point::Zeros());
      stencilDir.srcRatio() = Point::Ones() + Point::Basis(dir,1);
      tempNew = stencilDir(temp);
      temp.define(bxPost);
      tempNew.copyTo(temp);
      //h5.writePatch({"data"},1.0,temp,"average"+to_string(dir),0);
      bxPre = bxPost;
    }
  temp.copyTo(a_bdC);
}
template<typename T>
void Multigrid<T>::interpolate(
                            BoxData<T>& a_phi,
                            const BoxData<T>& a_deltaC)
{
  PR_TIMERS("Multigrid<T>::interpolate");
  // HDF5Handler h5;
  Box bxNext = a_deltaC.box();
  BoxData<T> temp,tempNew;
  temp.define(bxNext);
  temp.setVal(0.);
  Stencil<T> inject = 1.0*Shift(Point::Zeros());
  a_deltaC.copyTo(temp);
  //h5.writePatch({"data"},1.0,temp,"interpStart",0);
  for (int dir = 0;dir < DIM; dir++)
    {
      bxNext = bxNext.refine(Point::Ones() + Point::Basis(dir,1));
      tempNew.define(bxNext);
      tempNew.setVal(0.);
      Stencil<T> stencilOdd =
        .5*Shift(Point::Basis(dir,0)) +
        .5*Shift(Point::Basis(dir,1));
      stencilOdd.destRatio() = Point::Ones() + Point::Basis(dir,1);
      stencilOdd.destShift() = Point::Basis(dir,1);
      Stencil<T> stencilEven = 1.0*Shift(Point::Zeros());
      stencilEven.destRatio() = Point::Ones() + Point::Basis(dir,1);
      tempNew |= stencilEven(temp);     
      //h5.writePatch({"data"},1.0,tempNew,"interpEven"+to_string(dir),0);
      tempNew |= stencilOdd(temp);
      //h5.writePatch({"data"},1.0,tempNew,"interpOdd"+to_string(dir),0);      
      temp = inject(tempNew);
    }
  a_phi += temp;
  //h5.writePatch({"data"},1.0,temp,"interpEnd",0);
}
