#include <Grid/Grid_Eigen_Tensor.h>


NAMESPACE_BEGIN(Grid);
using namespace std;

template <typename FImpl>
class IVPhotonPropagator
{
  public: 
   
  typedef typename FImpl::ComplexField ComplexField;

  static void FeynmanGaugeMomentumSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta);

  static void CoulombGaugeMomentumSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta);

  static void LandauGaugeMomentumSpace(  ComplexField *phtn_prop_mu_nu, RealD min_momenta){ GRID_ASSERT(0); };

  static void FeynmanGaugePositionSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta);
  
  static void CoulombGaugePositionSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta);
  
  static void LandauGaugePositionSpace(  ComplexField *phtn_prop_mu_nu, RealD min_momenta){ GRID_ASSERT(0); };

};

  template <class FImpl>
  void IVPhotonPropagator<FImpl>::FeynmanGaugeMomentumSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta)
  {
     GridBase *grid = phtn_prop_mu_nu[0].Grid();
     int Nd = grid->Dimensions();
     Coordinate latt_size = grid->FullDimensions(); 

     LatticeRealD   coor(grid);
     LatticeRealD  k_sqr(grid);
     k_sqr = Zero();
  
     // zero out phtn_prop
     for(int i=0; i<Nd*Nd ; i++) phtn_prop_mu_nu[i] = Zero();

     for(int mu=0; mu<Nd; mu++) {
       RealD TwoPiL =  M_PI * 2.0/ latt_size[mu];
       LatticeCoordinate(coor, mu);
       // (-pi, pi]
       coor = where(coor > RealD(latt_size[mu]/2 - 1), coor - RealD(latt_size[mu]), coor);
       k_sqr = k_sqr + coor * coor * (TwoPiL * TwoPiL);       
     }

     LatticeRealD prop(grid);
     LatticeRealD one(grid);
     prop = Zero();                                                                                                                                                                                                                      
     one = RealD(1.0); 

     prop = where(k_sqr < min_momenta, prop , one / k_sqr);

     ComplexField complex_prop(grid);
     complex_prop = toComplex(prop);

     // only one loop over mu here because diagonal matrix
     for(int mu=0; mu<Nd; mu++) {
       phtn_prop_mu_nu[mu*Nd + mu] = complex_prop;     
     }


  };


  template <class FImpl>
  void IVPhotonPropagator<FImpl>::FeynmanGaugePositionSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta)
  {
   GridBase *grid = phtn_prop_mu_nu[0].Grid(); 
   int Nd = grid->Dimensions();

   FFT theFFT(dynamic_cast<GridCartesian *>(grid));

   IVPhotonPropagator<FImpl>::FeynmanGaugeMomentumSpace(&phtn_prop_mu_nu[0], min_momenta);

   // might have to add tmp instead of FFT in place
   // only do diagonal elements bc FFT * 0 = 0 
   for(int mu=0; mu<Nd; mu++) theFFT.FFT_all_dim(phtn_prop_mu_nu[mu*Nd + mu], phtn_prop_mu_nu[mu*Nd + mu], FFT::backward);   

  };


  template <class FImpl>
  void IVPhotonPropagator<FImpl>::CoulombGaugeMomentumSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta)
  {
     GridBase *grid = phtn_prop_mu_nu[0].Grid();
     int Nd = grid->Dimensions();
     Coordinate latt_size = grid->FullDimensions();

     LatticeRealD         coor(grid);
     LatticeRealD         coor2(grid);
     LatticeRealD        k_sqr(grid);
     LatticeRealD  space_k_sqr(grid);

     k_sqr = Zero();
     space_k_sqr = Zero();

     // zero out phtn_prop
     for(int i=0; i<Nd*Nd ; i++) phtn_prop_mu_nu[i] = Zero();

     // spacial momentum
     for(int mu=0; mu<Nd-1; mu++) {
       RealD TwoPiL =  M_PI * 2.0/ latt_size[mu];
       LatticeCoordinate(coor, mu);
       // (-pi, pi]
       coor = where(coor > RealD(latt_size[mu]/2 - 1), coor - RealD(latt_size[mu]), coor);
       space_k_sqr = space_k_sqr + coor * coor * (TwoPiL * TwoPiL);
     }

     // four momentum
     RealD TwoPiL =  M_PI * 2.0/ latt_size[Nd-1];
     LatticeCoordinate(coor, Nd-1);
     coor = where(coor > RealD(latt_size[Nd-1]/2 - 1), coor - RealD(latt_size[Nd-1]), coor);
     k_sqr = space_k_sqr + coor * coor * (TwoPiL * TwoPiL);

     LatticeRealD prop(grid);
     LatticeRealD one(grid);
     LatticeRealD zero(grid);
     zero = Zero();
     prop = Zero();
     one = RealD(1.0);

     prop = where(space_k_sqr < min_momenta, prop , one / space_k_sqr);

     ComplexField complex_prop(grid);
     complex_prop = toComplex(prop);

     // Grid has fourth index as time
     // D_{00} = 1/|\vec{p}^2|
     phtn_prop_mu_nu[Nd*Nd-1] = complex_prop;

     // D_{ij} = (d_{ij} - pi pj / |\vec{p}^2| ) / p^2
     for(int mu=0; mu<Nd-1; mu++) {
       for(int nu=0; nu<Nd-1; nu++) {
         RealD TwoPiL_mu =  M_PI * 2.0/ latt_size[mu];
         RealD TwoPiL_nu =  M_PI * 2.0/ latt_size[nu];
         // pi  
         LatticeCoordinate(coor, mu);
         // pj
         LatticeCoordinate(coor2, nu);
         coor = where(coor > RealD(latt_size[mu]/2 - 1), coor - RealD(latt_size[mu]), coor);
         coor2 = where(coor2 > RealD(latt_size[nu]/2 - 1), coor2 - RealD(latt_size[nu]), coor2);
 
         // - pi pj / |vec{p}^2|
         prop = - coor * coor2 * TwoPiL_mu *TwoPiL_nu / space_k_sqr;

         // d_ij
         if(mu == nu) prop = prop + one;
         
         // 1 / p**2 out front
         prop = where(space_k_sqr < min_momenta, zero , prop / k_sqr);

         // bug in the following with assertion error
         complex_prop = toComplex(prop);
         phtn_prop_mu_nu[mu*Nd + nu] = complex_prop;

       }
     }


  };



  template <class FImpl>
  void IVPhotonPropagator<FImpl>::CoulombGaugePositionSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta)
  {
   GridBase *grid = phtn_prop_mu_nu[0].Grid();
   int Nd = grid->Dimensions();
       
   FFT theFFT(dynamic_cast<GridCartesian *>(grid));
      
   IVPhotonPropagator<FImpl>::CoulombGaugeMomentumSpace(&phtn_prop_mu_nu[0], min_momenta);
     
   for(int mu=0; mu<Nd; mu++) {for(int nu=0; nu<Nd; nu++) theFFT.FFT_all_dim(phtn_prop_mu_nu[mu*Nd + nu], phtn_prop_mu_nu[mu*Nd + nu], FFT::backward); }
     
  };


NAMESPACE_END(Grid);

