#include <Grid/Grid_Eigen_Tensor.h>


NAMESPACE_BEGIN(Grid);
using namespace std;

template <typename FImpl>
class IVPhotonPropagator
{
  public: 
   
  typedef typename FImpl::ComplexField ComplexField;

  static void FeynmanGaugeMomentumSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta);

  static void CoulombGaugeMomentumSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta){ GRID_ASSERT(0); };

  static void LandauGaugeMomentumSpace(  ComplexField *phtn_prop_mu_nu, RealD min_momenta){ GRID_ASSERT(0); };

  static void FeynmanGaugePositionSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta);
  
  static void CoulombGaugePositionSpace( ComplexField *phtn_prop_mu_nu, RealD min_momenta){ GRID_ASSERT(0); };
  
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


NAMESPACE_END(Grid);

