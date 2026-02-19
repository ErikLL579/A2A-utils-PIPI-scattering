#include <Grid/Grid_Eigen_Tensor.h>


NAMESPACE_BEGIN(Grid);
using namespace std;

template <typename FImpl>
class IVPhotonPropagator
{
  public: 
   
  typedef typename FImpl::ComplexField ComplexField;

  static void FeynmanGaugeMomentumSpace( ComplexField *phtn_prop_mu_nu, ComplexD min_momenta);

  static void CoulombGaugeMomentumSpace( ComplexField *phtn_prop_mu_nu, ComplexD min_momenta){ GRID_ASSERT(0); };

  static void LandauGaugeMomentumSpace(  ComplexField *phtn_prop_mu_nu, ComplexD min_momenta){ GRID_ASSERT(0); };

  static void FeynmanGaugePositionSpace( ComplexField *phtn_prop_mu_nu, ComplexD min_momenta);
  
  static void CoulombGaugePositionSpace( ComplexField *phtn_prop_mu_nu, ComplexD min_momenta){ GRID_ASSERT(0); };
  
  static void LandauGaugePositionSpace(  ComplexField *phtn_prop_mu_nu, ComplexD min_momenta){ GRID_ASSERT(0); };

};

  template <class FImpl>
  void IVPhotonPropagator<FImpl>::FeynmanGaugeMomentumSpace( ComplexField *phtn_prop_mu_nu, ComplexD min_momenta)
  {
     GridBase *grid = phtn_prop_mu_nu[0].Grid();
     int Nd = grid->Dimensions();
     Coordinate latt_size = grid->FullDimensions(); 

     LatticeComplexD   coor(grid);
     LatticeComplexD  k_sqr(grid);
     k_sqr = Zero();
  
     // zero out phtn_prop
     for(int i=0; i<Nd*Nd ; i++) phtn_prop_mu_nu[i] = Zero();

     for(int mu=0; mu<Nd; mu++) {
       RealD TwoPiL =  M_PI * 2.0/ latt_size[mu];
       LatticeCoordinate(coor, mu);
       // (-pi, pi]
       coor = where(IsTrue(coor > RealD(latt_size[mu]/2 - 1)), coor - RealD(latt_size[mu]), coor);
       k_sqr = k_sqr + coor * coor * (TwoPiL * TwoPiL);       
     }

     LatticeComplexD k_sqr_safe = where(IsTrue(k_sqr < min_momenta), ComplexD(1.0, 0.0), k_sqr);
     LatticeComplexD prop = where(IsTrue(k_sqr < min_momenta), ComplexD(0.0, 0.0), 1.0 / k_sqr_safe);

     // only one loop over mu here because diagonal matrix
     for(int mu=0; mu<Nd; mu++) {
       phtn_prop_mu_nu[mu*Nd + mu] = prop;     
     }


  };



  template <class FImpl>
  void IVPhotonPropagator<FImpl>::FeynmanGaugePositionSpace( ComplexField *phtn_prop_mu_nu, ComplexD min_momenta)
  {
   GridBase *grid = phtn_prop_mu_nu[0].Grid(); 
   int Nd = grid->Dimensions();

   FFT theFFT(dynamic_cast<GridCartesian *>(grid));

   IVPhotonPropagator<FImpl>::FeynmanGaugeMomentumSpace(&phtn_prop_mu_nu[0], min_momenta);

   for(int i=0; i<Nd*Nd; i++) { theFFT.FFT_all_dim(phtn_prop_mu_nu[i], phtn_prop_mu_nu[i], FFT::backward);
   

  };


NAMESPACE_END(Grid);

