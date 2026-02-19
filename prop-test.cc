/*************************************************************************************
Erik Lundstrum 12/01/26
Based on Grid/tests/Test_meson_field.cc

*************************************************************************************/

#include <Grid/Grid.h>
#include <Grid/qcd/utils/A2Autils.h>
//#include "A2A_Mat_Vec_util.h"
#include "IV-photon-props.h"

using namespace Grid;
using namespace std;

typedef typename DomainWallFermionD::ComplexField ComplexField;
typedef typename DomainWallFermionD::FermionField FermionField;

int main(int argc, char *argv[])
{
  // initialization
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid initialized" << std::endl;

  // Lattice and rng setup 
  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(4, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian    grid(latt_size,simd_layout,mpi_layout);
  int Nt = GridDefaultLatt()[Tp];
  int Nd = grid.Dimensions();

  Lattice<iScalar<vInteger>> t(&grid);
  LatticeCoordinate(t, Tp);
  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&grid);
  pRNG.SeedFixedIntegers(seeds);

  // create test photon fields
  //vector<LatticeRealD> phtn_mu_nu(Nd*Nd, &grid);
  vector<ComplexField> phtn_mu_nu(Nd*Nd, &grid);

  cout << GridLogMessage << "START: PHOTON PROPAGATOR GENERATION  "<< endl;
  double start = usecond();

  RealD min = 0.2;
 
  IVPhotonPropagator<WilsonImplR>::FeynmanGaugeMomentumSpace( &phtn_mu_nu[0], min);

  double stop = usecond();
  std::cout << GridLogMessage << "PHOTON PROPAGATOR GENERATION TIME " << stop-start << " us" << std::endl;

  // claude insert check here //
  typename ComplexField::scalar_object site;
  Coordinate coor(Nd, 0);

  // Check diagonal elements at a few momentum sites
  cout << GridLogMessage << "=== Diagonal elements D_mu_mu(k) ===" << endl;
  for (int x = 0; x < 4; x++) {
    coor = Coordinate({x, 0, 0, 0});
    peekSite(site, phtn_mu_nu[0*Nd + 0], coor);  // D_00
    cout << GridLogMessage << "D_00 at n=(" << x << ",0,0,0) = " << site << endl;
  }

  // Check that off-diagonal elements are zero
  cout << GridLogMessage << "=== Off-diagonal elements (should be zero) ===" << endl;
  coor = Coordinate({1, 1, 0, 0});
  peekSite(site, phtn_mu_nu[0*Nd + 1], coor);  // D_01
  cout << GridLogMessage << "D_01 at n=(1,1,0,0) = " << site << endl;
  peekSite(site, phtn_mu_nu[1*Nd + 0], coor);  // D_10
  cout << GridLogMessage << "D_10 at n=(1,1,0,0) = " << site << endl;
  peekSite(site, phtn_mu_nu[2*Nd + 3], coor);  // D_23
  cout << GridLogMessage << "D_23 at n=(1,1,0,0) = " << site << endl;

  // Check IR cutoff: zero mode should be zero
  cout << GridLogMessage << "=== Zero mode (should be zero) ===" << endl;
  coor = Coordinate({0, 0, 0, 0});
  peekSite(site, phtn_mu_nu[0*Nd + 0], coor);
  cout << GridLogMessage << "D_00 at n=(0,0,0,0) = " << site << endl;

  // Manual check: at n=(1,0,0,0), k^2 = (2*pi/Lx)^2
  // prop = 1/k^2 = (Lx/(2*pi))^2
  cout << GridLogMessage << "=== Manual check ===" << endl;
  coor = Coordinate({1, 0, 0, 0});
  peekSite(site, phtn_mu_nu[0*Nd + 0], coor);
  RealD Lx = latt_size[0];
  cout << GridLogMessage << "D_00 at n=(1,0,0,0) = " << site << endl;
  cout << GridLogMessage << "Expected 1/k^2 = (Lx/(2*pi))^2 = " << (Lx*Lx)/(4.0*M_PI*M_PI) << endl;

  //////////////////////////////

  cout << GridLogMessage << "TRY POSITION SPACE FEYNMAN PROP  "<< endl;
  start = usecond();

  IVPhotonPropagator<WilsonImplR>::FeynmanGaugePositionSpace( &phtn_mu_nu[0], min);


  // claude position space test here //

  // Check diagonal at origin and a few sites
  cout << GridLogMessage << "=== Position space D_00(x) ===" << endl;
  for (int x = 0; x < 4; x++) {
    coor = Coordinate({x, 0, 0, 0});
    peekSite(site, phtn_mu_nu[0*Nd + 0], coor);
    cout << GridLogMessage << "D_00 at x=(" << x << ",0,0,0) = " << site << endl;
  }

  // Check off-diagonal still zero in position space (FFT preserves zero)
  cout << GridLogMessage << "=== Position space off-diagonal (should be zero) ===" << endl;
  coor = Coordinate({1, 1, 0, 0});
  peekSite(site, phtn_mu_nu[0*Nd + 1], coor);
  cout << GridLogMessage << "D_01 at x=(1,1,0,0) = " << site << endl;

  // Check symmetry: D(x) should equal D(-x) = D(L-x)
  cout << GridLogMessage << "=== Symmetry check: D_00(1,0,0,0) vs D_00(L-1,0,0,0) ===" << endl;
  coor = Coordinate({1, 0, 0, 0});
  peekSite(site, phtn_mu_nu[0*Nd + 0], coor);
  cout << GridLogMessage << "D_00 at x=(1,0,0,0)   = " << site << endl;
  coor = Coordinate({latt_size[0]-1, 0, 0, 0});
  peekSite(site, phtn_mu_nu[0*Nd + 0], coor);
  cout << GridLogMessage << "D_00 at x=(L-1,0,0,0) = " << site << endl;

  // Propagator should be real in position space
  cout << GridLogMessage << "=== Reality check (imag parts should be ~0) ===" << endl;
  coor = Coordinate({1, 1, 1, 1});
  peekSite(site, phtn_mu_nu[0*Nd + 0], coor);
  cout << GridLogMessage << "D_00 at x=(1,1,1,1) = " << site << endl;

  /////////////////////////////////////

  stop = usecond();
  std::cout << GridLogMessage << "POS SPACE PHOTON PROPAGATOR GENERATION TIME " << stop-start << " us" << std::endl;


  // epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();
  
  return EXIT_SUCCESS;
}

