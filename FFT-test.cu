/*************************************************************************************
Erik Lundstrum 12/01/26
Based on Grid/tests/Test_meson_field.cc

*************************************************************************************/

#include <Grid/Grid.h>
#include <Grid/qcd/utils/A2Autils.h>
#include "A2A_Mat_Vec_util.h"

using namespace Grid;
using namespace std;

const int TSRC = 0;  //timeslice where rho is nonzero
const int VDIM = 100; //length of each vector

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

  // MesonField lhs and rhs vectors
  const int Nem=1;
  std::vector<FermionField> phi1(VDIM,&grid);
  std::vector<FermionField> phi2(VDIM,&grid);
  std::vector<FermionField> phi3(VDIM,&grid);
  std::vector<FermionField> phi4(VDIM,&grid);

  std::vector<ComplexField> phi_mu(Nd, &grid);

  std::cout << GridLogMessage << "Initialising random meson fields" << std::endl;
  for (unsigned int i = 0; i < VDIM; ++i) {
    random(pRNG,phi1[i]);
    random(pRNG,phi2[i]);
    random(pRNG,phi3[i]);
    random(pRNG,phi4[i]);
  }

  // Gamma matrices used in the contraction
  std::vector<Gamma::Algebra> Gmu = {
    Gamma::Algebra::GammaT,
    Gamma::Algebra::GammaX,
    Gamma::Algebra::GammaY,
    Gamma::Algebra::GammaZ
  };

  // timer
  double start,stop;

  start = usecond();

// PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&y[0], M_test, &phi[0]);

 PipiA2Autils<WilsonImplR>::FFT_type1_prod( &phi_mu[0],
                                            &phi1[0],
                                            &phi2[0],
                                            Gmu);

 stop = usecond();
 std::cout << GridLogMessage << "FFT type 1 execution time " << stop-start << " us" << std::endl;


  // epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();
  
  return EXIT_SUCCESS;
}

