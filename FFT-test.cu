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
const int VDIM = 1000; //length of each vector

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
    Gamma::Algebra::GammaX,
    Gamma::Algebra::GammaT,
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

 cout << GridLogMessage << "CHECKING PHI_MU NORMS. 0 =  " << norm2(phi_mu[0]) << " 1 = " << norm2(phi_mu[1]) << " 2 = " << norm2(phi_mu[2])  << " 3 = " << norm2(phi_mu[3]) << endl;

  cout << GridLogMessage << "==============================================================" << endl;
  cout << GridLogMessage << "TESTING FFT TYPE 1" << endl; 
  cout << GridLogMessage << "==============================================================" << endl;


  cout << GridLogMessage << "START: RANDOM PHOTON PROPAGATOR FIELDS "<< endl;
  // create test photon fields
  vector<ComplexField> phtn_mu_nu(Nd*Nd, &grid);
  for(int mu=0; mu<Nd; mu++) {
    for(int nu=0; nu<Nd; nu++) {
      random(pRNG,phtn_mu_nu[mu*Nd + nu]);
    }
  }
  cout << GridLogMessage << "COMPLETE: RANDOM PHOTON PROPAGATOR FIELDS "<< endl;


  ComplexD Result;

  cout << GridLogMessage << "START: TYPE1 FFT  "<< endl;
  start = usecond();

  PipiA2Autils<WilsonImplR>::FFT_type1_convolve(Result,
		                                &phi_mu[0],
                  		                &phi_mu[0],
                              	    	        &phtn_mu_nu[0]);
  stop = usecond();
  std::cout << GridLogMessage << "FFT type 1 execution time " << stop-start << " us" << std::endl;
  cout << GridLogMessage << "COMPLETE: TYPE1 FFT  "<< endl;
 
  cout << GridLogMessage << "CHECK RESULT = " << Result  << endl;


  cout << GridLogMessage << "==============================================================" << endl;
  cout << GridLogMessage << "TESTING FFT TYPE 2" << endl; 
  cout << GridLogMessage << "==============================================================" << endl; 

  Result = Complex(0.0, 0.0);

  start = usecond();
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result,
  							 &phi1[0], &phi2[0], &phi3[0], &phi4[0],
							 &phtn_mu_nu[0],
							 Gmu);

  stop = usecond();
  std::cout << GridLogMessage << "FFT type 2 execution time " << stop-start << " us" << std::endl;
  cout << GridLogMessage << "COMPLETE: TYPE2 FFT  "<< endl;
  cout << GridLogMessage << "CHECK RESULT = " << Result  << endl;


  // epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();
  
  return EXIT_SUCCESS;
}

