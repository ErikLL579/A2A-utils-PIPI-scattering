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
const int VDIM = 8; //length of each vector

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
  Lattice<iScalar<vInteger>> t(&grid);
  LatticeCoordinate(t, Tp);
  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&grid);
  pRNG.SeedFixedIntegers(seeds);

  // MesonField lhs and rhs vectors
  const int Nem=1;
  std::vector<FermionField> phi(VDIM,&grid);
  std::vector<ComplexField> B0(Nem,&grid);
  std::vector<ComplexField> B1(Nem,&grid);
  std::cout << GridLogMessage << "Initialising random meson fields" << std::endl;
  for (unsigned int i = 0; i < VDIM; ++i){
    random(pRNG,phi[i]);
  }
  for (unsigned int i = 0; i < Nem; ++i){
    random(pRNG,B0[i]);
    random(pRNG,B1[i]);
  }
  std::cout << GridLogMessage << "Meson fields initialised, rho non-zero only for t = " << TSRC << std::endl;

  // Gamma matrices used in the contraction
  std::vector<Gamma::Algebra> Gmu = {
    Gamma::Algebra::GammaT
  };

  // momentum phases e^{ipx}
  std::vector<std::vector<double>> momenta = {
	  {0.,0.,0.},
  };

  std::cout << GridLogMessage << "Meson fields will be created for " << Gmu.size() << " Gamma matrices and " << momenta.size() << " momenta." << std::endl;

  std::cout << GridLogMessage << "Computing complex phases" << std::endl;
  std::vector<ComplexField> phases(momenta.size(),&grid);
  ComplexField coor(&grid);
  Complex Ci(0.0,1.0);
  for (unsigned int j = 0; j < momenta.size(); ++j)
  {
    phases[j] = Zero();
    for(unsigned int mu = 0; mu < momenta[j].size(); mu++)
    {
      LatticeCoordinate(coor, mu);
      phases[j] = phases[j] + momenta[j][mu]/GridDefaultLatt()[mu]*coor;
    }
    phases[j] = exp((Real)(2*M_PI)*Ci*phases[j]);
  }
  std::cout << GridLogMessage << "Computing complex phases done." << std::endl;

  Eigen::Tensor<ComplexD,5, Eigen::RowMajor> Mpp(momenta.size(),Gmu.size(),Nt,VDIM,VDIM);

  // timer
  double start,stop;

  /////////////////////////////////////////////////////////////////////////
  //execute meson field routine
  /////////////////////////////////////////////////////////////////////////
  start = usecond();
  A2Autils<WilsonImplR>::MesonField(Mpp,&phi[0],&phi[0],Gmu,phases,Tp);
  stop = usecond();
  std::cout << GridLogMessage << "M(phi,phi) created, execution time " << stop-start << " us" << std::endl;

 // ///////////////////////////////////////////////////////////////////////
 // TEST MATRIX-VECTOR OPERATION
 //////////////////////////////////////////////////////////////////////////

 // result of vector-matrix operation
 std::vector<FermionField> y(VDIM,&grid);
 for (unsigned int i = 0; i < VDIM; ++i){
    y[i] = Zero();
  }
  std::cout << GridLogMessage << " ZERO FIELD CREATED " << std::endl;


 // just use Nmodes x Nmodes matrix to begin
 Eigen::Tensor<ComplexD, 2, Eigen::RowMajor> M_test(VDIM, VDIM);
 //M_test = Mpp.chip(0, 0).chip(0, 0).chip(0, 0);

 M_test.setZero();
 for(int i=0; i<VDIM; i++) {
    M_test(i, i) = ComplexD(1.0, 0.0);
 }

 std::cout << GridLogMessage << " ATTEMPTING MAT-VEC " << std::endl;
 start = usecond();
 PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&y[0], M_test, &phi[0]);
 stop = usecond();
 std::cout << GridLogMessage << "M(phi,phi) created, execution time " << stop-start << " us" << std::endl;

 for (int i=0;i<VDIM;i++ ) {
 cout << GridLogMessage << "INITIAL VECTOR NORM = " << norm2(phi[i]) << endl;
 cout << GridLogMessage << "RESULT VECTOR NORM  = " << norm2(y[i]) << endl;
 }
 /*
 template <class FImpl>
template <typename TensorType>
void PipiA2Autils<FImpl>::ContractMesonFieldAndVector(FermionField *y_i1,
                                                      const TensorType &meson_field_ij,
                                                      const FermionField *wj)
 */ 



  // epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();
  
  return EXIT_SUCCESS;
}

