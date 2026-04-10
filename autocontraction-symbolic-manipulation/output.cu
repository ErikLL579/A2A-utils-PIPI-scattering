
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

  // initial w and v vectors (need to import these correctly)
  std::vector<FermionField> w(VDIM,&grid);
  std::vector<FermionField> v(VDIM,&grid);


// ====================================================== //
// == DEFINE AND CONSTRUCT PION SOURCE & PROD MATRICES == //
// ====================================================== //


  int times = 4;
  Eigen::Tensor<ComplexD, 3, Eigen::RowMajor> Pion_source_fields(times, VDIM, VDIM);


  int num_products_level_1 = 8;
  Eigen::Tensor<ComplexD, 3, Eigen::RowMajor> Pion_product_fields_level_1(num_products, VDIM, VDIM);


  int num_products_level_2 = 16;
  Eigen::Tensor<ComplexD, 3, Eigen::RowMajor> Pion_product_fields_level_2(num_products, VDIM, VDIM);

 
// CONTRACTION VECTORS FOR MATRIX-MATRIX MULTPLIES  //
  vector<int> A_vector_contractions = {60000,60000,30000,30000,30000,30000,0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,};
  vector<int> B_vector_contractions = {30000,0,60000,30000,30000,0,60000,30000,30000,5,30000,4,0,7,0,6,60000,1,60000,0,30000,3,30000,2,};
  vector<int> C_vector_contractions = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,};
 
// BUFFER FLAGS FOR MATRIX-MATRIX MULTPLIES  //
  vector<int> buffer_flag_A = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,};
  vector<int> buffer_flag_B = {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,};
 
// PERFORM MATRIX MULTIPLICATIONS //

  PipiA2Autils<WilsonImplR>::MesonField_MesonField_connected(Pion_source_fields,
                                                             Pion_product_fields_level_1,
                                                             Pion_product_fields_level_2,
                                                             num_products_level_1,
                                                             buffer_flag_A,
                                                             buffer_flag_B,
                                                             A_vector_contractions,
                                                             B_vector_contractions,
                                                             C_vector_contractions);


// ====================================================== //
// ========= COMPLETE MATRIX-MATRIX OPERATIONS ========== //
// ====================================================== //


// ======================================= //
// ===== GENERATE PHOTON PROPAGATORs ===== //
// ======================================= //


  RealD min = 1.0;
  vector<ComplexField> phtn_mu_nu(Nd*Nd, &grid);


  IVPhotonPropagator<WilsonImplR>::CoulombGaugeMomentumSpace( &phtn_mu_nu[0], min);


// ======================================= //
// == DEFINE ALL MATRIX-VECTOR PRODUCTS == //
// ======================================= //


  std::vector<FermionField> prod_vec1_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec1_mu[i] = Zero();


  std::vector<FermionField> prod_vec2_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec2_mu[i] = Zero();


  std::vector<FermionField> prod_vec3_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec3_mu[i] = Zero();


  std::vector<FermionField> prod_vec4_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec4_mu[i] = Zero();


  std::vector<FermionField> prod_vec5_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec5_mu[i] = Zero();


  std::vector<FermionField> prod_vec6_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec6_mu[i] = Zero();


  std::vector<FermionField> prod_vec7_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec7_mu[i] = Zero();


  std::vector<FermionField> prod_vec8_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec8_mu[i] = Zero();


  std::vector<FermionField> prod_vec9_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec9_mu[i] = Zero();


  std::vector<FermionField> prod_vec10_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec10_mu[i] = Zero();


  std::vector<FermionField> prod_vec11_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec11_mu[i] = Zero();


  std::vector<FermionField> prod_vec12_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec12_mu[i] = Zero();


  std::vector<FermionField> prod_vec13_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec13_mu[i] = Zero();


  std::vector<FermionField> prod_vec14_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec14_mu[i] = Zero();


  std::vector<FermionField> prod_vec15_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec15_mu[i] = Zero();


  std::vector<FermionField> prod_vec16_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec16_mu[i] = Zero();


  std::vector<FermionField> prod_vec17_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec17_mu[i] = Zero();


  std::vector<FermionField> prod_vec18_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec18_mu[i] = Zero();


  std::vector<FermionField> prod_vec19_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec19_mu[i] = Zero();


  std::vector<FermionField> prod_vec20_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec20_mu[i] = Zero();


  std::vector<FermionField> prod_vec21_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec21_mu[i] = Zero();


  std::vector<FermionField> prod_vec22_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec22_mu[i] = Zero();


  std::vector<FermionField> prod_vec23_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec23_mu[i] = Zero();


  std::vector<FermionField> prod_vec24_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec24_mu[i] = Zero();


  std::vector<FermionField> prod_vec25_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec25_mu[i] = Zero();


  std::vector<FermionField> prod_vec26_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec26_mu[i] = Zero();


  std::vector<FermionField> prod_vec27_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec27_mu[i] = Zero();


  std::vector<FermionField> prod_vec28_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec28_mu[i] = Zero();


// ======================================= //
// ===== BEGIN MAT-VECTOR OPERATIONS ===== //
// ======================================= //


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec1_mu[0], Pion_source_fields[3], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec2_mu[0], Pion_source_fields[1], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec3_mu[0], Pion_source_fields[2], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec4_mu[0], Pion_source_fields[0], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec5_mu[0], Pion_product_fields_level_1[0], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec6_mu[0], Pion_product_fields_level_1[1], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec7_mu[0], Pion_product_fields_level_1[2], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec8_mu[0], Pion_product_fields_level_1[3], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec9_mu[0], Pion_product_fields_level_1[4], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec10_mu[0], Pion_product_fields_level_1[5], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec11_mu[0], Pion_product_fields_level_1[6], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec12_mu[0], Pion_product_fields_level_1[7], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec13_mu[0], Pion_product_fields_level_2[1], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec14_mu[0], Pion_product_fields_level_2[2], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec15_mu[0], Pion_product_fields_level_2[3], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec16_mu[0], Pion_product_fields_level_2[4], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec17_mu[0], Pion_product_fields_level_2[5], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec18_mu[0], Pion_product_fields_level_2[6], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec19_mu[0], Pion_product_fields_level_2[7], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec20_mu[0], Pion_product_fields_level_2[8], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec21_mu[0], Pion_product_fields_level_2[9], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec22_mu[0], Pion_product_fields_level_2[10], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec23_mu[0], Pion_product_fields_level_2[11], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec24_mu[0], Pion_product_fields_level_2[12], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec25_mu[0], Pion_product_fields_level_2[13], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec26_mu[0], Pion_product_fields_level_2[14], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec27_mu[0], Pion_product_fields_level_2[15], &v[0]);


  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec28_mu[0], Pion_product_fields_level_2[0], &v[0]);


  std::vector<Gamma::Algebra> Gmu = {
      Gamma::Algebra::GammaX,
      Gamma::Algebra::GammaT,
      Gamma::Algebra::GammaY,
      Gamma::Algebra::GammaZ
  };


  ComplexD prod_Pi1_trace = Trace(Pion_product_fields_level_1[0]);


  ComplexD Result_term_Type10_0;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type10_0,
                                                         &w[0],
                                                         &prod_vec3_mu[0],
                                                         &w[0],
                                                         &prod_vec4_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type10_0 *= prod_Pi1_trace;


  // quark charge factor
  Result_term_Type10_0 *= (-2/9);


  ComplexD prod_Pi2_trace = Trace(Pion_product_fields_level_1[1]);


  ComplexD Result_term_Type10_1;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type10_1,
                                                         &w[0],
                                                         &prod_vec3_mu[0],
                                                         &w[0],
                                                         &prod_vec2_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type10_1 *= prod_Pi2_trace;


  // quark charge factor
  Result_term_Type10_1 *= (-2/9);


  ComplexD prod_Pi5_trace = Trace(Pion_product_fields_level_1[4]);


  ComplexD Result_term_Type10_2;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type10_2,
                                                         &w[0],
                                                         &prod_vec1_mu[0],
                                                         &w[0],
                                                         &prod_vec4_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type10_2 *= prod_Pi5_trace;


  // quark charge factor
  Result_term_Type10_2 *= (-2/9);


  ComplexD prod_Pi6_trace = Trace(Pion_product_fields_level_1[5]);


  ComplexD Result_term_Type10_3;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type10_3,
                                                         &w[0],
                                                         &prod_vec1_mu[0],
                                                         &w[0],
                                                         &prod_vec2_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type10_3 *= prod_Pi6_trace;


  // quark charge factor
  Result_term_Type10_3 *= (-2/9);


  std::vector<ComplexField> term_Type11_0_FFT1_phi_mu_A(Nd, &grid);
  std::vector<ComplexField> term_Type11_0_FFT1_phi_mu_B(Nd, &grid);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type11_0_FFT1_phi_mu_A[0], &w[0], &prod_vec12_mu[0], Gmu);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type11_0_FFT1_phi_mu_B[0], &w[0], &prod_vec7_mu[0], Gmu);


  ComplexD Result_term_Type11_0;
  PipiA2Autils<WilsonImplR>::FFT_type1_convolve(Result_term_Type11_0, &term_Type11_0_FFT1_phi_mu_A[0], &term_Type11_0_FFT1_phi_mu_B[0], &phtn_mu_nu[0]);


  // quark charge factor
  Result_term_Type11_0 *= (4/9);


  std::vector<ComplexField> term_Type11_1_FFT1_phi_mu_A(Nd, &grid);
  std::vector<ComplexField> term_Type11_1_FFT1_phi_mu_B(Nd, &grid);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type11_1_FFT1_phi_mu_A[0], &w[0], &prod_vec11_mu[0], Gmu);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type11_1_FFT1_phi_mu_B[0], &w[0], &prod_vec8_mu[0], Gmu);


  ComplexD Result_term_Type11_1;
  PipiA2Autils<WilsonImplR>::FFT_type1_convolve(Result_term_Type11_1, &term_Type11_1_FFT1_phi_mu_A[0], &term_Type11_1_FFT1_phi_mu_B[0], &phtn_mu_nu[0]);


  // quark charge factor
  Result_term_Type11_1 *= (4/9);


  ComplexD Result_term_Type12_0;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type12_0,
                                                         &w[0],
                                                         &prod_vec12_mu[0],
                                                         &w[0],
                                                         &prod_vec7_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type12_0 *= (-4/9);


  ComplexD Result_term_Type12_1;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type12_1,
                                                         &w[0],
                                                         &prod_vec11_mu[0],
                                                         &w[0],
                                                         &prod_vec8_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type12_1 *= (-4/9);


  ComplexD Result_term_Type1_0;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type1_0,
                                                         &w[0],
                                                         &prod_vec23_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type1_0 *= (-1/9);


  ComplexD Result_term_Type1_1;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type1_1,
                                                         &w[0],
                                                         &prod_vec21_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type1_1 *= (-1/9);


  ComplexD Result_term_Type1_2;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type1_2,
                                                         &w[0],
                                                         &prod_vec15_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type1_2 *= (-1/9);


  ComplexD Result_term_Type1_3;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type1_3,
                                                         &w[0],
                                                         &prod_vec13_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type1_3 *= (-1/9);


  ComplexD Result_term_Type2_0;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type2_0,
                                                         &w[0],
                                                         &prod_vec10_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type2_0 *= prod_Pi1_trace;


  // quark charge factor
  Result_term_Type2_0 *= (1/9);


  ComplexD Result_term_Type2_1;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type2_1,
                                                         &w[0],
                                                         &prod_vec9_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type2_1 *= prod_Pi2_trace;


  // quark charge factor
  Result_term_Type2_1 *= (1/9);


  ComplexD Result_term_Type2_2;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type2_2,
                                                         &w[0],
                                                         &prod_vec6_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type2_2 *= prod_Pi5_trace;


  // quark charge factor
  Result_term_Type2_2 *= (1/9);


  ComplexD Result_term_Type2_3;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type2_3,
                                                         &w[0],
                                                         &prod_vec5_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type2_3 *= prod_Pi6_trace;


  // quark charge factor
  Result_term_Type2_3 *= (1/9);


  ComplexD Result_term_Type3_0;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type3_0,
                                                         &w[0],
                                                         &prod_vec27_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type3_0 *= (-4/9);


  ComplexD Result_term_Type3_1;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type3_1,
                                                         &w[0],
                                                         &prod_vec25_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type3_1 *= (-4/9);


  ComplexD Result_term_Type3_2;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type3_2,
                                                         &w[0],
                                                         &prod_vec19_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type3_2 *= (-4/9);


  ComplexD Result_term_Type3_3;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type3_3,
                                                         &w[0],
                                                         &prod_vec17_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type3_3 *= (-4/9);


  ComplexD Result_term_Type4_0;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type4_0,
                                                         &w[0],
                                                         &prod_vec12_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type4_0 *= prod_Pi1_trace;


  // quark charge factor
  Result_term_Type4_0 *= (4/9);


  ComplexD Result_term_Type4_1;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type4_1,
                                                         &w[0],
                                                         &prod_vec11_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type4_1 *= prod_Pi5_trace;


  // quark charge factor
  Result_term_Type4_1 *= (4/9);


  ComplexD Result_term_Type4_2;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type4_2,
                                                         &w[0],
                                                         &prod_vec8_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type4_2 *= prod_Pi2_trace;


  // quark charge factor
  Result_term_Type4_2 *= (4/9);


  ComplexD Result_term_Type4_3;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type4_3,
                                                         &w[0],
                                                         &prod_vec7_mu[0],
                                                         &w[0],
                                                         &v[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // pi product factor
  Result_term_Type4_3 *= prod_Pi6_trace;


  // quark charge factor
  Result_term_Type4_3 *= (4/9);


  ComplexD Result_term_Type5_0;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type5_0,
                                                         &w[0],
                                                         &prod_vec22_mu[0],
                                                         &w[0],
                                                         &prod_vec2_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type5_0 *= (2/9);


  ComplexD Result_term_Type5_1;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type5_1,
                                                         &w[0],
                                                         &prod_vec20_mu[0],
                                                         &w[0],
                                                         &prod_vec4_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type5_1 *= (2/9);


  ComplexD Result_term_Type5_2;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type5_2,
                                                         &w[0],
                                                         &prod_vec14_mu[0],
                                                         &w[0],
                                                         &prod_vec2_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type5_2 *= (2/9);


  ComplexD Result_term_Type5_3;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type5_3,
                                                         &w[0],
                                                         &prod_vec28_mu[0],
                                                         &w[0],
                                                         &prod_vec4_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type5_3 *= (2/9);


  std::vector<ComplexField> term_Type6_0_FFT1_phi_mu_A(Nd, &grid);
  std::vector<ComplexField> term_Type6_0_FFT1_phi_mu_B(Nd, &grid);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type6_0_FFT1_phi_mu_A[0], &w[0], &prod_vec10_mu[0], Gmu);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type6_0_FFT1_phi_mu_B[0], &w[0], &prod_vec5_mu[0], Gmu);


  ComplexD Result_term_Type6_0;
  PipiA2Autils<WilsonImplR>::FFT_type1_convolve(Result_term_Type6_0, &term_Type6_0_FFT1_phi_mu_A[0], &term_Type6_0_FFT1_phi_mu_B[0], &phtn_mu_nu[0]);


  // quark charge factor
  Result_term_Type6_0 *= (1/9);


  std::vector<ComplexField> term_Type6_1_FFT1_phi_mu_A(Nd, &grid);
  std::vector<ComplexField> term_Type6_1_FFT1_phi_mu_B(Nd, &grid);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type6_1_FFT1_phi_mu_A[0], &w[0], &prod_vec9_mu[0], Gmu);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type6_1_FFT1_phi_mu_B[0], &w[0], &prod_vec6_mu[0], Gmu);


  ComplexD Result_term_Type6_1;
  PipiA2Autils<WilsonImplR>::FFT_type1_convolve(Result_term_Type6_1, &term_Type6_1_FFT1_phi_mu_A[0], &term_Type6_1_FFT1_phi_mu_B[0], &phtn_mu_nu[0]);


  // quark charge factor
  Result_term_Type6_1 *= (1/9);


  std::vector<ComplexField> term_Type7_0_FFT1_phi_mu_A(Nd, &grid);
  std::vector<ComplexField> term_Type7_0_FFT1_phi_mu_B(Nd, &grid);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type7_0_FFT1_phi_mu_A[0], &w[0], &prod_vec10_mu[0], Gmu);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type7_0_FFT1_phi_mu_B[0], &w[0], &prod_vec7_mu[0], Gmu);


  ComplexD Result_term_Type7_0;
  PipiA2Autils<WilsonImplR>::FFT_type1_convolve(Result_term_Type7_0, &term_Type7_0_FFT1_phi_mu_A[0], &term_Type7_0_FFT1_phi_mu_B[0], &phtn_mu_nu[0]);


  // quark charge factor
  Result_term_Type7_0 *= (-2/9);


  std::vector<ComplexField> term_Type7_1_FFT1_phi_mu_A(Nd, &grid);
  std::vector<ComplexField> term_Type7_1_FFT1_phi_mu_B(Nd, &grid);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type7_1_FFT1_phi_mu_A[0], &w[0], &prod_vec9_mu[0], Gmu);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type7_1_FFT1_phi_mu_B[0], &w[0], &prod_vec11_mu[0], Gmu);


  ComplexD Result_term_Type7_1;
  PipiA2Autils<WilsonImplR>::FFT_type1_convolve(Result_term_Type7_1, &term_Type7_1_FFT1_phi_mu_A[0], &term_Type7_1_FFT1_phi_mu_B[0], &phtn_mu_nu[0]);


  // quark charge factor
  Result_term_Type7_1 *= (-2/9);


  std::vector<ComplexField> term_Type7_2_FFT1_phi_mu_A(Nd, &grid);
  std::vector<ComplexField> term_Type7_2_FFT1_phi_mu_B(Nd, &grid);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type7_2_FFT1_phi_mu_A[0], &w[0], &prod_vec6_mu[0], Gmu);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type7_2_FFT1_phi_mu_B[0], &w[0], &prod_vec8_mu[0], Gmu);


  ComplexD Result_term_Type7_2;
  PipiA2Autils<WilsonImplR>::FFT_type1_convolve(Result_term_Type7_2, &term_Type7_2_FFT1_phi_mu_A[0], &term_Type7_2_FFT1_phi_mu_B[0], &phtn_mu_nu[0]);


  // quark charge factor
  Result_term_Type7_2 *= (-2/9);


  std::vector<ComplexField> term_Type7_3_FFT1_phi_mu_A(Nd, &grid);
  std::vector<ComplexField> term_Type7_3_FFT1_phi_mu_B(Nd, &grid);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type7_3_FFT1_phi_mu_A[0], &w[0], &prod_vec5_mu[0], Gmu);


  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&term_Type7_3_FFT1_phi_mu_B[0], &w[0], &prod_vec12_mu[0], Gmu);


  ComplexD Result_term_Type7_3;
  PipiA2Autils<WilsonImplR>::FFT_type1_convolve(Result_term_Type7_3, &term_Type7_3_FFT1_phi_mu_A[0], &term_Type7_3_FFT1_phi_mu_B[0], &phtn_mu_nu[0]);


  // quark charge factor
  Result_term_Type7_3 *= (-2/9);


  ComplexD Result_term_Type8_0;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type8_0,
                                                         &w[0],
                                                         &prod_vec10_mu[0],
                                                         &w[0],
                                                         &prod_vec5_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type8_0 *= (-1/9);


  ComplexD Result_term_Type8_1;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type8_1,
                                                         &w[0],
                                                         &prod_vec9_mu[0],
                                                         &w[0],
                                                         &prod_vec6_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type8_1 *= (-1/9);


  ComplexD Result_term_Type9_0;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type9_0,
                                                         &w[0],
                                                         &prod_vec3_mu[0],
                                                         &w[0],
                                                         &prod_vec24_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type9_0 *= (2/9);


  ComplexD Result_term_Type9_1;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type9_1,
                                                         &w[0],
                                                         &prod_vec3_mu[0],
                                                         &w[0],
                                                         &prod_vec16_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type9_1 *= (2/9);


  ComplexD Result_term_Type9_2;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type9_2,
                                                         &w[0],
                                                         &prod_vec1_mu[0],
                                                         &w[0],
                                                         &prod_vec26_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type9_2 *= (2/9);


  ComplexD Result_term_Type9_3;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve(Result_term_Type9_3,
                                                         &w[0],
                                                         &prod_vec1_mu[0],
                                                         &w[0],
                                                         &prod_vec18_mu[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);


  // quark charge factor
  Result_term_Type9_3 *= (2/9);

  // ====================================================== //
  // ============= SAVE RESULTS TO TXT FILE =============== //
  // ====================================================== //
  {
    std::ofstream outfile("EM_results.txt");
    outfile << "Type10_0 = " << Result_term_Type10_0 << std::endl;
    outfile << "Type10_1 = " << Result_term_Type10_1 << std::endl;
    outfile << "Type10_2 = " << Result_term_Type10_2 << std::endl;
    outfile << "Type10_3 = " << Result_term_Type10_3 << std::endl;
    outfile << "Type11_0 = " << Result_term_Type11_0 << std::endl;
    outfile << "Type11_1 = " << Result_term_Type11_1 << std::endl;
    outfile << "Type12_0 = " << Result_term_Type12_0 << std::endl;
    outfile << "Type12_1 = " << Result_term_Type12_1 << std::endl;
    outfile << "Type1_0 = " << Result_term_Type1_0 << std::endl;
    outfile << "Type1_1 = " << Result_term_Type1_1 << std::endl;
    outfile << "Type1_2 = " << Result_term_Type1_2 << std::endl;
    outfile << "Type1_3 = " << Result_term_Type1_3 << std::endl;
    outfile << "Type2_0 = " << Result_term_Type2_0 << std::endl;
    outfile << "Type2_1 = " << Result_term_Type2_1 << std::endl;
    outfile << "Type2_2 = " << Result_term_Type2_2 << std::endl;
    outfile << "Type2_3 = " << Result_term_Type2_3 << std::endl;
    outfile << "Type3_0 = " << Result_term_Type3_0 << std::endl;
    outfile << "Type3_1 = " << Result_term_Type3_1 << std::endl;
    outfile << "Type3_2 = " << Result_term_Type3_2 << std::endl;
    outfile << "Type3_3 = " << Result_term_Type3_3 << std::endl;
    outfile << "Type4_0 = " << Result_term_Type4_0 << std::endl;
    outfile << "Type4_1 = " << Result_term_Type4_1 << std::endl;
    outfile << "Type4_2 = " << Result_term_Type4_2 << std::endl;
    outfile << "Type4_3 = " << Result_term_Type4_3 << std::endl;
    outfile << "Type5_0 = " << Result_term_Type5_0 << std::endl;
    outfile << "Type5_1 = " << Result_term_Type5_1 << std::endl;
    outfile << "Type5_2 = " << Result_term_Type5_2 << std::endl;
    outfile << "Type5_3 = " << Result_term_Type5_3 << std::endl;
    outfile << "Type6_0 = " << Result_term_Type6_0 << std::endl;
    outfile << "Type6_1 = " << Result_term_Type6_1 << std::endl;
    outfile << "Type7_0 = " << Result_term_Type7_0 << std::endl;
    outfile << "Type7_1 = " << Result_term_Type7_1 << std::endl;
    outfile << "Type7_2 = " << Result_term_Type7_2 << std::endl;
    outfile << "Type7_3 = " << Result_term_Type7_3 << std::endl;
    outfile << "Type8_0 = " << Result_term_Type8_0 << std::endl;
    outfile << "Type8_1 = " << Result_term_Type8_1 << std::endl;
    outfile << "Type9_0 = " << Result_term_Type9_0 << std::endl;
    outfile << "Type9_1 = " << Result_term_Type9_1 << std::endl;
    outfile << "Type9_2 = " << Result_term_Type9_2 << std::endl;
    outfile << "Type9_3 = " << Result_term_Type9_3 << std::endl;
    outfile << "// ============ DUPLICATE DIAGRAMS (x1 <-> x2) ============" << std::endl;
    outfile << "Type10_4 = " << Result_term_Type10_0 << std::endl;  // x1 <-> x2 of Type10_0
    outfile << "Type10_5 = " << Result_term_Type10_2 << std::endl;  // x1 <-> x2 of Type10_2
    outfile << "Type10_6 = " << Result_term_Type10_1 << std::endl;  // x1 <-> x2 of Type10_1
    outfile << "Type10_7 = " << Result_term_Type10_3 << std::endl;  // x1 <-> x2 of Type10_3
    outfile << "Type11_2 = " << Result_term_Type11_1 << std::endl;  // x1 <-> x2 of Type11_1
    outfile << "Type11_3 = " << Result_term_Type11_0 << std::endl;  // x1 <-> x2 of Type11_0
    outfile << "Type12_2 = " << Result_term_Type12_1 << std::endl;  // x1 <-> x2 of Type12_1
    outfile << "Type12_3 = " << Result_term_Type12_0 << std::endl;  // x1 <-> x2 of Type12_0
    outfile << "Type1_4 = " << Result_term_Type1_0 << std::endl;  // x1 <-> x2 of Type1_0
    outfile << "Type1_5 = " << Result_term_Type1_1 << std::endl;  // x1 <-> x2 of Type1_1
    outfile << "Type1_6 = " << Result_term_Type1_2 << std::endl;  // x1 <-> x2 of Type1_2
    outfile << "Type1_7 = " << Result_term_Type1_3 << std::endl;  // x1 <-> x2 of Type1_3
    outfile << "Type2_4 = " << Result_term_Type2_0 << std::endl;  // x1 <-> x2 of Type2_0
    outfile << "Type2_5 = " << Result_term_Type2_1 << std::endl;  // x1 <-> x2 of Type2_1
    outfile << "Type2_6 = " << Result_term_Type2_2 << std::endl;  // x1 <-> x2 of Type2_2
    outfile << "Type2_7 = " << Result_term_Type2_3 << std::endl;  // x1 <-> x2 of Type2_3
    outfile << "Type3_4 = " << Result_term_Type3_0 << std::endl;  // x1 <-> x2 of Type3_0
    outfile << "Type3_5 = " << Result_term_Type3_1 << std::endl;  // x1 <-> x2 of Type3_1
    outfile << "Type3_6 = " << Result_term_Type3_2 << std::endl;  // x1 <-> x2 of Type3_2
    outfile << "Type3_7 = " << Result_term_Type3_3 << std::endl;  // x1 <-> x2 of Type3_3
    outfile << "Type4_4 = " << Result_term_Type4_0 << std::endl;  // x1 <-> x2 of Type4_0
    outfile << "Type4_5 = " << Result_term_Type4_1 << std::endl;  // x1 <-> x2 of Type4_1
    outfile << "Type4_6 = " << Result_term_Type4_2 << std::endl;  // x1 <-> x2 of Type4_2
    outfile << "Type4_7 = " << Result_term_Type4_3 << std::endl;  // x1 <-> x2 of Type4_3
    outfile << "Type5_4 = " << Result_term_Type5_1 << std::endl;  // x1 <-> x2 of Type5_1
    outfile << "Type5_5 = " << Result_term_Type5_3 << std::endl;  // x1 <-> x2 of Type5_3
    outfile << "Type5_6 = " << Result_term_Type5_0 << std::endl;  // x1 <-> x2 of Type5_0
    outfile << "Type5_7 = " << Result_term_Type5_2 << std::endl;  // x1 <-> x2 of Type5_2
    outfile << "Type6_2 = " << Result_term_Type6_1 << std::endl;  // x1 <-> x2 of Type6_1
    outfile << "Type6_3 = " << Result_term_Type6_0 << std::endl;  // x1 <-> x2 of Type6_0
    outfile << "Type7_4 = " << Result_term_Type7_3 << std::endl;  // x1 <-> x2 of Type7_3
    outfile << "Type7_5 = " << Result_term_Type7_1 << std::endl;  // x1 <-> x2 of Type7_1
    outfile << "Type7_6 = " << Result_term_Type7_2 << std::endl;  // x1 <-> x2 of Type7_2
    outfile << "Type7_7 = " << Result_term_Type7_0 << std::endl;  // x1 <-> x2 of Type7_0
    outfile << "Type8_2 = " << Result_term_Type8_1 << std::endl;  // x1 <-> x2 of Type8_1
    outfile << "Type8_3 = " << Result_term_Type8_0 << std::endl;  // x1 <-> x2 of Type8_0
    outfile << "Type9_4 = " << Result_term_Type9_2 << std::endl;  // x1 <-> x2 of Type9_2
    outfile << "Type9_5 = " << Result_term_Type9_0 << std::endl;  // x1 <-> x2 of Type9_0
    outfile << "Type9_6 = " << Result_term_Type9_3 << std::endl;  // x1 <-> x2 of Type9_3
    outfile << "Type9_7 = " << Result_term_Type9_1 << std::endl;  // x1 <-> x2 of Type9_1
    outfile.close();
  }

// epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();

  return EXIT_SUCCESS;
}
