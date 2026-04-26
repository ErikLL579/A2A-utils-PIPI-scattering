//#include <Grid/Hadrons/Global.hpp>
#include <Grid/Grid_Eigen_Tensor.h>


NAMESPACE_BEGIN(Grid);
using namespace std;

// From /Grid/qcd/utils/A2Autils.h
template <typename FImpl>
class PipiA2Autils
{
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  typedef iSpinMatrix<vector_type> SpinMatrix_v;
  typedef iSpinMatrix<scalar_type> SpinMatrix_s;
  typedef iSinglet<vector_type> Scalar_v;
  typedef iSinglet<scalar_type> Scalar_s;

  typedef iSpinColourMatrix<vector_type> SpinColourMatrix_v;

  // apply gamma_mu at the end

// ////////////////////////////////////////////////////////////
//     A2A contraction utilities for pipi scattering
// ////////////////////////////////////////////////////////////
// Fig 1. https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ellundstrum/K_to_pipi/pipi-scattering/pi_pi_scattering_A2A.pdf
// Eq. 5 & 6
// D({p_i}, {t_i}) = \sum_{z1,z2} [\sum_{i1, i2, i3} PI_{i3i1}(p_1, t_1) PI_{i1i2}(p_2, t_2) <w_{i2}(z1)| G |v_{i3}(z1)> ]
//                               *[\sum_{j1, j2, j3} PI_{j3j1}(p_3, t_3) PI_{j1j2}(p_4, t_4) <w_{j2}(z2)| G |v_{j3}(z2)> ]
//
// Function below executes series of matrix-vector operations
//
// y^{i1}(p, z1, t) = \sum_{i2} PI_{i1i2}(p, t) <w_{i2}(z1)| G
//
//
// ////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////



  // returns Fermionfield y with one outstanding A2A index i1
  template <typename TensorType>
  static void ContractMesonFieldAndVector(FermionField *y_i1,
                         const TensorType &meson_field_ij,
                         const FermionField *wj);


/*
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// See for example Eq. (A9)
// https://arxiv.org/pdf/2301.09286 (Masaaki pi-pi scattering PBC)
//
// Operation:
// Tr(Pi(p_1, t_1) \cdot Pi(p_2, t_2\) =  \sum_{i,j} Pi_{ij}(p_1, t_1) Pi_{ji}(p_2,t_2))
//
// for 0 < i,j < 2768
//
// Manageable on CPU but for many momenta p and times t it is better to do batched BLAS jobs
// (gotta go fast)
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
*/



// Generic Mesonfield type:
// Eigen::Tensor<ComplexD,4, Eigen::RowMajor> Mpp(momenta.size(),Nt,Nmodes,Nmodes);

  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  static void MesonField_MesonField_connected(TensorType_mesonfield &Mesonfield,
                                              TensorType_TraceMomTime &Result_round_1,
                                              TensorType_TraceMomTime &Result_round_2, // these might have to be changed for different lengths
                                              int level_1_contractions,
					      vector<int> & buffer_flag_A, // True when evaluaring things like Prod_Pi2 * Prod_Pi4 in the second level, false for Prod_Pi1 * Pi(k, tsrc)
					      vector<int> & buffer_flag_B,
                                              vector<int> &A_vector_contractions,
                                              vector<int> &B_vector_contractions,
                                              vector<int> &C_vector_contractions);


  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  static void MesonField_MesonField_disconnected(TensorType_mesonfield &Mesonfield,
                                              TensorType_TraceMomTime &Result,
 					      int level_1_contractions,
                                              vector<int> &A_vector_contractions,
                                              vector<int> &B_vector_contractions,
                                              vector<int> &C_vector_contractions);


  // may swap to using these
  
/*

  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  static void MesonField_to_device(TensorType_mesonfield &Mesonfield,
                                   deviceVector<ComplexD> &A);

*/
  /*
  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  static void MesonField_from_device(TensorType_mesonfield &Mesonfield,
                                     TensorType_TraceMomTime &Result,
                                     deviceVector<ComplexD* > &Cs);     


  static void MesonField_contract(deviceVector<ComplexD* > &As,       
                                  deviceVector<ComplexD* > &Bs,       
                                  deviceVector<ComplexD* > &Cs);

  */


/*
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// FFT functionality  
// 
// See https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ellundstrum/K_to_pipi/pipi-scattering/pi_pi_scattering_A2A.pdf
// FFT strategy 1: Appendix B
// \sum_i <A_i (z1)| \gamma_mu |B_i (z1)> = \phi_mu (z1)
//
//  
//
//
// FFT strategy 2: Appendix C
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
*/

// may want to add functionality to the following that splits up the mode indices by MPI rank

// \sum_i <A_i (z1)| \gamma_mu |B_i (z1)> = \phi_mu (z1)
                            // usage: &phi_mu[0]
  static void FFT_type1_prod( ComplexField *phi_mu,
  		              const FermionField *Ai,
		              const FermionField *Bj,
                              const int Nmodes,
		              const std::vector<Gamma::Algebra> gammas);



// FFT_type1_convolve 
 static void FFT_type1_convolve(ComplexD &Result,
                                const ComplexField *phi1_mu,
                                const ComplexField *phi2_nu,
                                const ComplexField *phtn_prop_mu_nu);


// FFT_type2_prod_and_convolve

static void FFT_type2_contract_convolve(ComplexD &Result,
                                        const FermionField *Ai,
                                        const FermionField *Bi,
                                        const FermionField *wi,
                                        const FermionField *vi,
                                        const int Nmodes,
                                        const ComplexField *phtn_prop_mu_nu,
                                        const std::vector<Gamma::Algebra> gammas);

// Batched-FFT version: packs all 4 gamma components into a single
// 4-component lattice field so Grid's FFT processes them in one cufftPlanMany call.
static void FFT_type2_contract_convolve_claude(ComplexD &Result,
                                        const FermionField *Ai,
                                        const FermionField *Bi,
                                        const FermionField *wi,
                                        const FermionField *vi,
                                        const int Nmodes,
                                        const ComplexField *phtn_prop_mu_nu,
                                        const std::vector<Gamma::Algebra> gammas);

// Chunked batched-FFT + Parseval version: batches BATCH i3 values into a single
// 8*BATCH-component FFT call (4 gammas * 2 bilinears * BATCH i3 values).
// Eliminates backward FFTs entirely via Parseval's theorem.
static void FFT_type2_contract_convolve_claude_level2(ComplexD &Result,
                                        const FermionField *Ai,
                                        const FermionField *Bi,
                                        const FermionField *wi,
                                        const FermionField *vi,
                                        const int Nmodes,
                                        const ComplexField *phtn_prop_mu_nu,
                                        const std::vector<Gamma::Algebra> gammas);

};

/////////////////////////////////////////////////////////////////////////

// the below already enters from the normal A2Autils header
//const int A2Ablocking=8;

template<typename vtype> using iVecSpinMatrixF = iVector<iMatrix<iScalar<vtype>, Ns>, A2Ablocking>;
typedef iVecSpinMatrixF<ComplexF  >           VecSpinMatrixF;
typedef iVecSpinMatrixF<vComplexF >           vVecSpinMatrixF;
typedef Lattice<vVecSpinMatrixF>              LatticeVecSpinMatrixF;

template<typename vtype> using iVecComplexF = iVector<iScalar<iScalar<vtype> >, A2Ablocking>;
typedef iVecComplexF<ComplexF  >           VecComplexF;
typedef iVecComplexF<vComplexF >           vVecComplexF;
typedef Lattice<vVecComplexF>              LatticeVecComplexF;

// 4-component complex vector for batching Ngamma=4 FFTs into one cufftPlanMany call
template<typename vtype> using iVec4Complex = iVector<iScalar<iScalar<vtype> >, 4>;
typedef iVec4Complex<ComplexF  >            Vec4Complex;
typedef iVec4Complex<vComplexF >            vVec4Complex;
typedef Lattice<vVec4Complex>               LatticeVec4Complex;

// Batch size for chunked FFT: process FFT_BATCH i3 values per FFT call.
// 8 components per i3 (4 gammas x 2 bilinears). Tune based on GPU memory.
// 24^3x64 lattice: ~3.5 MB/ComplexField/rank, so 8*BATCH * 3.5 MB per packed field.
// BATCH=64 => ~1.75 GB per packed field. Safe on 40 GB A100.
const int FFT_BATCH = 5;
template<typename vtype> using iBatchComplex = iVector<iScalar<iScalar<vtype> >, 8*FFT_BATCH>;
typedef iBatchComplex<ComplexF  >            BatchComplex;
typedef iBatchComplex<vComplexF >            vBatchComplex;
typedef Lattice<vBatchComplex>               LatticeBatchComplex;

#define A2A_GPU_KERNELS


// probably I want to define this with another argument gammas
template <class FImpl>
template <typename TensorType>
void PipiA2Autils<FImpl>::ContractMesonFieldAndVector(FermionField *y_i1,
                                                      const TensorType &meson_field_ij,
                                                      const FermionField *wj)
{

  const int block=A2Ablocking;
 // const int block = 8;
  typedef typename FImpl::SiteSpinor vobj;

  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  const int Nmodes = meson_field_ij.dimension(0);

  /*
  // check that the meson field is what I expect
  if (Nmodes != meson_field_ij.dimension(1)) {
    std::cout << GridLogMessage << "Error in ContractMesonFieldAndVector: Dimension mismatch!" << std::endl;
    std::cout << GridLogMessage << "Dimension 3 (Lmodes): " << Nmodes << std::endl;
    std::cout << GridLogMessage << "Dimension 4 (Rmodes): " << meson_field_ij.dimension(1) << std::endl;
    GRID_ASSERT(0); // Force the crash here
  }
 */

  GridBase *grid = wj[0].Grid();
  const int Nsimd = grid->Nsimd();

  //int Nt     = grid->GlobalDimensions()[orthogdim];
  // int Ngamma = gammas.size();

  LatticeVecSpinMatrixF SpinMat(grid);

  double t_view, t_gamma, t_kernel, t_momproj;
  t_view=0;
  t_gamma=0;
  t_kernel=0;
  t_momproj=0;

  std::vector<VecSpinMatrixF> sliced;

  // meson field vector on device
  static deviceVector<ComplexF> PI(Nmodes * Nmodes);

  // placeholder until I sort out DTYPE for Masaaki's meson fields
  // DTYPE flat_meson_field;

  // copy host to device and get memory address (and changing to single precision)
  //acceleratorCopyToDevice((void *)&flat_meson_field[0] ,(void *)&PI[0],Nmodes * Nmodes* sizeof(ComplexD));

  // works for single A2A vector stripped to NmodesxNmodes basis
  acceleratorCopyToDevice(meson_field_ij.data() ,(void *)&PI[0],Nmodes * Nmodes* sizeof(ComplexF));

//const ComplexD *PI_ptr = &PI[0]; 

  ComplexF *PI_ptr;
  acceleratorPut(PI_ptr, &PI[0]);

  for(int i=0; i<Nmodes; i++) {

    autoView(y_i,y_i1[i],AcceleratorWrite);

    for(int jo=0;jo<Nmodes;jo+=block){
      for(int j=jo;j<MIN(Nmodes,jo+block);j++){
        int jj=j%block;

        autoView(w,wj[j],AcceleratorRead); // create vector of views

        //ComplexF Pi_val = PI_ptr[i * Nmodes + j];

        accelerator_for(ss,grid->oSites(),(size_t)Nsimd,{
            // contract y_i = PI_ij w_j
            auto y_local = y_i(ss);
            auto w_local = w(ss);

            // y_vector must be zero to begin
            y_local = y_local + PI_ptr[i * Nmodes + j] * w_local;

            coalescedWrite(y_i[ss], y_local);
          });

      }
    }

  }

  std::cout << GridLogMessage<<"\\\\\\\\\\\\\\\\\\ MATRIX-VECTOR OPERATION COMPLETED \\\\\\\\\\\\\\\\\\\\\\" <<std::endl;

// actually just do these with the photon convolutions
// define vector y_mu (also need to add to function argument)
// vector<FermionField> y_mu(gammas.size()); 
/*
  if(gammas.size() > 0) {
   for(int mu=0; i<gammas.size(); mu++) {
     y_[mu] = y_i1 * Gamma(gammas[mu])
   }
  }

*/
};

/* // not needed atm
  template<class FImpl>
  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  void PipiA2Autils<FImpl>::MesonField_to_device(TensorType_mesonfield &Mesonfield,
                                                 deviceVector<ComplexD> &A)       
{

  GridBLAS blas;
    
  int num_momenta = Mesonfield.dimension(0);
  int timeslices  = Mesonfield.dimension(1);
  int Nmodes = Mesonfield.dimension(2);
    
  // make sure mesonfield is what I expect
  GRID_ASSERT(Nmodes == Mesonfield.dimension(3));

  const int max_batch_size = 100;
  
  // set up memory on device
  // THIS LINE WONT WORK BUT I AM NOT FIXING IT YET
  const int Ncomplex = Nmodes * Nmodes * contractions;

  // deviceVector<ComplexD> A(Ncomplex); // input vectors

  // need to only upload enough A fields to actually fill this out...
  acceleratorCopyToDevice(Mesonfield.data(), &A[0], Nmodes * Nmodes * contractions * sizeof(ComplexD));

};
*/


  template<class FImpl>
  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  void PipiA2Autils<FImpl>::MesonField_MesonField_disconnected(TensorType_mesonfield &Mesonfield,
                                                            TensorType_TraceMomTime &Result,
                                                            int level_1_contractions,
                                                            vector<int> &A_vector_contractions,
                                                            vector<int> &B_vector_contractions,
                                                            vector<int> &C_vector_contractions)
{
    const int block=A2Ablocking;
    typedef typename vobj::scalar_object sobj;
    typedef typename vobj::scalar_type scalar_type;
    typedef typename vobj::vector_type vector_type;

    GridBLAS blas;

    int num_momenta = Mesonfield.dimension(0);
    int timeslices  = Mesonfield.dimension(1);
    int Nmodes = Mesonfield.dimension(2);

    // make sure mesonfield is what I expect
    GRID_ASSERT(Nmodes == Mesonfield.dimension(3));

    // need to write something to determine batch size based on available device memory
    // ex single Meson field: 2700 * 2700 * 8 ~ 5.5MB
    // Perlmutter A100 has 40GB memory => 600 matrices = ~ 32 GB with 8GB for system

    const int max_batch_size = 100;
    const int contractions = A_vector_contractions.size();

    // check that contraction vector are the same size
    GRID_ASSERT(contractions == B_vector_contractions.size());
    GRID_ASSERT(contractions == C_vector_contractions.size());
    GRID_ASSERT(contractions < max_batch_size);

    // total number of matrices
    // timeslices * num_momenta * 2 mom orientations
    // ex: 24^3 x 64 ensemble: 64 * 4 * 2 = 312 pion mesonfields per config

    // set up memory on device
    const int Ncomplex = Nmodes * Nmodes * contractions;

    deviceVector<ComplexF> A(Ncomplex); // input vectors
    deviceVector<ComplexF> C(Ncomplex); // result of matrix operations on device

    // need to parse the input Eigen matrix appropriately
    // compute DC pieces at all tsrc

    // Input mesonfield Mpp(i, j, k, l)
    // use Mpp.data() to copy to device => Offset = i*(Nt*Nmodes*Nmodes) + j*(Nmodes*Nmodes) + k*(Nmodes) + l

    // need to only upload enough A fields to actually fill this out...
    acceleratorCopyToDevice(Mesonfield.data(), &A[0], Nmodes * Nmodes * contractions * sizeof(ComplexF));
  
    // wrapping this bit in {} so that As, Bs and Cs get automatically deallocated when they are no longer needed.
    {
    deviceVector<ComplexF* > As(contractions);
    // Same matrices as in As but in the order necessary for the contraction
    deviceVector<ComplexF* > Bs(contractions);
    deviceVector<ComplexF* > Cs(contractions);

    // vector of which matrices to contract

    for(int b=0; b<contractions; b++) {
      ComplexF *ptr;
      
      // this needs to be modified in the case that I want to use matrices more than once (which I do)
      // probably need another vector to organize these 
      ptr = &A[A_vector_contractions[b] * Nmodes * Nmodes];
      acceleratorPut(As[b], ptr);
      
      // this is where I need the contractions necessary to craft the appropriate B vector
      ptr = &A[B_vector_contractions[b] * Nmodes * Nmodes];
      acceleratorPut(Bs[b], ptr);

      ptr = &C[C_vector_contractions[b] * Nmodes * Nmodes];
      acceleratorPut(Cs[b], ptr);
    }

    ComplexF alpha(1.0);
    ComplexF beta(0.0);
    RealD flops = 8.0 * Nmodes * Nmodes * Nmodes * contractions;

    RealD t0 = usecond();

    // perform the matrix multiplication
    // (check that the matrices are transposed correctly)
    blas.gemmBatched(Nmodes, Nmodes, Nmodes, alpha, As, Bs, beta, Cs);
    blas.synchronise();
    

    RealD t1 = usecond();
    flops = flops / (t1 - t0) / 1.e3;

    cout << GridLogMessage << "=================================================== " << endl;
    cout << GridLogMessage << "=================================================== " << endl;
    cout << GridLogMessage << "TOTAL TIME BATCHED GEMM = " << (t1 - t0) / 1.e3 << endl;
    cout << GridLogMessage << "FLOPS = "  <<  flops << " Gflop/s" <<  endl; 
    cout << GridLogMessage << "=================================================== " << endl;
    cout << GridLogMessage << "=================================================== " << endl;

    // write out the DC pieces for use in zeroth order diagrams and EM corrections to DC diagram
    cout << GridLogMessage << "COPYING RESULTS TO HOST" << endl;

    //Eigen::Tensor<ComplexD,4, Eigen::RowMajor> c(momenta.size(),Nt,Nmodes,Nmodes);
    acceleratorCopyFromDevice(&C[0], Result.data(), Nmodes * Nmodes * sizeof(ComplexF) * contractions);
    }
};




  // edit so that the shape of the input meson field is just 3 indices
  template<class FImpl>
  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  void PipiA2Autils<FImpl>::MesonField_MesonField_connected(TensorType_mesonfield &Mesonfield,
                                                            TensorType_TraceMomTime &Result_round_1,
                                                            TensorType_TraceMomTime &Result_round_2, // these might have to be changed for different lengths
                                                            int level_1_contractions,
							    vector<int> &buffer_flag_A,
                                                            vector<int> &buffer_flag_B,
                                                            vector<int> &A_vector_contractions,
                                                            vector<int> &B_vector_contractions,
                                                            vector<int> &C_vector_contractions)
{ 
    const int block=A2Ablocking;
    typedef typename vobj::scalar_object sobj;
    typedef typename vobj::scalar_type scalar_type;
    typedef typename vobj::vector_type vector_type;
    
    GridBLAS blas;
    
    int timeslices  = Mesonfield.dimension(1);
    int Nmodes = Mesonfield.dimension(2);
    
    // total number of pion meson fields stored on device
    int num_matrices = timeslices;

    // make sure mesonfield is what I expect  
    GRID_ASSERT(Nmodes == Mesonfield.dimension(2));
    
    // need to write something to determine batch size based on available device memory
    // ex single Meson field: 2700 * 2700 * 8 ~ 5.5MB
    // Perlmutter A100 has 40GB memory => 600 matrices = ~ 32 GB with 8GB for system
      
    const int max_batch_size = 100;
    const int contractions = A_vector_contractions.size();
     
    // check that contraction vector are the same size
    GRID_ASSERT(contractions == B_vector_contractions.size());
    GRID_ASSERT(contractions == C_vector_contractions.size());
    GRID_ASSERT(contractions < max_batch_size);
      
    // total number of matrices
    // timeslices * num_momenta * 2 mom orientations
    // ex: 24^3 x 64 ensemble: 64 * 4 * 2 = 312 pion mesonfields per config
  
    // set up memory on device
    const int Ncomplex = Nmodes * Nmodes * num_matrices;
                                                            
    deviceVector<ComplexF> A(Ncomplex); // input vectors, holds all pion meson field matrices

    deviceVector<ComplexF> C(Nmodes * Nmodes * level_1_contractions); // result of first round of contractions on device

    // Input mesonfield Mpp(i, j, k, l)
    // use Mpp.data() to copy to device => Offset = i*(Nt*Nmodes*Nmodes) + j*(Nmodes*Nmodes) + k*(Nmodes) + l
    
    // need to only upload enough A fields to actually fill this out...
    acceleratorCopyToDevice(Mesonfield.data(), &A[0], Nmodes * Nmodes * num_matrices * sizeof(ComplexF));
   
    // wrapping this part in {} so that As1, Bs1 and Cs1 are automatically deallocaed when I finish with them
    { 
    // need different lengths in levels one and two
    deviceVector<ComplexF* > As1(level_1_contractions);
    // Same matrices as in As but in the order necessary for the contraction
    deviceVector<ComplexF* > Bs1(level_1_contractions);
    deviceVector<ComplexF* > Cs1(level_1_contractions);

 // vector of which matrices to contract
    for(int b=0; b<level_1_contractions; b++) {  
      ComplexF *ptr;
    
      // this needs to be modified in the case that I want to use matrices more than once (which I do)
      // probably need another vector to organize these
      ptr = &A[A_vector_contractions[b] * Nmodes * Nmodes];
      acceleratorPut(As1[b], ptr);             
                                              
      // this is where I need the contractions necessary to craft the appropriate B vector
      ptr = &A[B_vector_contractions[b] * Nmodes * Nmodes];
      acceleratorPut(Bs1[b], ptr);
                                                            
      ptr = &C[C_vector_contractions[b] * Nmodes * Nmodes];
      acceleratorPut(Cs1[b], ptr);
    }
                                              
    ComplexF alpha(1.0);
    ComplexF beta(0.0);
    RealD flops = 8.0 * Nmodes * Nmodes * Nmodes * level_1_contractions;
    
    RealD t0 = usecond();
    
    // perform the matrix multiplication
    // (check that the matrices are transposed correctly)
    blas.gemmBatched(Nmodes, Nmodes, Nmodes, alpha, As1, Bs1, beta, Cs1);
    blas.synchronise();
      

    RealD t1 = usecond();
    flops = flops / (t1 - t0) / 1.e3;

    cout << GridLogMessage << "=================================================== " << endl;
    cout << GridLogMessage << "=================================================== " << endl;
    cout << GridLogMessage << "TOTAL TIME BATCHED GEMM = " << (t1 - t0) / 1.e3 << endl;
    cout << GridLogMessage << "FLOPS = "  <<  flops << " Gflop/s" <<  endl;
    cout << GridLogMessage << "=================================================== " << endl;
    cout << GridLogMessage << "=================================================== " << endl;

    acceleratorCopyFromDevice(&C[0], Result_round_1.data(), Nmodes * Nmodes * sizeof(ComplexF) * (level_1_contractions) );

    // end variable scope
    }


    // ========================================================
    // second round of contractions
    // ========================================================

    // wrap the below in {} so that the device deallocates memory when finished
    {
    // need different lengths in levels one and two
    deviceVector<ComplexF* > As2(contractions - level_1_contractions); // again these are the source meson fields
    deviceVector<ComplexF* > Cs2(contractions - level_1_contractions); // these are the results from the first round of matrices

    deviceVector<ComplexF> results(Nmodes * Nmodes * (contractions - level_1_contractions) ); // needed to hold results in second round
    deviceVector<ComplexF* > level_2_result_s(contractions - level_1_contractions); // pntrs to results


    // issue is that the below could be either from A or C...
    for(int b=0; b<contractions- level_1_contractions; b++) {
      ComplexF *ptr;

      // source meson fields
      if (buffer_flag_A[b] == 0) ptr = &A[A_vector_contractions[b+ level_1_contractions] * Nmodes * Nmodes]; // ex: Prod_Pi2 * Pi(k, tsrc)
 
      else ptr = &C[A_vector_contractions[b + level_1_contractions] * Nmodes * Nmodes]; // ex: Prod_Pi2 * Prod_Pi3

      acceleratorPut(As2[b], ptr);
    
      // results from first round of contractions
      if (buffer_flag_B[b] == 0) ptr = &C[B_vector_contractions[b + level_1_contractions] * Nmodes * Nmodes];

      else ptr = &A[B_vector_contractions[b + level_1_contractions] * Nmodes * Nmodes];

      acceleratorPut(Cs2[b], ptr);
  
      // result from second round of contractions
      ptr = &results[C_vector_contractions[b + level_1_contractions] * Nmodes * Nmodes];    
      acceleratorPut(level_2_result_s[b], ptr);
    }

    RealD t0 = usecond();
    ComplexF alpha(1.0);
    ComplexF beta(0.0);

    // (check that the matrices are transposed correctly)
    blas.gemmBatched(Nmodes, Nmodes, Nmodes, alpha, As2, Cs2, beta, level_2_result_s);
    blas.synchronise();

    RealD t1 = usecond();
    RealD flops = 8.0 * Nmodes * Nmodes * Nmodes * (contractions - level_1_contractions);    
    flops = flops / (t1 - t0) / 1.e3;

    cout << GridLogMessage << "=================================================== " << endl;
    cout << GridLogMessage << "=================================================== " << endl;
    cout << GridLogMessage << "TOTAL TIME BATCHED GEMM = " << (t1 - t0) / 1.e3 << endl;
    cout << GridLogMessage << "FLOPS = "  <<  flops << " Gflop/s" <<  endl;
    cout << GridLogMessage << "=================================================== " << endl;
    cout << GridLogMessage << "=================================================== " << endl;


    cout << GridLogMessage << "COPYING RESULTS TO HOST" << endl;

    //Eigen::Tensor<ComplexD,4, Eigen::RowMajor> c(momenta.size(),Nt,Nmodes,Nmodes);
    // need both products of round 1 and round 2
    acceleratorCopyFromDevice(&results[0], Result_round_2.data(), Nmodes * Nmodes * sizeof(ComplexF) * (contractions - level_1_contractions) );
    }
};




template <class FImpl>                   // should be std vector but can just point to first mem location
void PipiA2Autils<FImpl>::FFT_type1_prod( ComplexField *phi_mu,
                                          const FermionField *Ai,
                                          const FermionField *Bj,
                                          const int Nmodes,
                                          const std::vector<Gamma::Algebra> gammas)
{

  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  const int block=A2Ablocking;
 
  GridBase *grid = Ai[0].Grid();
  const int Nsimd = grid->Nsimd();
  const int    Nd = grid->_ndimension;

  int Ngamma = gammas.size();

  
  // copy gamma matrices to device (for manual implementation)
  // static deviceVector<ComplexD> gamma(Ns * Ns * Ng);
  // acceleratorCopyToDevice(gammas.data() ,(void *)gamma[0],Ng * Ns * Ns * sizeof(ComplexD));


  cout << GridLogMessage << "============================" << endl;
  cout << GridLogMessage << "============================" << endl; 
  cout << GridLogMessage << "FFT TYPE 1 CONTRACTION START" << endl;
  cout << GridLogMessage << "============================" << endl;
  cout << GridLogMessage << "============================" << endl;

  // adjust this to be dynamic based on input
  //const int Nmodes = 10;
  // number of blocks (adjust to keep GPU memory ~80% full)
  // const int blocks = (Nmodes + block - 1 ) / block ; 

  for(int j=0; j<Nmodes; j++) {
    for(int Ng=0; Ng<Ngamma; Ng++) {
      FermionField tmp = Gamma(gammas[Ng]) * Bj[j];
      phi_mu[Ng] = phi_mu[Ng] + localInnerProduct(Ai[j], tmp);
    }
  }
  
  /* Manual implementation
  for(int j=0; j<Nmodes; j++) {
      autoView(A, Ai[j], AcceleratorRead);
      autoView(B, Bj[j], AcceleratorRead);

      for(int Ng=0; Ng<Ngamma; Ng++) {
        autoView(Complex_v, phi_mu[Ng], AcceleratorWrite);      

        accelerator_for(ss, grid->oSites(), (size_t)Nsimd,{ 
          // one should be conjugate or this is implicit in A2A vector as saved?
          auto left = A(ss);
          auto right = B(ss);

          auto vv = Complex_v(ss);

          for(int s1=0, s1<Ns; s1++) {
            for(int s2=0; s2<Ns; s2++) {
              // sum over final index = colour contraction
              // check left and right on the below
              vv = vv + left()(s2)(0) * gamma[Ng * Ns * Ns + s2 * Ns + s1] * right()(s1)(0)
                      + left()(s2)(1) * gamma[Ng * Ns * Ns + s2 * Ns + s1] * right()(s1)(1)
                      + left()(s2)(2) * gamma[Ng * Ns * Ns + s2 * Ns + s1] * right()(s1)(2);
          }
          coalescedWrite(Complex_v[ss], vv);
        });
      }

  }
  */

  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "FFT TYPE 1 CONTRACTION COMPLETE" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;

};



template <class FImpl>
void PipiA2Autils<FImpl>::FFT_type1_convolve(ComplexD &Result,
                                             const ComplexField *phi1_mu,
                                             const ComplexField *phi2_nu,
                                             const ComplexField *phtn_prop_mu_nu)
{
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "BEGIN: FFT TYPE 1 CONVOLUTION" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;

  GridBase *grid = phi1_mu[0].Grid(); 

  int Nd = grid->Dimensions();

  vector<ComplexField> tilde_phi1_mu(Nd, grid);
  vector<ComplexField> tilde_phi_phtn_nu(Nd, grid);
  for (int rho=0; rho<Nd; rho++) tilde_phi_phtn_nu[rho] = Zero();

  vector<ComplexField> phi_phtn_nu(Nd, grid);

  //FFT theFFT(&grid);
  FFT theFFT(dynamic_cast<GridCartesian *>(grid));

  for(int mu=0; mu<Nd; mu++) theFFT.FFT_all_dim(tilde_phi1_mu[mu], phi1_mu[mu], FFT::forward);


  for(int nu=0; nu<Nd; nu++) {
    //for(int mu=0; mu<Nd; mu++) {
    //  tilde_phi_phtn_nu[nu] = tilde_phi_phtn_nu[nu] + tilde_phi1_mu[mu] *  phtn_prop_mu_nu[nu*Nd + mu];
    //}
    tilde_phi_phtn_nu[nu] = tilde_phi1_mu[0] *  phtn_prop_mu_nu[nu*Nd + 0]
                          + tilde_phi1_mu[1] *  phtn_prop_mu_nu[nu*Nd + 1]
                          + tilde_phi1_mu[2] *  phtn_prop_mu_nu[nu*Nd + 2]
                          + tilde_phi1_mu[3] *  phtn_prop_mu_nu[nu*Nd + 3];
  }

  for(int nu=0; nu<Nd; nu++) {
    theFFT.FFT_all_dim(phi_phtn_nu[nu], tilde_phi_phtn_nu[nu], FFT::backward);
    
    Result += sum(phi_phtn_nu[nu] * phi2_nu[nu]);
  }

  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "COMPLETE: FFT TYPE 1 CONVOLUTION" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;

};



template <class FImpl>
void PipiA2Autils<FImpl>::FFT_type2_contract_convolve(ComplexD &Result,
                                                      const FermionField *Ai,
                                                      const FermionField *Bi,
                                                      const FermionField *wi,
                                                      const FermionField *vi,
					              const int Nmodes,
                                                      const ComplexField *phtn_prop_mu_nu,
                                                      const std::vector<Gamma::Algebra> gammas)
{

  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "BEGIN: FFT TYPE 2 CONV + CONT" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;


  GridBase *grid = Ai[0].Grid();
  
  int Ngamma = gammas.size();
  int Nd = grid->Dimensions();

  vector<ComplexField> g_i3i2_nu(Ngamma, grid);

  vector<ComplexField> Kg_i3i2_mu_phtn(Ngamma, grid);
  for (int rho=0; rho<Ngamma; rho++) Kg_i3i2_mu_phtn[rho] = Zero();

  FFT theFFT(dynamic_cast<GridCartesian *>(grid));

  for(int i2=0; i2<Nmodes; i2++){

    Vector<FermionField> v_g_nu(Ngamma, grid);
    for(int kk=0; kk<Ngamma; kk++) v_g_nu[kk] = Gamma(gammas[kk]) * vi[i2];

    for(int i3=0; i3<Nmodes; i3++) {

      for (int rho=0; rho<Ngamma; rho++) Kg_i3i2_mu_phtn[rho] = Zero();

      for(int nu=0; nu<Ngamma; nu++) {
        //FermionField tmp = Gamma(gammas[nu]) * vi[i2] ; // this can be moved outside of i3 loop
        g_i3i2_nu[nu] = localInnerProduct(wi[i3], v_g_nu[nu]);
        // using g_i3i2_nu as Kg_i3i2_nu for mem
        theFFT.FFT_all_dim(g_i3i2_nu[nu], g_i3i2_nu[nu], FFT::forward);

	for(int mu=0; mu<Ngamma; mu++) Kg_i3i2_mu_phtn[mu] = Kg_i3i2_mu_phtn[mu] + g_i3i2_nu[nu] * phtn_prop_mu_nu[nu*Nd + mu];
      }

      for(int mu=0; mu<Ngamma; mu++) {
        // using g_i3_i2_nu as G_i3i2_mu for mem
        theFFT.FFT_all_dim(g_i3i2_nu[mu], Kg_i3i2_mu_phtn[mu], FFT::backward);
        FermionField tmp = Gamma(gammas[mu]) * Bi[i3];
        ComplexField ttmp = localInnerProduct(Ai[i2], tmp);
        Result += innerProduct(g_i3i2_nu[mu], ttmp ) ;
      }


    }
  }

  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "COMPLETE: FFT TYPE 2 CONV + CONT" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;

};


// Batched-FFT version of FFT_type2_contract_convolve
// Packs all 4 gamma components into a LatticeVec4Complex so that
// Grid's FFT_all_dim calls cufftPlanMany with howmany=4*Nperp
// instead of howmany=Nperp, reducing 8 FFT calls to 2 per (i2,i3).
template <class FImpl>
void PipiA2Autils<FImpl>::FFT_type2_contract_convolve_claude(ComplexD &Result,
                                                      const FermionField *Ai,
                                                      const FermionField *Bi,
                                                      const FermionField *wi,
                                                      const FermionField *vi,
                                                      const int Nmodes,
                                                      const ComplexField *phtn_prop_mu_nu,
                                                      const std::vector<Gamma::Algebra> gammas)
{

  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "BEGIN: FFT TYPE 2 CONV + CONT (BATCHED)" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;

  GridBase *grid = Ai[0].Grid();

  int Ngamma = gammas.size();
  int Nd = grid->Dimensions();
  GRID_ASSERT(Ngamma == 4);

  // Packed 4-component fields for batched FFT
  LatticeVec4Complex g_packed(grid);
  LatticeVec4Complex g_packed_fft(grid);
  LatticeVec4Complex Kg_packed(grid);
  LatticeVec4Complex Kg_packed_ifft(grid);

  // Unpacked fields for the photon contraction and final combination
  vector<ComplexField> g_i3i2_nu(Ngamma, grid);
  vector<ComplexField> Kg_i3i2_mu(Ngamma, grid);

  FFT theFFT(dynamic_cast<GridCartesian *>(grid));

  GridStopWatch sw;
  double t_bilinear1 = 0, t_fft_fwd = 0, t_unpack = 0;
  double t_photon = 0, t_fft_bwd = 0, t_combine = 0;

  for(int i2=0; i2<Nmodes; i2++){

    // precompute gamma * v_{i2} for all nu (hoisted out of i3 loop)
    vector<FermionField> v_g_nu(Ngamma, grid);
    for(int nu=0; nu<Ngamma; nu++) v_g_nu[nu] = Gamma(gammas[nu]) * vi[i2];

    for(int i3=0; i3<Nmodes; i3++) {

      // ---- Forward FFT: pack all 4 bilinears, FFT once ----
      sw.Start();
      for(int nu=0; nu<Ngamma; nu++) {
        g_i3i2_nu[nu] = localInnerProduct(wi[i3], v_g_nu[nu]);
        PokeIndex<0>(g_packed, g_i3i2_nu[nu], nu);
      }
      sw.Stop(); t_bilinear1 += sw.Elapsed().count(); sw.Reset();

      sw.Start();
      theFFT.FFT_all_dim(g_packed_fft, g_packed, FFT::forward);
      sw.Stop(); t_fft_fwd += sw.Elapsed().count(); sw.Reset();

      // unpack FFT results
      sw.Start();
      for(int nu=0; nu<Ngamma; nu++) {
        g_i3i2_nu[nu] = PeekIndex<0>(g_packed_fft, nu);
      }
      sw.Stop(); t_unpack += sw.Elapsed().count(); sw.Reset();

      // ---- Photon propagator contraction in momentum space ----
      sw.Start();
      for(int mu=0; mu<Ngamma; mu++) {
        Kg_i3i2_mu[mu] = g_i3i2_nu[0] * phtn_prop_mu_nu[0*Nd + mu];
        for(int nu=1; nu<Ngamma; nu++) {
          Kg_i3i2_mu[mu] = Kg_i3i2_mu[mu] + g_i3i2_nu[nu] * phtn_prop_mu_nu[nu*Nd + mu];
        }
      }
      sw.Stop(); t_photon += sw.Elapsed().count(); sw.Reset();

      // ---- Backward FFT: pack all 4 Kg_mu, IFFT once ----
      sw.Start();
      for(int mu=0; mu<Ngamma; mu++) {
        PokeIndex<0>(Kg_packed, Kg_i3i2_mu[mu], mu);
      }

      theFFT.FFT_all_dim(Kg_packed_ifft, Kg_packed, FFT::backward);
      sw.Stop(); t_fft_bwd += sw.Elapsed().count(); sw.Reset();

      // ---- Combine with second bilinear ----
      sw.Start();
      for(int mu=0; mu<Ngamma; mu++) {
        ComplexField Kg_mu = PeekIndex<0>(Kg_packed_ifft, mu);
        FermionField tmp = Gamma(gammas[mu]) * Bi[i3];
        ComplexField ttmp = localInnerProduct(Ai[i2], tmp);
        Result += innerProduct(Kg_mu, ttmp);
      }
      sw.Stop(); t_combine += sw.Elapsed().count(); sw.Reset();

    }
  }

  double t_total = t_bilinear1 + t_fft_fwd + t_unpack + t_photon + t_fft_bwd + t_combine;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "FFT TYPE 2 TIMING BREAKDOWN" << endl;
  cout << GridLogMessage << "  Bilinear <w|g|v> + pack: " << t_bilinear1 << " s (" << 100*t_bilinear1/t_total << "%)" << endl;
  cout << GridLogMessage << "  FFT forward:             " << t_fft_fwd   << " s (" << 100*t_fft_fwd/t_total   << "%)" << endl;
  cout << GridLogMessage << "  Unpack:                  " << t_unpack    << " s (" << 100*t_unpack/t_total     << "%)" << endl;
  cout << GridLogMessage << "  Photon contraction:      " << t_photon    << " s (" << 100*t_photon/t_total     << "%)" << endl;
  cout << GridLogMessage << "  FFT backward:            " << t_fft_bwd   << " s (" << 100*t_fft_bwd/t_total   << "%)" << endl;
  cout << GridLogMessage << "  Combine <A|g|B> + sum:   " << t_combine   << " s (" << 100*t_combine/t_total   << "%)" << endl;
  cout << GridLogMessage << "  TOTAL:                   " << t_total     << " s" << endl;
  cout << GridLogMessage << "===============================" << endl;

};


// Chunked batched-FFT + Parseval version of FFT_type2_contract_convolve.
// Batches FFT_BATCH i3 values into one FFT call with 8*FFT_BATCH components
// (4 gammas × 2 bilinears × FFT_BATCH i3 values).
// Eliminates backward FFTs via Parseval: innerProduct(IFFT[Kg], h) = (1/V) * innerProduct(Kg, FFT[h]).
//
// Component layout in the packed field:
//   [0 .. 4*BATCH-1]         : first bilinears  g_nu^{i3}  = <w_{i3}|gamma_nu|v_{i2}>
//   [4*BATCH .. 8*BATCH-1]   : second bilinears h_mu^{i3}  = <A_{i2}|gamma_mu|B_{i3}>
// Within each group of 4: index = gamma index (nu or mu = 0,1,2,3)
template <class FImpl>
void PipiA2Autils<FImpl>::FFT_type2_contract_convolve_claude_level2(ComplexD &Result,
                                                      const FermionField *Ai,
                                                      const FermionField *Bi,
                                                      const FermionField *wi,
                                                      const FermionField *vi,
                                                      const int Nmodes,
                                                      const ComplexField *phtn_prop_mu_nu,
                                                      const std::vector<Gamma::Algebra> gammas)
{

  cout << GridLogMessage << "============================================" << endl;
  cout << GridLogMessage << "BEGIN: FFT TYPE 2 CONV + CONT (CHUNKED BATCH)" << endl;
  cout << GridLogMessage << "============================================" << endl;

  GridBase *grid = Ai[0].Grid();

  int Ngamma = gammas.size();
  int Nd = grid->Dimensions();
  GRID_ASSERT(Ngamma == 4);

  const int BATCH = FFT_BATCH;

  // Lattice volume for Parseval normalization
  // Grid forward FFT is unnormalized, backward divides by V.
  // Parseval: sum_x conj(f(x)) g(x) = (1/V) sum_k conj(f~(k)) g~(k)
  double vol = 1.0;
  for(int d=0; d<Nd; d++) vol *= grid->FullDimensions()[d];
  RealD inv_vol = 1.0 / vol;

  // Packed field: 8*BATCH components
  LatticeBatchComplex packed(grid);
  LatticeBatchComplex packed_fft(grid);

  // Temporary ComplexFields for unpacking and photon contraction
  vector<ComplexField> g_tilde_nu(Ngamma, grid);  // FFT of first bilinear per i3
  vector<ComplexField> Kg_mu(Ngamma, grid);       // after photon contraction per i3
  vector<ComplexField> h_tilde_mu(Ngamma, grid);  // FFT of second bilinear per i3

  FFT theFFT(dynamic_cast<GridCartesian *>(grid));

  for(int i2=0; i2<Nmodes; i2++){

    // precompute gamma * v_{i2} for all nu (hoisted out of i3 loop)
    vector<FermionField> v_g_nu(Ngamma, grid);
    for(int nu=0; nu<Ngamma; nu++) v_g_nu[nu] = Gamma(gammas[nu]) * vi[i2];

    for(int i3_base=0; i3_base < Nmodes; i3_base += BATCH) {

      int i3_end = min(i3_base + BATCH, Nmodes);
      int chunk = i3_end - i3_base; // active i3 values in this chunk

      // ---- Pack all bilinears for this chunk ----
      // Zero the packed field so unused slots (if chunk < BATCH) are clean
      packed = Zero();

      for(int b=0; b < chunk; b++) {
        int i3 = i3_base + b;

        // First bilinear: g_nu = <w_{i3} | gamma_nu | v_{i2}>
        // goes into components [4*b .. 4*b+3]
        for(int nu=0; nu<Ngamma; nu++) {
          ComplexField g_nu(grid);
          g_nu = localInnerProduct(wi[i3], v_g_nu[nu]);
          PokeIndex<0>(packed, g_nu, 4*b + nu);
        }

        // Second bilinear: h_mu = <A_{i2} | gamma_mu | B_{i3}>
        // goes into components [4*BATCH + 4*b .. 4*BATCH + 4*b+3]
        for(int mu=0; mu<Ngamma; mu++) {
          FermionField tmp = Gamma(gammas[mu]) * Bi[i3];
          ComplexField h_mu(grid);
          h_mu = localInnerProduct(Ai[i2], tmp);
          PokeIndex<0>(packed, h_mu, 4*BATCH + 4*b + mu);
        }
      }

      // ---- Single forward FFT for entire chunk ----
      theFFT.FFT_all_dim(packed_fft, packed, FFT::forward);

      // ---- Unpack, photon contract, and combine per i3 in chunk ----
      for(int b=0; b < chunk; b++) {

        // Unpack FFT of first bilinears
        for(int nu=0; nu<Ngamma; nu++) {
          g_tilde_nu[nu] = PeekIndex<0>(packed_fft, 4*b + nu);
        }

        // Photon propagator contraction: Kg_mu = sum_nu g~_nu * Delta~_{nu,mu}
        for(int mu=0; mu<Ngamma; mu++) {
          //Kg_mu[mu] = g_tilde_nu[0] * phtn_prop_mu_nu[0*Nd + mu];
          //for(int nu=1; nu<Ngamma; nu++) {
          //  Kg_mu[mu] = Kg_mu[mu] + g_tilde_nu[nu] * phtn_prop_mu_nu[nu*Nd + mu];
          //}
          Kg_mu[mu] = g_tilde_nu[0] * phtn_prop_mu_nu[0*Nd + mu]
                    + g_tilde_nu[1] * phtn_prop_mu_nu[1*Nd + mu]
                    + g_tilde_nu[2] * phtn_prop_mu_nu[2*Nd + mu]
                    + g_tilde_nu[3] * phtn_prop_mu_nu[3*Nd + mu];
        }

        // Unpack FFT of second bilinears
        for(int mu=0; mu<Ngamma; mu++) {
          h_tilde_mu[mu] = PeekIndex<0>(packed_fft, 4*BATCH + 4*b + mu);
        }

        // Parseval combination: innerProduct(IFFT[Kg], h) = (1/V) * innerProduct(Kg, FFT[h])
        for(int mu=0; mu<Ngamma; mu++) {
          Result += inv_vol * innerProduct(Kg_mu[mu], h_tilde_mu[mu]);
        }
      }

    } // i3_base
  } // i2

  cout << GridLogMessage << "============================================" << endl;
  cout << GridLogMessage << "COMPLETE: FFT TYPE 2 CONV + CONT (CHUNKED BATCH)" << endl;
  cout << GridLogMessage << "============================================" << endl;

};


NAMESPACE_END(Grid);



