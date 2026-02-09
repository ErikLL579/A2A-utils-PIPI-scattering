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
                                              TensorType_TraceMomTime &Result,
                                              vector<int> &A_vector_contractions,
                                              vector<int> &B_vector_contractions,
                                              vector<int> &C_vector_contractions);


  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  static void MesonField_MesonField_disconnected(TensorType_mesonfield &Mesonfield,
                                              TensorType_TraceMomTime &Result,
                                              vector<int> &A_vector_contractions,
                                              vector<int> &B_vector_contractions,
                                              vector<int> &C_vector_contractions);



/*
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// FFT functionality  
// 
// See https://rbc.phys.columbia.edu/rbc_ukqcd/individual_postings/ellundstrum/K_to_pipi/pipi-scattering/pi_pi_scattering_A2A.pdf
// FFT strategy 1: Appendix B
// \sum_i <A_i (z1)| \gamma_mu |B_i (z1)> = \phi_mu (z1)
//
// FFT strategy 2: Appendix C
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
*/

// \sum_i <A_i (z1)| \gamma_mu |B_i (z1)> = \phi_mu (z1)
static void FFT_type1_prod( std::vector<ComplexField> *phi_mu,
		            const FermionField *Ai,
		            const FermionField *Bj,
		            std::vector<Gamma::Algebra> gammas);

// FFT_type1_convolveo


// FFT_type2_prod_and_convolve

};

/////////////////////////////////////////////////////////////////////////

// the below already enters from the normal A2Autils header
//const int A2Ablocking=8;

template<typename vtype> using iVecSpinMatrix = iVector<iMatrix<iScalar<vtype>, Ns>, A2Ablocking>;
typedef iVecSpinMatrix<Complex  >             VecSpinMatrix;
typedef iVecSpinMatrix<vComplex >             vVecSpinMatrix;
typedef Lattice<vVecSpinMatrix>               LatticeVecSpinMatrix;

template<typename vtype> using iVecComplex = iVector<iScalar<iScalar<vtype> >, A2Ablocking>;
typedef iVecComplex<Complex  >             VecComplex;
typedef iVecComplex<vComplex >             vVecComplex;
typedef Lattice<vVecComplex>               LatticeVecComplex;

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

  LatticeVecSpinMatrix SpinMat(grid);

  double t_view, t_gamma, t_kernel, t_momproj;
  t_view=0;
  t_gamma=0;
  t_kernel=0;
  t_momproj=0;

  std::vector<VecSpinMatrix> sliced;

  // meson field vector on device
  static deviceVector<ComplexD> PI(Nmodes * Nmodes);

  // placeholder until I sort out DTYPE for Masaaki's meson fields
  // DTYPE flat_meson_field;

  // copy host to device and get memory address
  //acceleratorCopyToDevice((void *)&flat_meson_field[0] ,(void *)&PI[0],Nmodes * Nmodes* sizeof(ComplexD));

  // works for single A2A vector stripped to NmodesxNmodes basis
  acceleratorCopyToDevice(meson_field_ij.data() ,(void *)&PI[0],Nmodes * Nmodes* sizeof(ComplexD));

//const ComplexD *PI_ptr = &PI[0]; 

  ComplexD *PI_ptr;
  acceleratorPut(PI_ptr, &PI[0]);

  for(int i=0; i<Nmodes; i++) {

    autoView(y_i,y_i1[i],AcceleratorWrite);

    for(int jo=0;jo<Nmodes;jo+=block){
      for(int j=jo;j<MIN(Nmodes,jo+block);j++){
        int jj=j%block;

        autoView(w,wj[j],AcceleratorRead); // create vector of views

        Complex Pi_val = PI_ptr[i * Nmodes + j];

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


  template<class FImpl>
  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  void PipiA2Autils<FImpl>::MesonField_MesonField_disconnected(TensorType_mesonfield &Mesonfield,
                                                            TensorType_TraceMomTime &Result,
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
    // ex single Meson field: 2700 * 2700 * 16 ~ 11MB
    // Perlmutter A100 has 40GB memory => 300 matrices = ~ 32 GB with 8GB for system

    const int max_batch_size = 100;
    const int contractions = A_vector_contractions.size();

    // check that contraction vector are the same size
    GRID_ASSERT(contractions = B_vector_contractions.size());
    GRID_ASSERT(contractions = C_vector_contractions.size());
    GRID_ASSERT(contractions < max_batch_size);

    // total number of matrices
    // timeslices * num_momenta * 2 mom orientations
    // ex: 24^3 x 64 ensemble: 64 * 4 * 2 = 312 pion mesonfields per config

    // set up memory on device
    const int Ncomplex = Nmodes * Nmodes * contractions;

    deviceVector<ComplexD> A(Ncomplex); // input vectors
    deviceVector<ComplexD> C(Ncomplex); // result of matrix operations on device

    // need to parse the input Eigen matrix appropriately
    // compute DC pieces at all tsrc

    // Input mesonfield Mpp(i, j, k, l)
    // use Mpp.data() to copy to device => Offset = i*(Nt*Nmodes*Nmodes) + j*(Nmodes*Nmodes) + k*(Nmodes) + l

    // need to only upload enough A fields to actually fill this out...
    acceleratorCopyToDevice(Mesonfield.data(), &A[0], Nmodes * Nmodes * contractions * sizeof(ComplexD));

    deviceVector<ComplexD* > As(contractions);
    // Same matrices as in As but in the order necessary for the contraction
    deviceVector<ComplexD* > Bs(contractions);
    deviceVector<ComplexD* > Cs(contractions);

    // vector of which matrices to contract

    for(int b=0; b<contractions; b++) {
      ComplexD *ptr;
      
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

    ComplexD alpha(1.0);
    ComplexD beta(0.0);
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
    acceleratorCopyFromDevice(&C[0], Result.data(), Nmodes * Nmodes * sizeof(ComplexD) * contractions);
};





  template<class FImpl>
  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  void PipiA2Autils<FImpl>::MesonField_MesonField_connected(TensorType_mesonfield &Mesonfield,
                                                            TensorType_TraceMomTime &Result,
                                                            vector<int> &A_vector_contractions,
                                                            vector<int> &B_vector_contractions,
                                                            vector<int> &C_vector_contractions)
 
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
    // ex single Meson field: 2700 * 2700 * 16 ~ 11MB
    // Perlmutter A100 has 40GB memory => 300 matrices = ~ 32 GB with 8GB for system
      
    const int max_batch_size = 100;
    const int contractions = A_vector_contractions.size();
     
    // check that contraction vector are the same size
    GRID_ASSERT(contractions = B_vector_contractions.size());
    GRID_ASSERT(contractions = C_vector_contractions.size());
    GRID_ASSERT(contractions < max_batch_size);
      
    // total number of matrices
    // timeslices * num_momenta * 2 mom orientations
    // ex: 24^3 x 64 ensemble: 64 * 4 * 2 = 312 pion mesonfields per config
  
    // set up memory on device
    const int Ncomplex = Nmodes * Nmodes * contractions;
                                                            
    deviceVector<ComplexD> A(Ncomplex); // input vectors
    deviceVector<ComplexD> C(Ncomplex); // result of matrix operations on device
    
    // need to parse the input Eigen matrix appropriately
    // compute DC pieces at all tsrc
    
    // Input mesonfield Mpp(i, j, k, l)
    // use Mpp.data() to copy to device => Offset = i*(Nt*Nmodes*Nmodes) + j*(Nmodes*Nmodes) + k*(Nmodes) + l
    
    // need to only upload enough A fields to actually fill this out...
    acceleratorCopyToDevice(Mesonfield.data(), &A[0], Nmodes * Nmodes * contractions * sizeof(ComplexD));
    
    deviceVector<ComplexD* > As(contractions);
    // Same matrices as in As but in the order necessary for the contraction
    deviceVector<ComplexD* > Bs(contractions);
    deviceVector<ComplexD* > Cs(contractions);

 // vector of which matrices to contract
  
    for(int b=0; b<contractions; b++) {  
      ComplexD *ptr;
    
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
                                              
    ComplexD alpha(1.0);
    ComplexD beta(0.0);
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

    // ======================================================================================================
    // cant use same ptr for two arguments: blas.gemmBatched(Nmodes, Nmodes, Nmodes, alpha, Cs, Bs, beta, Cs);
    // will have to istead mix up the ones that are fed into the arguments
    // ======================================================================================================
 
    RealD t0 = usecond();
    
    // (rotated arguments here for illustration)
    // (check that the matrices are transposed correctly)
    blas.gemmBatched(Nmodes, Nmodes, Nmodes, alpha, Cs, As, beta, Bs);
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
    acceleratorCopyFromDevice(&C[0], Result.data(), Nmodes * Nmodes * sizeof(ComplexD) * contractions);
};




template <class FImpl>                   // should be std vector but can just point to first mem location
void PipiA2Autils<FImpl>::FFT_type1_prod( ComplexField *phi_mu,
                                          const FermionField *Ai,
                                          const FermionField *Bj,
                                          const std::vector<Gamma::Algebra> gammas)
{

  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  const int block=A2Ablocking;
 
  GridBase *grid = lhs_wi[0].Grid();
  const int Nsimd = grid->Nsimd();
  const int    Nd = grid->_ndimension;

  int Nt     = grid->GlobalDimensions()[orthogdim];
  int Ngamma = gammas.size();

  // copy gamma matrices to device
  static deviceVector<ComplexD> gamma(Ns * Ns * Ng);
  
  // move gammas to Eigen matrix or something to make the below work as intended
  acceleratorCopyToDevice(gammas.data() ,(void *)gamma[0],Ng * Ns * Ns * sizeof(ComplexD));


  cout << GridLogMessage << "============================" << endl;
  cout << GridLogMessage << "============================" << endl; 
  cout << GridLogMessage << "FFT TYPE 1 CONTRACTION START" << endl;
  cout << GridLogMessage << "============================" << endl;
  cout << GridLogMessage << "============================" << endl;

  // adjust this to be dynamic based on input
  const int Nmodes = 100;
  // number of blocks (adjust to keep GPU memory ~80% full)
  // const int blocks = (Nmodes + block - 1 ) / block ; 

  // might be able to speed this up by copying larger blocks?
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

  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "FFT TYPE 1 CONTRACTION COMPLETE" << endl;
  cout << GridLogMessage << "===============================" << endl;
  cout << GridLogMessage << "===============================" << endl;

};


NAMESPACE_END(Grid);
