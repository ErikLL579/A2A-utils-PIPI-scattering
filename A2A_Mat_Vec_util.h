//#include <Grid/Hadrons/Global.hpp>
#include <Grid/Grid_Eigen_Tensor.h>


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


NAMESPACE_BEGIN(Grid);

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

  // returns Fermionfield y with one outstanding A2A index i1
  template <typename TensorType>
  static void ContractMesonFieldAndVector(FermionField *y_i1,
                         const TensorType &meson_field_ij,
                         const FermionField *wj);
};

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////

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

  const int Nmodes = 8; //= meson_field_ij.dimension(0);

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

NAMESPACE_END(Grid);
