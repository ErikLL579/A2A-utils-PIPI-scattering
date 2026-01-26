#include <Grid/Grid_Eigen_Tensor.h>

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

  template <typename TensorType>
  static void ContractMesonFieldAndVector(FermionField *y_i1,
                                          const TensorType &meson_field_ij,
                                          const FermionField *wj);

  template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
  static void TraceMesonFields(TensorType_mesonfield &Mesonfield,
                               TensorType_TraceMomTime &Result);
};

/////////////////////////////////////////////////////////////////////////

template<typename vtype> using iVecSpinMatrix = iVector<iMatrix<iScalar<vtype>, Ns>, A2Ablocking>;
typedef iVecSpinMatrix<Complex  >             VecSpinMatrix;
typedef iVecSpinMatrix<vComplex >             vVecSpinMatrix;
typedef Lattice<vVecSpinMatrix>               LatticeVecSpinMatrix;

template<typename vtype> using iVecComplex = iVector<iScalar<iScalar<vtype> >, A2Ablocking>;
typedef iVecComplex<Complex  >             VecComplex;
typedef iVecComplex<vComplex >             vVecComplex;
typedef Lattice<vVecComplex>               LatticeVecComplex;

#define A2A_GPU_KERNELS


template <class FImpl>
template <typename TensorType>
void PipiA2Autils<FImpl>::ContractMesonFieldAndVector(FermionField *y_i1,
                                                      const TensorType &meson_field_ij,
                                                      const FermionField *wj)
{
  typedef typename FImpl::SiteSpinor vobj;

  const int Nmodes = meson_field_ij.dimension(0);

  GridBase *grid = wj[0].Grid();
  const int Nsimd = grid->Nsimd();

  // Copy meson field to device
  deviceVector<ComplexD> PI(Nmodes * Nmodes);
  acceleratorCopyToDevice(meson_field_ij.data(), &PI[0], Nmodes * Nmodes * sizeof(ComplexD));
  ComplexD *PI_ptr = &PI[0];

  for(int i = 0; i < Nmodes; i++) {

    autoView(y_i, y_i1[i], AcceleratorWrite);

    for(int j = 0; j < Nmodes; j++) {

      autoView(w, wj[j], AcceleratorRead);

      accelerator_for(ss, grid->oSites(), Nsimd, {
        auto w_local = coalescedRead(w[ss]);
        auto y_local = coalescedRead(y_i[ss]);

        y_local = y_local + PI_ptr[i * Nmodes + j] * w_local;

        coalescedWrite(y_i[ss], y_local);
      });

    }
  }

  std::cout << GridLogMessage << "MATRIX-VECTOR OPERATION COMPLETED" << std::endl;
}


template<class FImpl>
template<typename TensorType_mesonfield, typename TensorType_TraceMomTime>
void PipiA2Autils<FImpl>::TraceMesonFields(TensorType_mesonfield &Mesonfield,
                                           TensorType_TraceMomTime &Result)
{
  // Placeholder - same as original
  const int block = A2Ablocking;

  int num_momenta = Mesonfield.dimension(0);
  int timeslices  = Mesonfield.dimension(2);
  int Nmodes = Mesonfield.dimension(3);

  GRID_ASSERT(Nmodes == Mesonfield.dimension(4));
}

NAMESPACE_END(Grid);
