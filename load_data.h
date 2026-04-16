#pragma once

#include <Grid/Grid.h>

using namespace Grid;

typedef Eigen::Matrix<ComplexF, -1, -1, Eigen::RowMajor> MesonFieldMatrix;

/******************************************************************************
 *               Load binned A2A vectors from SCIDAC multiFile format         *
 *                                                                            *
 *  Reads filestem.<traj>/elem0.bin, elem1.bin, ... and unpacks into          *
 *  individual FermionField vectors.                                          *
 *                                                                            *
 *  binSize:    number of vectors packed per file (template parameter)        *
 *  vec:        output vector, must be pre-sized (vec.size() % binSize == 0)  *
 *  filestem:   e.g. "./vl_grid.173"                                          *
 *  trajectory: e.g. 5070                                                     *
 *  grid:       the 4D GridCartesian                                          *
 ******************************************************************************/
struct A2AVecRecord : Serializable
{
    GRID_SERIALIZABLE_CLASS_MEMBERS(A2AVecRecord,
                                    unsigned int, index);
    A2AVecRecord(void) : index(0) {}
};

template <int binSize, typename FermionField>
void loadBinnedA2AVecs(std::vector<FermionField> &vec,
                       const std::string &filestem,
                       const int trajectory,
                       GridBase *grid)
{
    typedef typename FermionField::vector_object::element::element vector_type;
    typedef iVector<iVector<iVector<vector_type, Nc>, Ns>, binSize> SiteSpinorSet;
    typedef Lattice<SiteSpinorSet> BinnedField;

    assert(vec.size() % binSize == 0);
    unsigned int nBins = vec.size() / binSize;

    std::string dir = filestem + "." + std::to_string(trajectory);

    std::cout << "Loading " << vec.size() << " A2A vectors from "
              << nBins << " binned files (binSize=" << binSize << ")" << std::endl;

    for (unsigned int b = 0; b < nBins; ++b)
    {
        BinnedField bvec(grid);
        A2AVecRecord record;
        std::string filename = dir + "/elem" + std::to_string(b) + ".bin";

        std::cout << "Reading " << filename << std::endl;
        ScidacReader reader;
        reader.open(filename);
        reader.readScidacFieldRecord(bvec, record);
        reader.close();

        for (int j = 0; j < binSize; ++j)
        {
            vec[b * binSize + j] = peekLorentz(bvec, j);
        }
    }

    std::cout << "Loaded " << vec.size() << " vectors" << std::endl;
}

inline std::vector<MesonFieldMatrix> loadMesonFields(const std::string &filename,
                                                      const std::string &datasetName,
                                                      const unsigned int nt)
{
#ifndef HAVE_HDF5
    assert(0 && "Grid must be compiled with HDF5 support");
    return {};
#else
    Hdf5Reader reader(filename);
    push(reader, datasetName);
    auto &group = reader.getGroup();

    H5NS::DataSet   dataset   = group.openDataSet("a2aMatrix");
    H5NS::CompType  datatype  = dataset.getCompType();
    H5NS::DataSpace dataspace = dataset.getSpace();

    std::vector<hsize_t> hdim(dataspace.getSimpleExtentNdims());
    dataspace.getSimpleExtentDims(hdim.data());

    unsigned int ni = hdim[1], nj = hdim[2];
    assert(hdim[0] == nt);

    std::vector<MesonFieldMatrix> mf(nt);

    std::vector<hsize_t> count  = {1, ni, nj},
                         stride = {1, 1, 1},
                         block  = {1, 1, 1};
    std::vector<hsize_t> memCount = {ni, nj};
    H5NS::DataSpace memspace(memCount.size(), memCount.data());

    for (unsigned int t = 0; t < nt; ++t)
    {
        mf[t].resize(ni, nj);
        std::vector<hsize_t> offset = {t, 0, 0};
        dataspace.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data(),
                                  stride.data(), block.data());
        dataset.read(mf[t].data(), datatype, memspace, dataspace);
    }

    std::cout << "Loaded " << filename << ": " << nt << " timeslices, "
              << ni << " x " << nj << std::endl;

    return mf;
#endif
}
