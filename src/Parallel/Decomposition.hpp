#ifndef DECOMPOSITION_HPP
#define DECOMPOSITION_HPP

#include "core/mpi/Communicator.hpp"
#include "MPIindex/MPIindex.hpp"

namespace parallel
{
    class Decomposition
    {
private:
        mpi::Communicator kpool_comm_;  
        mpi::Communicator band_comm_;

        MPIindex<3> mpindex_;
public:
        Decomposition() = default;
        inline void initialize(const std::array<int,3>& kgrid, int num_bands_total,
                    int num_kpool, int num_bandpool);

        // int local_num_kpoints() const;
        // const std::vector<int>& local_kpoint_indices() const;
        const mpi::Communicator& kpool_comm() const { return kpool_comm_; }
        const mpi::Communicator& band_comm() const { return band_comm_; }
        const MPIindex<3>& mpindex() const { return mpindex_; }
        // wrapper delle operazioni collettive che oggi fai a mano su kpool_comm
        // template<typename T>
        // void reduce_over_kpool(T* local, T* reduced, size_t count, MPI_Op op, int root) const;
    };

    inline void Decomposition::initialize(const std::array<int,3>& kgrid, int num_bands_total,
                        int num_kpool, int num_bandpool = 1)   
    {
        /* Simplified. for now we only split over k pools */
        int kpool_index      = mpi::Communicator::world().rank() / num_bandpool;
        int band_group_index = mpi::Communicator::world().rank() % num_bandpool;

        band_comm_.generate(mpi::Communicator::world(), kpool_index);        // raggruppa per stesso k-pool
        kpool_comm_.generate(mpi::Communicator::world(), band_group_index);  // raggruppa per stesso band-group        /* initialize splitting for operator-like variables */
        mpindex_.initialize(kgrid, num_bands_total * num_bands_total);
    }
}

#endif