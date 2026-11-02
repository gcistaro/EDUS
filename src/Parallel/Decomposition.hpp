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
        MPIindex<3> mpindex_q_;
public:
        Decomposition() = default;
        inline void initialize(const std::array<int,3>& kgrid__,
                               const std::array<int,3>& qgrid__,
                               int num_bands_total__,
                               int num_kpool__, 
                               int num_bandpool__ = 1);

        // int local_num_kpoints() const;
        // const std::vector<int>& local_kpoint_indices() const;
        const mpi::Communicator& kpool_comm() const { return kpool_comm_; }
        const mpi::Communicator& band_comm() const { return band_comm_; }
        const MPIindex<3>& mpindex() const { return mpindex_; }
        const MPIindex<3>& mpindex_q() const { return mpindex_q_; }
        // wrapper delle operazioni collettive che oggi fai a mano su kpool_comm
        // template<typename T>
        // void reduce_over_kpool(T* local, T* reduced, size_t count, MPI_Op op, int root) const;
    };

    inline void Decomposition::initialize(const std::array<int,3>& kgrid__,
                                          const std::array<int,3>& qgrid__,
                                          int num_bands_total__,
                                          int num_kpool__, 
                                          int num_bandpool__)
    {
        /* Simplified. for now we only split over k pools */
        int kpool_index      = mpi::Communicator::world().rank() / num_bandpool__;
        int band_group_index = mpi::Communicator::world().rank() % num_bandpool__;

        band_comm_.generate(mpi::Communicator::world(), kpool_index);        // group for same kpool
        kpool_comm_.generate(mpi::Communicator::world(), band_group_index);  // group for same band-group        
        /* initialize splitting for operator-like variables */
        mpindex_.initialize(kgrid__, num_bands_total__ * num_bands_total__);
        
        int nq = qgrid__[0] * qgrid__[1] * qgrid__[2];
        mpindex_q_.initialize(kgrid__, nq * num_bands_total__ * num_bands_total__);
    }
}

#endif