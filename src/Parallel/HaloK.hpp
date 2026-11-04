#ifndef HALOK_HPP
#define HALOK_HPP

#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>
#include "core/mpi/Communicator.hpp"
#include "MPIindex/MPIindex.hpp"
#include "Operator/BlockMatrix.hpp"
#include "mdContainers/mdContainers.hpp"

namespace parallel
{
    /// @brief Communication of the k points needed by the equation of motion with q components.
    /// The k points are split by FFTW-MPI in contiguous slabs along k1, one per rank. The commutator at the local k
    /// needs the operators also at k + p for every p of the q grid, which can belong to other ranks.
    /// initialize() computes once, without communication of the data, which k points every rank needs and who has
    /// them: each rank knows the slabs of all the ranks and the global map ikq, so it computes the requests of the
    /// others too. exchange() then sends at each call only those k points, with one MPI_Alltoallv.
    ///
    /// The halo buffer has first the local k points (in the local order), then the received ones, grouped by the rank
    /// that sends them and ordered by global index. position()[ik_glob] is the block of ik_glob in the halo buffer
    /// (-1 if this rank does not need it).
    class HaloK
    {
        private:
            int nk_glob_ = 0;
            int nlocal_ = 0;
            int offset_ = 0;
            int n_halo_ = 0;
            std::vector<int> position_;
            /// local k (index in the slab) to send, grouped by destination rank
            std::vector<int> send_list_;
            /// number of k points sent to / received from each rank, and where they start (in k points)
            std::vector<int> send_counts_, send_displs_, recv_counts_, recv_displs_;

        public:
            HaloK() = default;

            /// @param ikq__ ikq__(ik_glob, ip) = global index of k + p[ip] in the k grid
            /// @param mpindex__ splitting of the k grid among the ranks
            void initialize(const mdarray<int,2>& ikq__, const MPIindex<3>& mpindex__)
            {
                nk_glob_ = int(ikq__.get_Size(0));
                const int nq = int(ikq__.get_Size(1));
                nlocal_ = mpindex__.get_nlocal();
                offset_ = int(mpindex__.get_LocalRange().first);
                int nranks = 1;
                int me = 0;
                std::vector<int> nlocal_all(1, nlocal_), offset_all(1, offset_);
#ifdef EDUS_MPI
                auto& comm = mpi::Communicator::world();
                nranks = comm.size();
                me = comm.rank();
                nlocal_all.resize(nranks);
                offset_all.resize(nranks);
                MPI_Allgather(&nlocal_, 1, MPI_INT, nlocal_all.data(), 1, MPI_INT, comm.communicator());
                MPI_Allgather(&offset_, 1, MPI_INT, offset_all.data(), 1, MPI_INT, comm.communicator());
#endif
                /* owner of each global k */
                std::vector<int> owner(nk_glob_, -1);
                for( int r = 0; r < nranks; ++r ) {
                    for( int ik = offset_all[r]; ik < offset_all[r] + nlocal_all[r]; ++ik ) {
                        owner[ik] = r;
                    }
                }
                if( std::find(owner.begin(), owner.end(), -1) != owner.end() ) {
                    throw std::runtime_error("HaloK: the slabs of the ranks do not cover the k grid\n");
                }
                /* k points of other ranks needed by rank r: k + p for k in the slab of r */
                auto needed_by = [&](int r) {
                    std::set<int> needed;
                    for( int ik = offset_all[r]; ik < offset_all[r] + nlocal_all[r]; ++ik ) {
                        for( int ip = 0; ip < nq; ++ip ) {
                            const int ikp = ikq__(ik, ip);
                            if( owner[ikp] != r ) {
                                needed.insert(ikp); // std::set: ordered by global index
                            }
                        }
                    }
                    return needed;
                };

                /* receive: what I need, grouped by owner (and ordered by global index inside a group) */
                position_.assign(nk_glob_, -1);
                for( int ik = 0; ik < nlocal_; ++ik ) {
                    position_[offset_ + ik] = ik;
                }
                recv_counts_.assign(nranks, 0);
                recv_displs_.assign(nranks, 0);
                auto mine = needed_by(me);
                n_halo_ = nlocal_;
                for( int s = 0; s < nranks; ++s ) {
                    recv_displs_[s] = n_halo_;
                    for( int ik : mine ) {
                        if( owner[ik] == s ) {
                            position_[ik] = n_halo_++;
                            ++recv_counts_[s];
                        }
                    }
                }
                /* send: my k points that the other ranks need, in the order they expect */
                send_counts_.assign(nranks, 0);
                send_displs_.assign(nranks, 0);
                send_list_.clear();
                for( int r = 0; r < nranks; ++r ) {
                    send_displs_[r] = int(send_list_.size());
                    if( r == me ) {
                        continue;
                    }
                    for( int ik : needed_by(r) ) {
                        if( owner[ik] == me ) {
                            send_list_.push_back(ik - offset_);
                            ++send_counts_[r];
                        }
                    }
                }
            }

            /// number of k points in the halo buffer (local + received)
            int n_halo() const { return n_halo_; }
            /// position()[ik_glob] = block of ik_glob in the halo buffer, -1 if not there
            const std::vector<int>& position() const { return position_; }
            /// global index of the first local k
            int offset() const { return offset_; }

            /// halo__: BlockMatrix with n_halo() k points and the same nq, nrows, ncols of local__ (local k points)
            template<typename T>
            void exchange(BlockMatrix<T>& halo__, const BlockMatrix<T>& local__) const
            {
                const int block = local__.get_nq()*local__.get_nrows()*local__.get_ncols();
                if( local__.get_nk() != nlocal_ || halo__.get_nk() != n_halo_ || halo__.get_nq() != local__.get_nq() ||
                    halo__.get_nrows() != local__.get_nrows() || halo__.get_ncols() != local__.get_ncols() ) {
                    throw std::runtime_error("HaloK::exchange: halo and local BlockMatrix are not compatible\n");
                }
                std::copy(local__.data(), local__.data() + std::size_t(nlocal_)*block, halo__.data());
#ifdef EDUS_MPI
                auto& comm = mpi::Communicator::world();
                if( comm.size() == 1 ) {
                    return;
                }
                std::vector<T> send_buffer(send_list_.size()*block);
                #pragma omp parallel for
                for( int i = 0; i < int(send_list_.size()); ++i ) {
                    std::copy(local__.data() + std::size_t(send_list_[i])*block,
                              local__.data() + std::size_t(send_list_[i] + 1)*block,
                              send_buffer.data() + std::size_t(i)*block);
                }
                const int nranks = comm.size();
                std::vector<int> sc(nranks), sd(nranks), rc(nranks), rd(nranks);
                for( int r = 0; r < nranks; ++r ) {
                    sc[r] = send_counts_[r]*block;
                    sd[r] = send_displs_[r]*block;
                    rc[r] = recv_counts_[r]*block;
                    rd[r] = recv_displs_[r]*block;
                }
                MPI_Alltoallv(send_buffer.data(), sc.data(), sd.data(), mpi::type_wrapper<T>(),
                              halo__.data(), rc.data(), rd.data(), mpi::type_wrapper<T>(), comm.communicator());
#endif
            }
    };
}

#endif
