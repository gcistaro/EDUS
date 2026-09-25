
#ifndef MPIINDEX_HPP
#define MPIINDEX_HPP

#include <memory>

#include "mpi.h"
#include "fftw3-mpi.h"
#include "MultiIndex/MultiIndex.hpp"
#include "core/mpi/Communicator.hpp"

template<size_t dim>
class MPIindex
{
    private:
        std::pair<std::ptrdiff_t, std::ptrdiff_t> GlobalRange_1D;
        std::pair<std::ptrdiff_t, std::ptrdiff_t> LocalRange_1D;
        int RecommendedAllocate_fftw;
        std::array<int, dim> ValuesToSplit;
        std::array<int, dim> Offset;
        MultiIndex<dim> multindex;
        int howmany; //so far, this is just for getting right allocation.
#ifdef EDUS_MPI
        std::shared_ptr<mpi::Communicator> mpi_comm_;
#endif
    public:
        int nlocal = 0; //number of local points

        MPIindex(){};

        MPIindex( const std::array<int, 3>& ValuesToSplit__, const int& howmany__=1);
        MPIindex( const std::array<int, 2>& ValuesToSplit__);

        void initialize( const std::array<int, 3>& ValuesToSplit__, const int& howmany__=1);
        void initialize( const std::array<int, 2>& ValuesToSplit__ );
        
        inline std::ptrdiff_t glob1D_to_loc1D(const std::ptrdiff_t& global) const;
        inline std::ptrdiff_t loc1D_to_glob1D(const std::ptrdiff_t& local) const;

        template<typename... Args>
        inline std::ptrdiff_t globnD_to_glob1D (const Args&... args) const;

        template<typename... Args>
        inline std::ptrdiff_t globnD_to_loc1D (const Args&... args) const;

        inline std::array<int, dim> loc1D_to_globnD (const std::ptrdiff_t& loc1D) const;

        inline std::array<int, dim> glob1D_to_globnD (const std::ptrdiff_t& glob1D) const;

        inline int get_RecommendedAllocate_fftw() const;

        inline const auto& get_GlobalRange() const {return GlobalRange_1D; };
        inline auto& get_GlobalRange() {return GlobalRange_1D; };


        inline const auto& get_LocalRange() const {return LocalRange_1D; };
        inline auto& get_LocalRange() {return LocalRange_1D; };
        
        inline int get_nlocal() const;
        bool is_local(const int& index) const;

};


#include "MPIindex/MPIindex_definitions.hpp"
#endif