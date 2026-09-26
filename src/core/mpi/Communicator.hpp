//this file is adapted from sirius code
#ifndef COMMUNICATORS_HPP
#define COMMUNICATORS_HPP


#ifdef EDUS_MPI

#include <complex>
#include "mpi.h"
#include <cassert>
#include <iostream>
#include <cstdint>

namespace mpi
{
template <typename T>
struct type_wrapper;

template <>
struct type_wrapper<float>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_FLOAT;
    }
};

template <>
struct type_wrapper<std::complex<float>>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_C_FLOAT_COMPLEX;
    }
};

template <>
struct type_wrapper<double>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_DOUBLE;
    }
};

template <>
struct type_wrapper<std::complex<double>>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_C_DOUBLE_COMPLEX;
    }
};

template <>
struct type_wrapper<long double>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_LONG_DOUBLE;
    }
};

template <>
struct type_wrapper<int>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_INT;
    }
};

template <>
struct type_wrapper<int16_t>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_SHORT;
    }
};

template <>
struct type_wrapper<char>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_CHAR;
    }
};

template <>
struct type_wrapper<unsigned char>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_UNSIGNED_CHAR;
    }
};

template <>
struct type_wrapper<unsigned long long>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_UNSIGNED_LONG_LONG;
    }
};

template <>
struct type_wrapper<unsigned long>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_UNSIGNED_LONG;
    }
};

template <>
struct type_wrapper<bool>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_C_BOOL;
    }
};

template <>
struct type_wrapper<uint32_t>
{
    operator MPI_Datatype() const noexcept
    {
        return MPI_UINT32_T;
    }
};




int tag ( const int& sender, const int& receiver );

class Communicator {
    private:
        MPI_Comm communicator_{MPI_COMM_NULL};
        int rank_{-1};
        int size_{-1};
    
    public:
        //Default
        Communicator(){};

        void init()
        {
            assert(communicator_ != MPI_COMM_NULL);
            MPI_Comm_rank(communicator_, &rank_);
            MPI_Comm_size(communicator_, &size_);
        };

        //From raw communicator
        Communicator(MPI_Comm mpi_comm__)
            : communicator_(mpi_comm__)
        {
            init();
        };

        /// Rank of actual process inside the communicator
        inline int rank() const
        {
            return rank_;
        };
    
        /// Size of actual communicator
        inline int size() const
        {
            return size_;
        };

        //communicator of the object
        inline MPI_Comm&
        communicator() 
        {
            return communicator_;
        }

        inline const MPI_Comm&
        communicator() const
        {
            return communicator_;
        }

        //split from a communicator with given color
        void generate( const Communicator& father_comm, int color )
        {
            MPI_Comm_split(father_comm.communicator(), color, father_comm.rank(), &communicator_);
            init();
        }

        template<typename T>
        void send( const T* message, const int& receiver, const int& count=1 ) const
        {
            MPI_Send( message, count, type_wrapper<T>(), receiver, mpi::tag(rank(), receiver), communicator_);
        }

        template<typename T>
        void receive( T* message, const int& sender, const int& count=1 ) const
        {
            MPI_Recv( message, count, type_wrapper<T>(), sender, mpi::tag(sender, rank()), communicator_, MPI_STATUS_IGNORE);
        }

        //send message -- to be optimized with any type 
        template<typename T>
        void isend( T* message, const int& receiver, const int& count, MPI_Request& request) const
        {
            MPI_Isend( message, count, type_wrapper<T>(), receiver, mpi::tag(rank(), receiver), communicator_, &request);
        }



        //receive message -- to be optimized with any type 
        template<typename T>
        void ireceive( T* message, const int& sender, const int& count, MPI_Request& request ) const
        {
            MPI_Irecv( message, count, type_wrapper<T>(), sender, mpi::tag(sender, rank()), communicator_, &request);
        }

        //gather
        void reduce( std::complex<double>* to_send, std::complex<double>* to_recv, int count, const MPI_Op& mpi_op, int root)
        {
            MPI_Reduce( to_send, to_recv, count, MPI_C_DOUBLE_COMPLEX, mpi_op, root, communicator_);
        }

        /// MPI global initialization.
        static void initialize(int required__)
        {
            int provided;

            MPI_Init_thread(NULL, NULL, required__, &provided);
            MPI_Query_thread(&provided);
            if ((provided < required__) && (Communicator::world().rank() == 0)) {
                printf("Warning! Required level of thread support is not provided.\n");
                printf("provided: %d \nrequired: %d\n", provided, required__);
            }
        };

        inline void
        barrier() const
        {
            assert(communicator_ != MPI_COMM_NULL);
            MPI_Barrier(communicator_);
        }

        // MPI global finalization
        static void finalize()
        {
            MPI_Finalize();
        };
        //retrieve MPI_COMM_WORLD communicator
        static const Communicator& world()
        {
            static Communicator comm(MPI_COMM_WORLD);
            return comm;
        };
};


}//end namespace mpi
#else //EDUS_MPI
namespace mpi
{
    class Communicator 
    {
        public:
            Communicator(){};

            static Communicator& world()
            {
                static Communicator comm;
                return comm;
            };

            void barrier() {};
            int rank() const { return 0; }
            int size() const { return 1; }
    };

};
#endif //EDUS_MPI
#endif//COMMUNICATORS_HPP
