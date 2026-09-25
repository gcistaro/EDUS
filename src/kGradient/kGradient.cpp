#include "kGradient.hpp"

#include "omp.h"//TODO:: FIX ME PUT ME IN THE RIGHT PLACE!!


kGradient::kGradient(const MeshGrid& kmesh__) 
{
    this->initialize(kmesh__);
}

void kGradient::initialize(const MeshGrid& mesh__)
{
    assert(mesh__.get_type() == cube);
    if(mesh__.get_space() == k) {
        kmesh = std::make_shared<MeshGrid>( get_GammaCentered_grid(mesh__) );
    }
    else if(mesh__.get_space() == R) {
        Rmesh = std::make_shared<MeshGrid>( get_GammaCentered_grid(mesh__) );  //TODO:: Maybe we can always have the two meshes?
    }
    initialize();
}



void kGradient::initialize()
{
    //evaluates weights, number of shells, k+b indices.
    if(kmesh && !Rmesh) {
        // the neighbours k+b are taken with their global index, so the whole grid must be on one rank
        if( mpi::Communicator::world().size() > 1 ) {
            throw std::runtime_error("kGradient: the gradient in k space (gradient_space = \"k\") works only with 1 MPI rank, use gradient_space = \"R\"\n");
        }
        mpindex.initialize(kmesh->get_Size());
        ikshell = SortInShells(*kmesh);
        Calculate_nshellsAndweights(nshells, Weight, *kmesh, ikshell);
        ikpb = Find_kpb(*kmesh, ikshell);
    }
    else if(Rmesh && !kmesh) {
        //in this case, we already have everything
        mpindex.initialize(Rmesh->get_Size());
    }
    else {
        throw std::runtime_error("Something went wrong in kGradient::initialize() -> either both or none Rmesh and kmesh are initialized\n");
    }
}

std::vector<std::vector<int>> SortInShells(const MeshGrid& kmesh)
{
    std::vector<std::vector<int>> ikshell; //[ishell][ik] contains all ik with norm ishell
    //find all norms of k vectors
    std::vector<double> knorms; //array with norm of k vectors (not repeated)
    for( auto& k_ : kmesh.get_mesh() ) {
        auto k_norm = k_.norm();
        auto it = std::find_if(knorms.begin(), knorms.end(), 
                               [&k_norm](const auto& norm) { return std::abs( norm - k_norm ) < threshold; });
        bool found = ( it != knorms.end() );
        if( !found ) {
            knorms.push_back( k_norm );
        }                 
    }
    std::sort( knorms.begin(), knorms.end() );
    ikshell.resize(knorms.size());
    //find indices of each shell
    for( int ik=0; ik< kmesh.get_TotalSize(); ++ik ) {
        //find what shell k_ belongs to
        auto k_norm = kmesh[ik].norm();
        auto it = std::find_if(knorms.begin(), knorms.end(), 
                               [&k_norm](const auto& norm) { return std::abs( norm - k_norm ) < 1.e-07; });
        assert(it != knorms.end());
        auto ishell = it - knorms.begin();
        //push back the index in ikshell
        ikshell[ishell].push_back(ik);
    }
    //remove 0 from kshells 
    assert( std::abs(knorms[0]) < threshold );
    ikshell.erase(ikshell.begin());
    knorms.erase(knorms.begin());

    //recap
    std::ofstream of("Cartesian.txt");
    for(int ishell=0; ishell < int( ikshell.size() ); ishell++) {
        for(int ik=0; ik < int( ikshell[ishell].size() ); ik++) {
            of << ishell << " " <<knorms[ishell] << " " <<  ikshell[ishell][ik] << " " << kmesh[ikshell[ishell][ik]].get("Cartesian");//get(LatticeVectors(k));
        }
    }
    of.close();

    return ikshell;
}


void Calculate_nshellsAndweights(int& nshells, mdarray<double,1>& Weight, 
                                 const MeshGrid& kmesh, const std::vector<std::vector<int>>& ikshell)
{    
    auto q = Matrix<double>({6,1});

    q(0,0) = ( kmesh.get_Size()[0] > 1 ? 1 : 0 ) ;     
    q(1,0) = ( kmesh.get_Size()[1] > 1 ? 1 : 0 ) ;   
    q(2,0) = ( kmesh.get_Size()[2] > 1 ? 1 : 0 ) ; 
    q(3,0) = 0.;     
    q(4,0) = 0.;    
    q(5,0) = 0.;


    bool EnoughShells = false;
    nshells = 0;
    Matrix<double> w;
    while(!EnoughShells) {
        nshells++;
        auto A_ = GradientMatrix(nshells, kmesh, ikshell); //computes A as in CPC 178 (2008) 685-599 between eq.25 and 26
        
        //compute w = A^{-1}*q
        auto invA = A_.pseudoinv();
        w = invA*q;
        //try A*w = qguess
        auto qguess = A_*w;
        //we have enough shells only when qguess is equal to q.
        EnoughShells = ( ( qguess - q ).norm() < threshold );
    }

    assert(w.get_nrows() == nshells);
    assert(w.get_ncols() == 1);

    Weight = mdarray<double, 1>({nshells});
    for(int ishell = 0; ishell < nshells; ++ishell) {
        Weight(ishell) = w(ishell, 0);
    }
}


/*
 *         MAP
 *   j      0        1        2       3       4      5
 * alpha    x        y        z       x       x      y           
 * beta     x        y        z       y       z      z      
 */ 
int alpha( const size_t& j ) 
{
    if( j == 0 )       return 0;
    if( j == 1 )       return 1;
    if( j == 2 )       return 2;
    if( j == 3 )       return 0;
    if( j == 4 )       return 0;
    if( j == 5 )       return 1;
    return -1;
}

int beta( const size_t& j ) 
{
    if( j == 0 )       return 0;
    if( j == 1 )       return 1;
    if( j == 2 )       return 2;
    if( j == 3 )       return 1;
    if( j == 4 )       return 2;
    if( j == 5 )       return 2;
    return -1;
}

/*
 * This routine evalues A(j,s) = sum_{i\in s} b^{i}_\alpha b^{i}_\beta
 *
*/
Matrix<double> GradientMatrix(const size_t& nshells, const MeshGrid& kmesh, const std::vector<std::vector<int>>& ik_sorted)
{
    Matrix<double> A(6, nshells);
    A.fill(0.);

    for( size_t j=0; j<6; ++j ) { // j is a one d index for alpha, beta
        auto alpha_ = alpha(j);
        auto beta_  = beta(j);

        for( size_t ishell = 0; ishell < nshells; ishell++) {
            for ( auto& ikshell : ik_sorted[ishell] ) {// go through indices in shell
                auto& k = kmesh[ikshell].get("Cartesian");
                A(j, ishell) += k[alpha_]*k[beta_];
            }       
        }
    }
    return A;
}

std::vector<std::vector<std::vector<int>>> kGradient::Find_kpb(const MeshGrid& kmesh, const std::vector<std::vector<int>>& ikshell)
{
#ifdef EDUS_MPI
    if( mpi::Communicator::world().size() > 1 ) {
        std::stringstream ss;
        ss << "\nfind_kpb still not implemented in MPI and you are using "<< mpi::Communicator::world().size() << "processors\n";
        ss << "SUGGESTION: Propagate gradient in R!\n";
        std::runtime_error(ss.str());
    }
#endif
    PROFILE("kgradient::Find_kpb");
    std::vector<std::vector<std::vector<int>>> ikpb_local(mpindex.nlocal);

    #pragma omp parallel for schedule(static)
    for( int ik = 0; ik < mpindex.nlocal; ++ik) {
        ikpb_local[ik].resize(nshells);
        for( int ishell = 0; ishell < nshells; ++ishell ) {
            ikpb_local[ik][ishell].resize( ikshell[ishell].size() );
        }
    }
    //find k+b in kmesh for every k, every b

    #pragma omp parallel for schedule(static)
    for( int ik_loc = 0; ik_loc < mpindex.nlocal; ++ik_loc) {
        auto ik_global = ik_loc;//mpindex.loc1D_to_glob1D(ik_loc);
        for( int ishell = 0; ishell < int( ikpb_local[ik_loc].size() ); ++ishell ) {
            for( int ib = 0; ib < int( ikpb_local[ik_loc][ishell].size() ); ++ib ) {
                ikpb_local[ik_loc][ishell][ib] = kmesh.find(kmesh[ik_global] + kmesh[ikshell[ishell][ib]]);
            }
        }
    }
    return ikpb_local;
}

