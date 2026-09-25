#include "Electrons/System.hpp"
#include "GridStructure/GridStructure.hpp"
#include "Wannier/PrintWannier.hpp"
#include "ConvertUnits.hpp"
namespace electron
{
    void System::initialize(Material& material, 
                            const GridStructure& gridstructure, 
                            const MPIindex<3>& mpindex, 
                            const kGradient& kgradient, 
                            const Space& gradient_space, 
                            const SystemParameters& parameters)
    {
        parameters_ = parameters;
        num_bands_ = material.H.get_Operator_R().get_nrows();

        allocate_memory(gridstructure, mpindex);
        load_from_material(material,gridstructure);
        bandstructure_.initialize(H0_(Space::k));   
        if ( std::abs(parameters_.opengap) > 0. ) {
            open_gap();
        }
        compute_velocity(kgradient, gradient_space, gridstructure.size());
        compute_DM0(parameters_.occupations);
        compute_wannier_centers(material);
    }

    void System::allocate_memory(const GridStructure& gridstructure, const MPIindex<3>& mpindex)
    {
        PROFILE("System::allocate_operators");
        H0_.initialize_fft(gridstructure.Rgrid(), gridstructure.kgrid(), num_bands_, mpindex, "H0");
        for (auto ix : {0, 1, 2}) {
            std::stringstream tagname; tagname << "r_" << ix;
            r_[ix].initialize_fft(gridstructure.Rgrid(), gridstructure.kgrid(), num_bands_, mpindex, tagname.str() );
        }
        for (auto ix : {0, 1, 2}) {
            std::stringstream tagname; tagname << "Velocity_" << ix;
            Velocity_[ix].initialize_fft(gridstructure.Rgrid(), gridstructure.kgrid(), num_bands_, mpindex, tagname.str() );
        }
        DM0_.initialize_fft(gridstructure.Rgrid(), gridstructure.kgrid(), num_bands_, mpindex, "DM0");
        //Band_energies_.resize();
        wannier_centers_.resize(num_bands_);
    }

    void System:: load_from_material(Material& material,const GridStructure& gridstructure)
    {
        PROFILE("System::load_from_material");

        material.H.dft(gridstructure.kgrid()->get_mesh(), +1);
        for (auto ix : { 0, 1, 2 }) {
            material.r[ix].dft(gridstructure.kgrid()->get_mesh(), +1);
            material.r[ix].get_Operator(Space::k).make_hermitian();
        }
        std::copy(material.H(Space::k).begin(), 
                  material.H(Space::k).end(), 
                  H0_(Space::k).begin());
        for(auto& ix : {0,1,2}) {
            std::copy(material.r[ix].get_Operator(Space::k).begin(),
                      material.r[ix].get_Operator(Space::k).end(),
                      r_[ix].get_Operator(Space::k).begin());
        }
        H0_.lock_space(Space::k);           H0_.go_to_R();
        r_[0].lock_space(Space::k);         r_[0].go_to_R();
        r_[1].lock_space(Space::k);         r_[1].go_to_R();
        r_[2].lock_space(Space::k);         r_[2].go_to_R();
    }

    /// Volume of the unit cell (3D) or area (2D), used to normalize the velocity operator
    static double jacobian(const Matrix<double>& A__, const std::array<int,3>& grid_size__)
    {
        /* check if we are considering a slab or a 3D material, from the k grid */
        double j = 0.0;
        auto dim = 0;
        for (auto& ix : {0,1,2}) {
            dim += ( grid_size__[ix] > 1 ? 1 : 0 );
        }

        if ( dim == 3 ) {
            j = std::abs(A__.determinant());
        }
        else if ( dim == 2 ) {
            /* check which lattice vectors is perpendicular to the slab */
            int perpendicular = -1;
            for (auto& ix : {0,1,2}) {
                if( grid_size__[ix] <= 1 ) {
                    perpendicular = ix;
                    break;
                }
            }
            if(perpendicular != 2) {
                std::stringstream ss;
                ss << "perpendicular = " << perpendicular << " !=2 -> implement me!\n";
                throw std::runtime_error(ss.str());
            }
            j = std::abs(A__(0,0)*A__(1,1) - A__(1,0)*A__(0,1));
        }
        else if ( dim == 1 ) {
            throw std::runtime_error("Implement me!\n");
        }
        return j;
    }

    void System::compute_velocity(const kGradient& kgradient, const Space& gradient_space, const std::array<int,3>& grid_size)
    {
        PROFILE("System::compute_velocity");
        std::vector<Coordinate> direction(3);
        direction[0].initialize(1, 0, 0);
        direction[1].initialize(0, 1, 0);
        direction[2].initialize(0, 0, 1);

        auto det = jacobian(Coordinate::get_Basis(LatticeVectors(Space::R)).get_M(), grid_size);

        for (int ix : { 0, 1, 2 }) {
            Velocity_[ix].lock_space(k);
            Velocity_[ix].get_Operator_k().fill(0.);

            /* V = -i*[r,H0] */
            commutator(Velocity_[ix].get_Operator_k(), -im, r_[ix].get_Operator_k(), H0_.get_Operator_k());

            if (gradient_space == R ) {
                Velocity_[ix].go_to_R();
            }
            /* V += i*R*H0 */
            kgradient.Calculate(1., Velocity_[ix].get_Operator(gradient_space),
                                    H0_.get_Operator(gradient_space), direction[ix], false);
            if (gradient_space == R ) {
                Velocity_[ix].go_to_k();
            }
            /* apply right constants for the normalization of wavefunctions */
            auto& Vk = Velocity_[ix].get_Operator(Space::k);
            #pragma omp parallel for
            for (int iblock = 0; iblock < Vk.get_nblocks(); ++iblock) {
                for (int irow = 0; irow < Vk.get_nrows(); ++irow) {
                    for (int icol = 0; icol < Vk.get_ncols(); ++icol) {
                        Vk(iblock, irow, icol) /= det;
                    }
                }
            }
        }
    }

    /// @brief The equilibrium density matrix is diagonal in the bloch gauge, with the occupations on the diagonal.
    /// We rotate it to the wannier gauge:
    /// @f[
    /// \rho^0_{nm}(\textbf{k}) = \sum_b U_{nb}(\textbf{k}) f_b U^*_{mb}(\textbf{k})
    /// @f]
    void System::compute_DM0(const std::vector<double>& occupations)
    {
        PROFILE("System::compute_DM0");
        if ( int(occupations.size()) != num_bands_ ) {
            throw std::runtime_error("System::compute_DM0: occupations size differs from number of bands");
        }
        auto& DM0k = DM0_.get_Operator(Space::k);
        auto& U = bandstructure_.U();
        #pragma omp parallel for
        for (int ik = 0; ik < DM0k.get_nblocks(); ++ik) {
            for (int irow = 0; irow < num_bands_; ++irow) {
                for (int icol = 0; icol < num_bands_; ++icol) {
                    std::complex<double> value = 0.;
                    for (int ib = 0; ib < num_bands_; ++ib) {
                        value += U[ik](irow, ib) * occupations[ib] * std::conj(U[ik](icol, ib));
                    }
                    DM0k(ik, irow, icol) = value;
                }
            }
        }
        DM0_.lock_gauge(wannier);
        DM0_.lock_space(Space::k);
        /* after this, both k and R components are valid */
        DM0_.go_to_R();
    }

    void System::initialize_device()
    {
        H0_.initialize_device();
        H0_.transfer_to(Processor::device);
        for (auto ix : {0, 1, 2}) {
            r_[ix].initialize_device();
            r_[ix].transfer_to(Processor::device);
        }
        DM0_.initialize_device();
        DM0_.transfer_to(Processor::device);
    }

    /// @brief Wannier centers from @f$ \langle n\textbf{0}|\textbf{r}|n\textbf{0} \rangle @f$.
    /// We use the operator of the material because it is not distributed among the ranks: 
    /// r_ is, and the origin R=0 is only in one of them.
    /// @brief Opens the gap by shifting rigidly the bands in the bloch gauge:
    /// @f[
    /// \varepsilon_{n\textbf{k}} \rightarrow \varepsilon_{n\textbf{k}} \mp \frac{\Delta}{2}
    /// @f]
    /// with - for valence bands and + for conduction bands. Eigenvectors are unchanged, and H0 is
    /// rebuilt in the wannier gauge as @f$ H_0(\textbf{k}) = U(\textbf{k}) \varepsilon(\textbf{k}) U^\dagger(\textbf{k}) @f$
    void System::open_gap()
    {
        PROFILE("System::open_gap");
        std::vector<double> shift(num_bands_);
        for (int ib = 0; ib < num_bands_; ++ib) {
            shift[ib] = ( ib < parameters_.filledbands ? -1. : +1. ) * parameters_.opengap / 2.;
        }
        bandstructure_.shift_energies(shift);

        auto& U = bandstructure_.U();
        auto& energies = bandstructure_.energies();
        auto& H0k = H0_.get_Operator(Space::k);
        #pragma omp parallel for
        for (int ik = 0; ik < H0k.get_nblocks(); ++ik) {
            for (int irow = 0; irow < num_bands_; ++irow) {
                for (int icol = 0; icol < num_bands_; ++icol) {
                    std::complex<double> value = 0.;
                    for (int ib = 0; ib < num_bands_; ++ib) {
                        value += U[ik](irow, ib) * energies[ik](ib) * std::conj(U[ik](icol, ib));
                    }
                    H0k(ik, irow, icol) = value;
                }
            }
        }
        H0_.lock_space(Space::k);
        H0_.go_to_R();
    }

    /// Gathers on rank 0 of comm__ the R blocks of an operator, distributed in contiguous slabs among the ranks
    static mdarray<std::complex<double>,3> gather_R(const BlockMatrix<std::complex<double>>& OR__, const int nR_total__, 
                                                    const mpi::Communicator& comm__)
    {
        int nbnd = OR__.get_nrows();
        mdarray<std::complex<double>,3> global({nR_total__, nbnd, nbnd});
        int count = OR__.get_nblocks() * nbnd * nbnd;
#ifdef EDUS_MPI
        std::vector<int> counts(comm__.size()), displs(comm__.size(), 0);
        MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm__.communicator());
        for (int irank = 1; irank < comm__.size(); ++irank) {
            displs[irank] = displs[irank-1] + counts[irank-1];
        }
        MPI_Gatherv(const_cast<std::complex<double>*>(&(OR__(0,0,0))), count, MPI_CXX_DOUBLE_COMPLEX, 
                    global.data(), counts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX, 0, comm__.communicator());
#else
        std::copy(&(OR__(0,0,0)), &(OR__(0,0,0)) + count, global.begin());
#endif
        return global;
    }

    void System::print_wannier(const std::string& filename__, const GridStructure& gridstructure__, 
                               const mpi::Communicator& comm__) const
    {
        auto& Rgrid = *gridstructure__.Rgrid_GammaCentered();
        int nR = Rgrid.get_TotalSize();

        auto H = gather_R(H0_.get_Operator(R), nR, comm__);
        std::array<mdarray<std::complex<double>,3>, 3> r;
        for (auto ix : {0, 1, 2}) {
            r[ix] = gather_R(r_[ix].get_Operator(R), nR, comm__);
        }
        if ( comm__.rank() != 0 ) {
            return;
        }
        Convert_iterable(H, AuEnergy, ElectronVolt);
        for (auto ix : {0, 1, 2}) {
            Convert_iterable(r[ix], AuLength, Angstrom);
        }

        mdarray<double,2> A({3,3}); 
        for (auto ix : {0,1,2}) {
            for (auto jx : {0,1,2}) {
                A(ix,jx) = Convert(Coordinate::get_Basis(LatticeVectors(R)).get_M()(jx,ix), AuLength, Angstrom);
            }
        }    
        std::vector<int> Degeneracy(nR, 1);
        mdarray<double,2> Rmesh_gamma({nR, 3});
        for (int iR = 0; iR < nR; iR++) {
            for (auto ix : {0,1,2}) {
                Rmesh_gamma(iR,ix) = Rgrid[iR].get(LatticeVectors(R))[ix];
            }
        }
        wann::print(filename__, num_bands_, nR, A, Degeneracy, Rmesh_gamma, H, r);
    }

    void System::print_recap() const
    {
        output::title("ELECTRONIC SYSTEM");
        output::print("# bands                  *", num_bands_);
        output::print("filledbands              *", parameters_.filledbands);
        output::print("opengap                  *", parameters_.opengap, " a.u.",
                                                    Convert(parameters_.opengap, AuEnergy, ElectronVolt), " eV");
        auto& A = Coordinate::get_Basis(LatticeVectors(R)).get_M();
        output::print("          :              *", A(0, 0), A(0, 1), A(0, 2));
        output::print("    A     :              *", A(1, 0), A(1, 1), A(1, 2));
        output::print("          :              *", A(2, 0), A(2, 1), A(2, 2));
        auto& B = Coordinate::get_Basis(LatticeVectors(k)).get_M();
        output::print("          :              *", B(0, 0), B(0, 1), B(0, 2));
        output::print("    B     :              *", B(1, 0), B(1, 1), B(1, 2));
        output::print("          :              *", B(2, 0), B(2, 1), B(2, 2));
        output::print("jacobian(A)              *", jacobian(A, H0_.get_Operator_R().get_MeshGrid()->get_Size()));
        output::print("wannier centers (a.u.)   *");
        for (int iwann = 0; iwann < num_bands_; ++iwann) {
            auto& center = wannier_centers_[iwann].get("Cartesian");
            std::string label = "    center " + std::to_string(iwann);
            label.resize(25, ' ');
            output::print(label + "*", center[0], center[1], center[2]);
        }
        output::stars();
    }

    void System::compute_wannier_centers(Material& material)
    {
        PROFILE("System::compute_wannier_centers");

        auto index_origin = material.r[0].get_Operator_R().get_MeshGrid()->find(Coordinate(0,0,0));
        auto& x0 = material.r[0].get_Operator_R()[index_origin];
        auto& y0 = material.r[1].get_Operator_R()[index_origin];
        auto& z0 = material.r[2].get_Operator_R()[index_origin];
        for(int iwann=0; iwann<x0.get_nrows(); ++iwann) {
            wannier_centers_[iwann] = Coordinate(x0(iwann,iwann).real(), 
                                                 y0(iwann,iwann).real(), 
                                                 z0(iwann,iwann).real());
        }
    }
}// namespace electron