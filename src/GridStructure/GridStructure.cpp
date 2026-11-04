#include "GridStructure/GridStructure.hpp"
#include "GlobalFunctions.hpp"

GridStructure::GridStructure(const std::array<int,3>& size__, const std::array<int,3>& qsize__)
{
    initialize(size__, qsize__);
}

void GridStructure::initialize(const std::array<int,3>& size__, const std::array<int,3>& qsize__)
{
    size_ = size__;
    qsize_ = qsize__;
    /* Check on the sizes. All must be > 0 */
    for (auto& ix : {0,1,2}) {
        if (size_[ix] <= 0) {
            std::stringstream ss;
            ss << "k/R grid size must be positive. \n"
               << "k grid size: [" << size_[0] << ", " 
               << size_[1] << ", " << size_[2] << "]\n";
            throw std::runtime_error(ss.str());
        }
        if (qsize_[ix] <= 0) {
            std::stringstream ss;
            ss << "q grid size must be positive. \n"
               << "q grid size: [" << qsize_[0] << ", " 
               << qsize_[1] << ", " << qsize_[2] << "]\n";
            throw std::runtime_error(ss.str());
        }
    }
    R_ = std::make_shared<MeshGrid>(MeshGrid(R, size__));
    /* get_GammaCentered_grid still relies on the static MasterRgrid */
    MeshGrid::MasterRgrid = *R_;
    RGamma_ = std::make_shared<MeshGrid>(get_GammaCentered_grid(*R_));
    MeshGrid::MasterRgrid_GammaCentered = *RGamma_;
    k_ = std::make_shared<MeshGrid>(fftPair(*R_));

    /* Supercell simulations -- check that q grid is commensurate with k grid */
    for (auto& ix : {0,1,2}) {
        if (size_[ix] % qsize_[ix] != 0) {
            std::stringstream ss;
            ss << "q grid is not commensurate with k grid. \n"
               << "k grid size: [" << size_[0] << ", " << size_[1] << ", " << size_[2] << "]\n"
               << "q grid size: [" << qsize_[0] << ", " << qsize_[1] << ", " << qsize_[2] << "]  \n";
            throw std::runtime_error(ss.str());
        }
    }
    /* the q grid is not distributed among the ranks */
    q_ = std::make_shared<MeshGrid>(MeshGrid(k, qsize__, 0., false));

    /* map k -> k + q on the k grid */
    int nk = k_->get_TotalSize();
    int nq = q_->get_TotalSize();
    ikq_.initialize({nk, nq});
    for (int ik = 0; ik < nk; ++ik) {
        for (int iq = 0; iq < nq; ++iq) {
            ikq_(ik, iq) = k_->find((*k_)[ik] + (*q_)[iq]);
            if (ikq_(ik, iq) == -1) {
                std::stringstream ss;
                ss << "k + q not found in the k grid: ik = " << ik << ", iq = " << iq << "\n";
                throw std::runtime_error(ss.str());
            }
        }
    }

    /* map (q, p) -> q - p on the q grid */
    iqmp_.initialize({nq, nq});
    for (int iq = 0; iq < nq; ++iq) {
        for (int ip = 0; ip < nq; ++ip) {
            iqmp_(iq, ip) = q_->find((*q_)[iq] - (*q_)[ip]);
            if (iqmp_(iq, ip) == -1) {
                std::stringstream ss;
                ss << "q - p not found in the q grid: iq = " << iq << ", ip = " << ip << "\n";
                throw std::runtime_error(ss.str());
            }
        }
    }
}


void GridStructure::print_recap() const
{
    output::title("GRID");
    output::print("k/R grid                 *", std::string(8, ' '), "[", size_[0], ", ", size_[1], ", ", size_[2], "]");
    output::print("q grid                   *", std::string(8, ' '), "[", qsize_[0], ", ", qsize_[1], ", ", qsize_[2], "]");
    output::print("# R points               *", R_->get_TotalSize());
    output::print("# k points               *", k_->get_TotalSize());
    output::print("# q points               *", q_->get_TotalSize());
    output::stars();
}
