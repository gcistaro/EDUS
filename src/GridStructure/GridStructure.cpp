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
            throw std::runtime_error("k/R grid size must be positive. \n
                                     k grid size: [" + std::to_string(size_[0]) + 
                                     ", " + std::to_string(size_[1]) + 
                                     ", " + std::to_string(size_[2]) + "]\n");
        }
        if (qsize_[ix] <= 0) {
            throw std::runtime_error("q grid size must be positive. \n
                                     q grid size: [" + std::to_string(qsize_[0]) + 
                                     ", " + std::to_string(qsize_[1]) + 
                                     ", " + std::to_string(qsize_[2]) + "]\n");
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
            throw std::runtime_error("q grid is not commensurate with k grid. \n
                                     k grid size: [" + std::to_string(size_[0]) + 
                                     ", " + std::to_string(size_[1]) + 
                                     ", " + std::to_string(size_[2]) + "]\n
                                     q grid size: [" + std::to_string(qsize_[0]) + 
                                     ", " + std::to_string(qsize_[1]) + 
                                     ", " + std::to_string(qsize_[2]) + "]  ");
        }
    }
    q_ = std::make_shared<MeshGrid>(MeshGrid(k, qsize__));
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
