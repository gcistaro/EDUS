#include "GridStructure/GridStructure.hpp"
#include "GlobalFunctions.hpp"

GridStructure::GridStructure(const std::array<int,3>& size__)
{
    initialize(size__);
}

void GridStructure::initialize(const std::array<int,3>& size__)
{
    size_ = size__;
    R_ = std::make_shared<MeshGrid>(MeshGrid(R, size_));
    /* get_GammaCentered_grid still relies on the static MasterRgrid */
    MeshGrid::MasterRgrid = *R_;
    RGamma_ = std::make_shared<MeshGrid>(get_GammaCentered_grid(*R_));
    MeshGrid::MasterRgrid_GammaCentered = *RGamma_;
    k_ = std::make_shared<MeshGrid>(fftPair(*R_));
}


void GridStructure::print_recap() const
{
    output::title("GRID");
    output::print("grid                     *", std::string(8, ' '), "[", size_[0], ", ", size_[1], ", ", size_[2], "]");
    output::print("# R points               *", R_->get_TotalSize());
    output::print("# k points               *", k_->get_TotalSize());
    output::stars();
}
