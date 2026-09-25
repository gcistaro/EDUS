#ifndef GRIDSTRUCTURE_HPP
#define GRIDSTRUCTURE_HPP

#include "MeshGrid/MeshGrid.hpp"

class GridStructure
{
private:
    /// @brief Size of the kgrid/Rgrid 
    std::array<int,3> size_;
    /// @brief R grid, where all the operators are defined
    std::shared_ptr<MeshGrid> R_;
    /// @brief R grid, where all the operators are defined, but Gamma centered
    std::shared_ptr<MeshGrid> RGamma_;
    /// @brief k grid, where all the operators are defined
    std::shared_ptr<MeshGrid> k_;

public: 
    GridStructure() = default;
    explicit GridStructure(const std::array<int,3>& size__);
    void initialize(const std::array<int,3>& size__);
    void print_recap() const;

    /* getter functions */
    const std::array<int,3>& size() const {return size_;};
    const std::shared_ptr<MeshGrid>& Rgrid() const {return R_;};
    const std::shared_ptr<MeshGrid>& Rgrid_GammaCentered() const {return RGamma_;};
    const std::shared_ptr<MeshGrid>& kgrid() const {return k_;};
};

#endif 

