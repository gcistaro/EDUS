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
    /// @brief q grid for supercell simulations
    std::shared_ptr<MeshGrid> q_;
    /// @brief Size of the q grid on each direction, for supercell simulations
    std::array<int,3> qsize_;
    /// @brief ikq_(ik, iq) = global index of k[ik] + q[iq] in the k grid (q grid commensurate)
    mdarray<int,2> ikq_;
    /// @brief iqmp_(iq, ip) = global index of q[iq] - q[ip] in the q grid
    mdarray<int,2> iqmp_;

public: 
    GridStructure() = default;
    explicit GridStructure(const std::array<int,3>& size__, const std::array<int,3>& qsize__ = {1,1,1});
    void initialize(const std::array<int,3>& size__, const std::array<int,3>& qsize__ = {1,1,1});
    void print_recap() const;

    /* getter functions */
    const std::array<int,3>& size() const {return size_;};
    const mdarray<int,2>& ikq() const {return ikq_;};
    const mdarray<int,2>& iqmp() const {return iqmp_;};
    const std::shared_ptr<MeshGrid>& Rgrid() const {return R_;};
    const std::shared_ptr<MeshGrid>& Rgrid_GammaCentered() const {return RGamma_;};
    const std::shared_ptr<MeshGrid>& kgrid() const {return k_;};
    const std::shared_ptr<MeshGrid>& qgrid() const {return q_;};
};

#endif 

