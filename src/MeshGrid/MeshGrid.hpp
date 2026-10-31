

#ifndef MESHGRID_HPP
#define MESHGRID_HPP

#include <vector>
#include <algorithm>

#include "Constants.hpp"
#include "Geometry/Coordinate.hpp"
#include "MPIindex/MPIindex.hpp"

struct ShellInfo
{
    double Radius;
    int NumberOfPoints;
    int StartingIndex;
};


enum TypeMeshGrid{cube, read_, sphere, path};
 
class MeshGrid
{
    private:
        std::vector<Coordinate> mesh;
        
        //cube parameter
        std::array<int,3> Size;

        //sphere parameter
        double maxRadius = 0;
        std::vector<ShellInfo> shellinfo;

        //auxiliary vector for out of bounds (empty vector)
        Coordinate EmptyVector;

        int TotalSize=0;
        TypeMeshGrid type;
        Space space;

        int id; //each MeshGrid has a unique id,  to exchange the indices with others
        static int counter_id; //neded to count objects

        double shift_k = 0;


    public:
        static MeshGrid MasterRgrid;
        static MeshGrid MasterRgrid_GammaCentered;

        MPIindex<3> mpindex;
        static std::map<std::array<int,3>, mdarray<int,2> > ConvolutionIndex;//to call it: [{id1,id2,id3}][{iR1,iR3}]

        MeshGrid(){};
        MeshGrid(const MeshGrid& m) = default;
        MeshGrid& operator=(const MeshGrid& mg) = default;

        MeshGrid(MeshGrid&& m) = default;
        MeshGrid& operator=(MeshGrid&& mg) = default;

        MeshGrid(const Space& space__, const std::vector<Coordinate>& ReadMesh); 
        MeshGrid(const Space& space__, const mdarray<double,2>& bare_mg, const std::string& KeyForBasis);
        MeshGrid(const Space& space__, const std::array<int,3>& Size__, const double& shiftk__=0., const bool distributed__=true);
        MeshGrid(const Space& space__, const double& Radius_, const std::array<double, 3>& resolution = {1., 1., 1.});
        MeshGrid(const Space& space__, const std::vector<Coordinate>& PathPoints, const double& resolution);

        void initialize(const Space& space__, const std::array<int,3>& Size__, const double& shiftk__=0., const bool distributed__=true);
        void initialize(const Space& space__, const double& Radius_, const std::array<double, 3>& resolution = {1., 1., 1.});
        void initialize(const Space& space__, const mdarray<double,2>& bare_mg, const std::string& KeyForBasis);
        void initialize(const Space& space__, const std::vector<Coordinate>& PathPoints, const double& resolution);

        std::array<double,3> get_absmax() const;
        
        friend MeshGrid fftPair(const MeshGrid& KnownMG);

        friend MeshGrid Opposite(const MeshGrid& MG);
        
        Coordinate& operator[](const int& i);
        const Coordinate& operator[](const int& i) const;

        int find(const Coordinate& v, int start_index = 0) const;
        Coordinate reduce(const Coordinate& v) const;
        Coordinate reduce(const Coordinate& v, const double& low_limit, const double& up_limit) const;
        Coordinate reduce(const Coordinate& v, const std::array<double,3>& low_limit, const std::array<double,3>& up_limit) const;

        std::pair<int,int> get_shellindices(int shellNumber) const;
        const std::vector<Coordinate>& get_mesh() const;
        const std::array<int,3>& get_Size() const;
        int get_TotalSize() const;
        int get_LocalSize() const;
        int get_id() const;
        static int get_counter_id(){return counter_id;}; //neded to count objects

        static void Calculate_ConvolutionIndex(const MeshGrid& m1, const MeshGrid& m2, const MeshGrid& m3);
        static mdarray<int,2>& get_ConvolutionIndex(const MeshGrid& m1, const MeshGrid& m2, const MeshGrid& m3){
            return ConvolutionIndex[{m1.get_id(), m2.get_id(), m3.get_id()}];
        }
        TypeMeshGrid get_type() const
        {
            return type;
        } 

        friend std::ostream& operator<<(std::ostream& os, const MeshGrid& MG_);
        auto get_space() const { return space; };

        friend MeshGrid get_GammaCentered_grid(const MeshGrid& kmesh__);
        friend class kGradient;
};




#endif
