#include "MeshGrid.hpp"
#include "initialize.hpp"

int MeshGrid::counter_id = 0;
std::map<std::array<int,3>, mdarray<int,2> > MeshGrid::ConvolutionIndex;

MeshGrid MeshGrid::MasterRgrid;
MeshGrid MeshGrid::MasterRgrid_GammaCentered;

/*
MeshGrid::MeshGrid(const MeshGrid& mg)
{
    *this = mg;
}

MeshGrid& MeshGrid::operator=(const MeshGrid& mg)
{
    mesh.resize(mg.mesh.size());
    std::copy(&(mg.mesh[0]),&(mg.mesh[0])+mg.mesh.size(),&(mesh[0]));
    std::copy(&(mg.Size[0]),&(mg.Size[0])+3,&(Size[0]));
    std::copy(&(mg.type),&(mg.type)+1,&type);
    std::copy(&(mg.space),&(mg.space)+1,&space);
    std::copy(&(mg.maxRadius), &(mg.maxRadius)+1, &(maxRadius));
    shellinfo.resize(mg.shellinfo.size());
    std::copy(mg.shellinfo.begin(), mg.shellinfo.end(), shellinfo.begin());
    EmptyVector = mg.EmptyVector;
    TotalSize = mg.TotalSize;
    id = mg.id;
    return *this;
}

MeshGrid::MeshGrid(MeshGrid&& mg)
{
    *this = mg;
}

MeshGrid& MeshGrid::operator=(MeshGrid&& mg)
{
    mesh = std::move(mg.mesh);
    Size = std::move(mg.Size);
    type = std::move(mg.type);
    space = std::move(mg.space);
    maxRadius = std::move(mg.maxRadius);
    shellinfo = std::move(mg.shellinfo);
    EmptyVector = std::move(EmptyVector);
    TotalSize = std::move(mg.TotalSize);
    id = std::move(mg.id);
    return *this;
}
*/


MeshGrid::MeshGrid(const Space& space__, const std::vector<Coordinate>& ReadMesh) :
mesh(ReadMesh), type(read_), space(space__)
{
    TotalSize = mesh.size();
    ++counter_id;
    id = counter_id;
}

MeshGrid::MeshGrid(const Space& space__, const std::array<int,3>& Size__, const double& shiftk__)
{
    this->initialize(space__, Size__, shiftk__);
}

MeshGrid::MeshGrid(const Space& space__, const mdarray<double,2>& bare_mg, const std::string& KeyForBasis)
{
    this->initialize(space__, bare_mg, KeyForBasis);
}

MeshGrid::MeshGrid(const Space& space__, const double& Radius_, const std::array<double, 3>& resolution)
{
    this->initialize(space__, Radius_, resolution);
}

MeshGrid::MeshGrid(const Space& space__, const std::vector<Coordinate>& PathPoint, const double& resolution)
{
    this->initialize(space__, PathPoint, resolution);
}

void MeshGrid::initialize(const Space& space__, const std::vector<Coordinate>& PathPoint, const double& resolution)
{
    id = ++counter_id;
    space = space__;
    int NumberOfLines = PathPoint.size()-1;
    type = path;

    mesh.push_back(PathPoint[0]);
    for(int iline=0; iline<NumberOfLines; ++iline){
        int NumberOfPointsInLine;
        if (resolution != 0.0) NumberOfPointsInLine = (PathPoint[iline+1]-PathPoint[iline]).norm()/resolution;
        else NumberOfPointsInLine = 2;
        for(int it=1; it<NumberOfPointsInLine; ++it){
            double t = double(it)/(NumberOfPointsInLine-1);
            mesh.push_back( (1-t)*PathPoint[iline] + t*PathPoint[iline+1] );
        }
    }
    TotalSize = mesh.size();
}

void MeshGrid::initialize(const Space& space__, const std::array<int,3>& Size__, const double& shiftk__)
{
    //cubic grids are always [0, Size-1] so that are fft friendly
    id = ++counter_id;

    space = space__;
    type=cube;

    Size[0] = Size__[0];
    Size[1] = Size__[1];
    Size[2] = Size__[2];
    shift_k = shiftk__;

    TotalSize = Size[0]*Size[1]*Size[2];
    mesh.resize(TotalSize);

    //defining vectors of the mesh
    switch(space){
        case(k):
        {
            for(int ix0=0; ix0<Size[0]; ix0++){
                for(int ix1=0; ix1<Size[1]; ix1++){
                    for(int ix2=0; ix2<Size[2]; ix2++){
                        int index = ix2 + Size[2] * (ix1 + Size[1] * ix0);
                        mesh[index].initialize((Size[0]==1 ? 0. : shift_k) + double(ix0)/Size[0],
                                               (Size[1]==1 ? 0. : shift_k) + double(ix1)/Size[1],
                                               (Size[2]==1 ? 0. : shift_k) + double(ix2)/Size[2],
                                               LatticeVectors(space));
                    }
                }
            }
            break;
        }
        case(R):
        {
            for(int ix0=0; ix0<Size[0]; ix0++){
                for(int ix1=0; ix1<Size[1]; ix1++){
                    for(int ix2=0; ix2<Size[2]; ix2++){
                        int index = ix2 + Size[2] * (ix1 + Size[1] * ix0);
                        mesh[index].initialize(double(ix0),
                                               double(ix1),
                                               double(ix2),
                                               LatticeVectors(space));
                    }
                }
            }
            break;
        }
    }
    mpindex.initialize(this->get_Size());
}

void MeshGrid::initialize(const Space& space__, const mdarray<double,2>& bare_mg, const std::string& KeyForBasis)
{
    id = ++counter_id;
    space = space__;
    type = read_;

    assert( bare_mg.get_Size(1) == 3 );
    for(int i=0; i<bare_mg.get_Size(0); ++i){
        mesh.push_back(Coordinate(bare_mg(i,0), bare_mg(i,1), bare_mg(i,2), KeyForBasis));
    }
    TotalSize = mesh.size();
}

void MeshGrid::initialize(const Space& space__, const double& Radius_, const std::array<double, 3>& resolution)
{
    PROFILE("MeshGrid::initializeSphere");
    //assert(space == R);
    id = ++counter_id;
    space = space__;
    type = sphere;

    //defining vectors of the mesh
    double i0_threshold, i1_threshold, i2_threshold;

    i0_threshold = 50*resolution[0];
    i1_threshold = 50*resolution[1];
    i2_threshold = 50*resolution[2];
    for(int i0=-i0_threshold; i0<=i0_threshold; i0++){
        for(int i1=-i1_threshold; i1<=i1_threshold; i1++){
            for(int i2=-i2_threshold; i2<=i2_threshold; i2++){
                auto v = Coordinate(double(i0)/resolution[0], double(i1)/resolution[1], double(i2)/resolution[2],
                                           LatticeVectors(space));
                if(v.norm() < Radius_+threshold){
                    mesh.push_back(v);
                }
            }//end i2 loop
        }//end i1 loop
    }//end i0 loop

    std::sort(mesh.begin(), mesh.end(), [](const auto& v1, const auto& v2){ return v1.norm() < v2.norm();});
    TotalSize = mesh.size();

    maxRadius = Radius_;
    //here we order everything by shells
    double maxRadius = -1.;
    for(size_t iR=0; iR<mesh.size(); iR++){
        if(abs(mesh[iR].norm()-maxRadius) > 1.e-07){
            ShellInfo si;
            si.Radius = mesh[iR].norm();
            si.NumberOfPoints = 1;
            si.StartingIndex = iR;
            shellinfo.push_back(si);
            maxRadius = mesh[iR].norm();
        }
        else{
            ++shellinfo[shellinfo.size()-1].NumberOfPoints;
        }
    }
}

std::array<double,3> MeshGrid::get_absmax() const
{
    std::array< std::vector<double>, 3> absCoord;
    for (auto& absCoord_ : absCoord){
        absCoord_.reserve(mesh.size());
    }
    for(auto& vector : mesh){
        auto& CrystalCoordinate = vector.get(LatticeVectors(space));
        absCoord[0].push_back(abs(CrystalCoordinate[0]));
        absCoord[1].push_back(abs(CrystalCoordinate[1]));
        absCoord[2].push_back(abs(CrystalCoordinate[2]));
    }
    std::array<double, 3> maxCoord;
    for(int ix=0; ix<3; ++ix){
        maxCoord[ix] = *(std::max_element(absCoord[ix].begin(), absCoord[ix].end()));
    }
    return maxCoord;
}


MeshGrid fftPair(const MeshGrid& KnownMG)
{
    PROFILE("MeshGrid::fftPair");

    MeshGrid fftMeshGrid;
    auto fftSpace = (KnownMG.get_space() == k ? R : k);
    switch(KnownMG.type)
    {
        case(sphere): case (read_):
        {
            auto maxCoord = KnownMG.get_absmax();
            std::array<int,3> maxCoordInt;
            for(int ix=0; ix<3; ix++){
                maxCoordInt[ix] = 2*int(maxCoord[ix])+1;
		        maxCoordInt[ix] += (maxCoordInt[ix]==0);
            }
            fftMeshGrid.initialize(fftSpace, maxCoordInt);
            break;
        }
        case(cube):
        {
            fftMeshGrid.initialize(fftSpace, KnownMG.get_Size());
            break;
        }
        default:
            break;

    }
    return fftMeshGrid;
}

MeshGrid Opposite(const MeshGrid& MG)
{
    std::vector<Coordinate> MGminus(MG.get_TotalSize());
    for(int i=0; i<MG.get_TotalSize(); i++){
        MGminus[i] = -MG[i];//MG.reduce(-MG[i]);
    }
    return MeshGrid(MG.get_space(), MGminus);
}

const Coordinate& MeshGrid::operator[](const int& i) const
{
    if(i < 0){
        return EmptyVector;
    }
    return mesh[i];
}

Coordinate& MeshGrid::operator[](const int& i)
{
    return const_cast<Coordinate&>(static_cast<const MeshGrid&>(*this)[i]);
}

int MeshGrid::find(const Coordinate& v, int start_index) const
{
    int index = -1;
    switch(type)
    {
        case cube:
        {
            auto v_reduced = reduce(v);
            auto notcart = v_reduced.get(LatticeVectors(space));

            if ( space == k ) {
                for ( auto& ix : { 0, 1, 2 } ) {
                    notcart[ix]= (notcart[ix]-(Size[ix]==1 ? 0. : shift_k) )*Size[ix];
                }
            } 
            index = int(round(notcart[2] + Size[2] * (notcart[1] + Size[1] * notcart[0])));//warning! round is important, if not it will get the integer part, usually wrong.
            break;
	    }
        case sphere:
        {
            if(v.norm() > maxRadius){
                return -1;
            }
            auto shellinfo_iterator = std::find_if(shellinfo.begin(), shellinfo.end(), [&v](const auto& si){ return abs(si.Radius-v.norm()) < 1.e-07; });
            auto BeginShell = mesh.begin()+(*shellinfo_iterator).StartingIndex;
            auto EndShell = mesh.begin()+(*shellinfo_iterator).StartingIndex+(*shellinfo_iterator).NumberOfPoints;
            auto v_iterator = std::find_if(BeginShell, EndShell, [&v](const auto& v_){return (v_-v).norm() < 1.e-07;});
            assert(v_iterator != EndShell);
            index = v_iterator-mesh.begin();
            break;
        }
        default:
        {

            auto v_iterator = std::find_if(mesh.begin()+start_index, mesh.end(), [&v](const auto& v_){ return (v_-v).norm() < 1.e-06;});
            if(v_iterator == mesh.end()){
                return -1;
            }
            index = v_iterator-mesh.begin();
            break;
        }
    }
    if(index >= TotalSize) index = -1;
    return index;
}


Coordinate MeshGrid::reduce(const Coordinate& v) const
{
    assert(type == cube);
    auto notcart = v.get(LatticeVectors(space));
    
    std::array<double,3> low_limit = (space==k) ? std::array<double,3>{shift_k,shift_k,shift_k} 
                                                : std::array<double,3>{0.,0.,0.};
    std::array<double,3> up_limit = (space==k) ? std::array<double,3>{shift_k+1., 
                                                                      shift_k+1., 
                                                                      shift_k+1.}
                                              : std::array<double,3>{double(Size[0]), 
                                                                     double(Size[1]), 
                                                                     double(Size[2])};

    auto v_reduced = reduce(v, low_limit, up_limit);
//    Coordinate v_reduced;
//    v_reduced.initialize(std::fmod(notcart[0], grid_limit[0]),
//                         std::fmod(notcart[1], grid_limit[1]),
//                         std::fmod(notcart[2], grid_limit[2]),
//                         LatticeVectors(space));    
    return v_reduced;
}

Coordinate MeshGrid::reduce(const Coordinate& v, const std::array<double,3>& low_limit, const std::array<double,3>& up_limit) const
{
    //assert(type == cube);

    std::array<int,3> grid_limit = (space==k) ? std::array<int,3>{1, 1, 1}
                                              : Size;
    auto notcart = v.get(LatticeVectors(space));
    for(int ix=0; ix<3; ix++){
        while( notcart[ix] < low_limit[ix] && std::abs( notcart[ix]-low_limit[ix] ) > threshold ){
            notcart[ix] += grid_limit[ix];
        }
        while( notcart[ix] > up_limit[ix] || std::abs(notcart[ix]-up_limit[ix]) < threshold ){
            notcart[ix] -= grid_limit[ix];
        }
    }

    Coordinate v_reduced;
    v_reduced.initialize(notcart[0],
                         notcart[1],
                         notcart[2],
                         LatticeVectors(space));
    return v_reduced;

}

Coordinate MeshGrid::reduce(const Coordinate& v, const double& low_limit, const double& up_limit) const
{
    assert( space == k ||  ( Size[0] == Size[1] && Size[1] == Size[2] ) );
    std::array<double,3> low_limit_array = {low_limit, low_limit, low_limit};
    std::array<double,3> up_limit_array = {up_limit, up_limit, up_limit};
    return this->reduce(v, low_limit_array, up_limit_array);

}


const std::vector<Coordinate>& MeshGrid::get_mesh() const
{
    return mesh;
}


const std::array<int,3>& MeshGrid::get_Size() const
{
    return Size;
}

int MeshGrid::get_TotalSize() const
{
    return TotalSize;
}

int MeshGrid::get_LocalSize() const
{
    auto LocalRange = mpindex.get_LocalRange();
    return LocalRange.second-LocalRange.first + 1;
}

int MeshGrid::get_id() const
{
    return id;
}

void MeshGrid::Calculate_ConvolutionIndex(const MeshGrid& m1, const MeshGrid& m2, const MeshGrid& m3)
{
    PROFILE("MeshGrid::CalculateConvolutionIndex");
    auto i1 = m1.get_id();
    auto i2 = m2.get_id();
    auto i3 = m3.get_id();

    auto& ci = ConvolutionIndex[{i1,i2,i3}];

    ci = mdarray<int,2>({m1.get_TotalSize(), m3.get_TotalSize()});

    #pragma omp parallel for
    for(int iR1=0; iR1<m1.get_TotalSize(); iR1++){
        for(int iR3=0; iR3<m3.get_TotalSize(); iR3++){
                ci(iR1, iR3) = m2.find(m1[iR1]-m3[iR3]);
        }
    }
}

std::pair<int,int> MeshGrid::get_shellindices(int shellNumber) const
{
    std::pair<int,int> indices;
    if(shellNumber < 0){
        shellNumber += shellinfo.size();
    }
    assert(shellNumber >= 0 && shellNumber < int(shellinfo.size()));

    indices.first = shellinfo[shellNumber].StartingIndex;
    indices.second = indices.first + shellinfo[shellNumber].NumberOfPoints;

    return indices;

}


std::ostream& operator<<(std::ostream& os, const MeshGrid& MG_)
{
    for( auto& v : MG_.mesh ){
        os << v.get(LatticeVectors(MG_.get_space())) ;
    }
    os << std::endl;
    return os;
}

MeshGrid get_GammaCentered_grid(const MeshGrid& mesh__)
{
    if (mesh__.type == read_) return mesh__;
    
    auto space = mesh__.get_space();
    auto size_mg = mesh__.get_mesh().size();
    MeshGrid mg;
    std::array<double,3> low_limit;
    std::array<double,3> up_limit;
    for(auto& ix : {0,1,2}) {
        low_limit[ix] = ( space == k ? -0.5 : -MeshGrid::MasterRgrid.get_Size()[ix]/2.);//-mesh__.get_Size()[ix]/2 );
        up_limit[ix] = ( space == k ? 0.5 : MeshGrid::MasterRgrid.get_Size()[ix]/2.);//(mesh__.get_Size()[ix])/2 );
        if(up_limit[ix] == 0) {
            up_limit[ix] = 1;
        }
    } 
    mdarray<double, 2> bare_mg( { int(size_mg), 3 } );

    #pragma omp parallel for
    for( int ik=0; ik<int(size_mg); ++ik ) {
        auto k_ = mesh__[ik];
        k_ = mesh__.reduce(k_, low_limit, up_limit);
        auto kcrys = k_.get( LatticeVectors(space) );
        for( auto& ix : { 0, 1, 2 } ) {
            bare_mg( ik, ix ) = kcrys[ix];
        }
    } 
    mg.initialize( mesh__.get_space(), bare_mg, LatticeVectors(space) );
    mg.type = mesh__.type;
    mg.Size = mesh__.Size;
    mg.mpindex = mesh__.mpindex;
    return mg;
}
