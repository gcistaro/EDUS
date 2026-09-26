#ifndef LASER_HPP
#define LASER_HPP

//#include <math.h>

#include "Constants.hpp"
#include "GlobalFunctions.hpp"
#include "ConvertUnits.hpp"
#include "Geometry/Coordinate.hpp"
#include "TimeGrid/TimeGrid.hpp"

struct period
{   
    double value;
    period(const double& value_) : value(value_){};
};

struct frequency
{
    double value;
    frequency(const double& value_) : value(value_){};
};

struct wavelength
{
    double value;
    wavelength(const double& value_) : value(value_){};
};

//next three are the equations we will use for the others:
double get_period(const frequency& omega_);
double get_frequency(const wavelength& lambda_);
double get_wavelength(const period& period_);

//using previous three equatioons
double get_period(const wavelength& lambda_);
double get_wavelength(const frequency& omega_);
double get_frequency(const period& period_);



class Wave{
    //properties for something that is described by E(t) = Sin(omega*(t-t0)+phi)
    private:
        double Period = 0;
        double Omega = 0;
        double Lambda = 0;
        double InitialTime = 0;
        double Phase = 0;

    public:
        Wave(){};
        double operator()(const double& time);
        void set_Period(const double& Period_);
        void set_Omega(const double& Omega_);
        void set_Lambda(const double& WaveLength_);
        void set_InitialTime(const double& InitialTime_);
        void set_Phase(const double& Phase_);

        double get_Period();
        double get_Omega();
        double get_Lambda();
        double get_InitialTime();
        double get_Phase();
        friend class Laser;
};


class Envelope{
    //properties for something that is described by E(t) = Sin^2((t-t0)/T) for 0<t-t0<T
    private:
        double Duration = 0;
        double InitialTime = 0;
    
    public:
        double operator()(const double& time);
        void set_Duration(const double& Duration_);
        void set_InitialTime(const double& InitialTime_);

        double get_Duration();
        double get_InitialTime();
        double get_FinalTime();

        friend class Laser;
};


class Laser{
    private:    
        std::vector<Coordinate> Values;
        std::shared_ptr<TimeGrid> timegrid;

        Wave PlaneWave;
        Envelope envelope;

        double NumberOfCycles = 0;
        double Intensity = 0;
        double Amplitude = 0;
        Coordinate Polarization;//you need this only in cartesian coordinates.

        Coordinate A = Coordinate(0.,0.,0.); 
        double previous_Time = 0.;


    public:
        Laser(){};
        Coordinate operator()(const double& Time);
        Coordinate VectorPotential(const double& Time);
        //setters
        //only one of the following three calculates the others
        void set_Period(const double& Period_, const Unit& InputUnit);
        void set_Omega(const double& Omega_, const Unit& InputUnit);
        void set_Lambda(const double& WaveLength_, const Unit& InputUnit);
        ////////////////////////////////////////////////////////
        void set_InitialTime(const double& InitialTime_, const Unit& InputUnit);
        void set_Phase(const double& Phase_);
        void set_NumberOfCycles(const double& NumberOfCycles_);
        void set_Intensity(const double& Intensity_, const Unit& InputUnit);
        void set_Polarization(const Coordinate& Polarization_);
        void set_TimeGrid(const TimeGrid& TimeGrid_);
        void print_info();

        double get_Duration();
        double get_InitialTime();
        double get_FinalTime();
        double get_NumberOfCycles();
        double get_Lambda();
        double get_Omega();
};


class SetOfLaser
{
    private:
        std::vector<Laser> LaserArray;
    public:
        SetOfLaser(){};
        Laser& operator[](const int& i)
        {
            return LaserArray[i];
        };
        const Laser& operator[](const int& i) const
        {
            return LaserArray[i];
        };

        auto operator()(const double& Time) -> Coordinate
        {
            Coordinate result({0,0,0});
            for(auto& laser : LaserArray){
                result += laser(Time);
            }
            return result;
        };

        auto VectorPotential(const double& Time) -> Coordinate
        {
            Coordinate result({0,0,0});
            for(auto& laser : LaserArray){
                result += laser.VectorPotential(Time);
            }
            return result;
        };

        void push_back(const Laser& laser)
        {
            LaserArray.push_back(laser);
        }

        int size()
        {
            return int( LaserArray.size() );
        }
};

#endif