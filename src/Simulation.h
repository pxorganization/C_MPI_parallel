/*
 * Copyright (C) 2019 Maitreya Venkataswamy - All Rights Reserved
 */

#ifndef CA_TRAFFIC_SIMULATION_SIMULATION_H
#define CA_TRAFFIC_SIMULATION_SIMULATION_H

#include <vector>

#include "Road.h"
#include "Inputs.h"
#include "Statistic.h"

/**
 * Class for the simulation. Has a method for running the simulation.
 */
class Simulation {
private:
    Road* road_ptr;
    int time;
    std::vector<Vehicle*> vehicles;
    Inputs inputs;
    int next_id;
    Statistic* travel_time;
    int rank;
public:
    Simulation(Inputs inputs, int road_length_per_process);
    ~Simulation();
    int run_simulation(int rank, int size, int road_length_per_process);
};


#endif //CA_TRAFFIC_SIMULATION_SIMULATION_H
