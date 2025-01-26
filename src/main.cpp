/*
 * Copyright (C) 2019 Maitreya Venkataswamy - All Rights Reserved
 */

#include <iostream>

#include "Inputs.h"
#include "Simulation.h"
#include <mpi.h>

/**
 * Main point of execution of the program
 * @param argc number of command line arguments
 * @param argv command line arguments
 * @return 0 if successful, nonzero otherwise
 */
int main(int argc, char** argv) {
    
#ifndef DEBUG
    srand(time(NULL));
#endif

    // Create an Inputs object to contain the simulation parameters
    Inputs inputs = Inputs();
    if (inputs.loadFromFile() != 0) {
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    if (size < 2) { // check for processes number
        std::cerr << "It takes at least 2 processes to run the program!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int road_length = inputs.length;
    int segment_size = road_length / size;
    int remainder = road_length % size; // upologizei to megethos toy dromou gia kathe diergasia

    int start_pos = rank * segment_size + std::min(rank, remainder);
    int end_pos = start_pos + segment_size - 1;
    if (rank < remainder) end_pos++;

    int road_length_per_process = end_pos - start_pos;

    if(rank ==0){
        std::cout << "================================================" << std::endl;
        std::cout << "||    CELLULAR AUTOMATA TRAFFIC SIMULATION    ||" << std::endl;
        std::cout << "================================================" << std::endl;
    }

    // Create a Simulation object for the current simulation
    Simulation* simulation_ptr = new Simulation(inputs, road_length_per_process);

    // Run the Simulation
    simulation_ptr->run_simulation(rank, size, road_length_per_process);

    // Delete the Simulation object
    delete simulation_ptr;

    // Return with no errors
    return 0;
}
