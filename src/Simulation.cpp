/*
 * Copyright (C) 2019 Maitreya Venkataswamy - All Rights Reserved
 */

#include <chrono>
#include <algorithm>
#include <cmath>
#include "Road.h"
#include "Simulation.h"
#include "Vehicle.h"
#include <mpi.h>
/**
 * Constructor for the Simulation
 * @param inputs
 */
Simulation::Simulation(Inputs inputs, int road_length_per_process) {

    // Create the Road object for the simulation
    this->road_ptr = new Road(inputs, road_length_per_process);
    
    // Initialize the first Vehicle id
    this->next_id = 0;

    // Obtain the simulation inputs
    this->inputs = inputs;

    // Initialize Statistic for travel time
    this->travel_time = new Statistic();
}

/**
 * Destructor for the Simulation
 */
Simulation::~Simulation() {
    // Delete the Road object in the simulation
    delete this->road_ptr;

    // Delete all the Vehicle objects in the Simulation
    for (int i = 0; i < (int) this->vehicles.size(); i++) {
        delete this->vehicles[i];
    }
}

/**
 * Executes the simulation in parallel using the specified number of threads
 * @param num_threads number of threads to run the simulation with
 * @return 0 if successful, nonzero otherwise
 */
int Simulation::run_simulation(int rank, int size, int road_length_per_process) {

    int neighbor_left = rank - 1;
    int neighbor_right = rank + 1;

    int vector_size;
    int total_car_time;

    struct VehicleData {
        int lane;
        int id;
        int position;
        int speed;
        int time_on_road;
    };

    std::chrono::steady_clock::time_point begin;

    begin = std::chrono::steady_clock::now();

    // Set the simulation time to zero
    this->time = 0;

    // Declare a vector for vehicles to be removed each step
    std::vector<int> vehicles_to_remove;
    std::vector<VehicleData> send_to_right;
    std::vector<VehicleData> send_to_left;
    std::vector<int> removed_vehicles_times;
    VehicleData vdata;
    
    std::vector<Vehicle*> local_vehicles;
    std::vector<Lane*> lanes = road_ptr->getLanes();
    std::vector<int> local_vehicles_to_remove;

    while (this->time < this->inputs.max_time) {

        if(rank == 0){
            // Perform the lane switch step for all vehicles
            for (int n = 0; n < (int) this->vehicles.size(); n++) {
                this->vehicles[n]->updateGaps(this->road_ptr);
                this->vehicles[n]->performLaneSwitch(this->road_ptr);
                this->vehicles[n]->updateGaps(this->road_ptr);
            }

            for (int n = 0; n < (int) this->vehicles.size(); n++) {
                int time_on_road = this->vehicles[n]->performLaneMove();

                if (time_on_road != 0) {
                    vehicles_to_remove.push_back(n);
                    vdata = {
                        this->vehicles[n]->getVehicleLane(),
                        this->vehicles[n]->getId(),
                        this->vehicles[n]->getNewPosition(),
                        this->vehicles[n]->getSpeed(),
                        time_on_road
                    };
                    send_to_right.push_back(vdata); 
                }
            }
        }

        if(rank > 0){
            int send_count;
            MPI_Recv(&send_count, 1, MPI_INT, neighbor_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            send_to_right.resize(send_count); // Resize before receiving
            MPI_Recv(send_to_right.data(), send_count * sizeof(VehicleData), MPI_BYTE, neighbor_left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            Vehicle* new_vehicle;

            for (const auto& vdata : send_to_right) {
                for (auto existing_lane : lanes) { 
                    if (existing_lane->getLaneNumber() == vdata.lane) {
                        new_vehicle = new Vehicle(existing_lane, vdata.id, vdata.position, this->inputs);
                    }  
                }
                new_vehicle->setSpeed(vdata.speed);
                new_vehicle->setTimeOnRoad(vdata.time_on_road);
                local_vehicles.push_back(new_vehicle);
            }

            send_to_right.clear();

            for (int i = 0; i < local_vehicles.size(); ++i) {
                auto vehicle = local_vehicles[i];
                                                                    
                vehicle->updateGaps(this->road_ptr);                
                vehicle->performLaneSwitch(this->road_ptr);         
                vehicle->updateGaps(this->road_ptr);

                int prev_pos = vehicle->getPrevPosition();                                       
                int time_on_road = vehicle->performLaneMove();   
                int new_pos = vehicle->getNewPosition();
                
                if (time_on_road != 0) {
                    if(prev_pos + new_pos < road_length_per_process + this->inputs.max_speed ){
                        vdata = {
                            vehicle->getVehicleLane(), 
                            vehicle->getId(), 
                            new_pos, 
                            vehicle->getSpeed(),
                            time_on_road
                        };
                        vehicle->setTempPosition(road_length_per_process);
                    }else{
                        vdata = {
                            vehicle->getVehicleLane(),
                            vehicle->getId(),
                            new_pos,
                            vehicle->getSpeed(),
                            time_on_road
                        };
                        local_vehicles_to_remove.push_back(i);
                        vehicle->getLane()->removeVehicle(new_pos);
                    }
                    send_to_right.push_back(vdata);   
                }

                // Remove and delete vehicles in reverse order to avoid invalidating indices
                std::sort(local_vehicles_to_remove.rbegin(), local_vehicles_to_remove.rend());
                for (int index : local_vehicles_to_remove) {
                    if (this->time > this->inputs.warmup_time) {
                        this->travel_time->addValue(local_vehicles[index]->getTravelTime(this->inputs));
                    }
                    delete local_vehicles[index];
                    local_vehicles.erase(local_vehicles.begin() + index);
                }
                local_vehicles_to_remove.clear();
            }
        }

        if (neighbor_right < size){
            int send_count = send_to_right.size();
            MPI_Send(&send_count, 1, MPI_INT, neighbor_right, 0, MPI_COMM_WORLD);
            MPI_Send(send_to_right.data(), send_count * sizeof(VehicleData), MPI_BYTE, neighbor_right, 0, MPI_COMM_WORLD);
            send_to_right.clear();
        }

        // End of iteration steps
        // Increment time
        this->time++;

        if(rank == 0){
            // Remove finished vehicles
            std::sort(vehicles_to_remove.begin(), vehicles_to_remove.end());
            for (int i = vehicles_to_remove.size() - 1; i >= 0; i--) {
                // Update travel time statistic if beyond warm-up period
                if (this->time > this->inputs.warmup_time) {
                    this->travel_time->addValue(this->vehicles[vehicles_to_remove[i]]->getTravelTime(this->inputs));
                }

                // Delete the Vehicle
                delete this->vehicles[vehicles_to_remove[i]];
                this->vehicles.erase(this->vehicles.begin() + vehicles_to_remove[i]);
            }
            vehicles_to_remove.clear();

            // Spawn new Vehicles
            this->road_ptr->attemptSpawn(this->inputs, &(this->vehicles), &(this->next_id));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double average = this->travel_time->getAverage();
    double variance = this->travel_time->getVariance();
    int numSamples = this->travel_time->getNumSamples();

    if (rank > 0) {
        // Pack the data into an array
        double data[3] = {average, variance, static_cast<double>(numSamples)};
        // Send to rank 0
        MPI_Send(data, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }else {
        // Initialize totals with rank 0's values
        double total_sum = average * numSamples;
        double total_samples = numSamples;
        double total_squared_sum = variance * numSamples + average * average * numSamples;

        // Receive and accumulate data from other ranks
        for (int source = 1; source < size; source++) {
            double received_data[3];
            MPI_Recv(received_data, 3, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Add to totals
            total_samples += received_data[2];
            total_sum += received_data[0] * received_data[2];
            total_squared_sum += (received_data[1] * received_data[2]) + 
                                (received_data[0] * received_data[0] * received_data[2]);
        }

        // Calculate final statistics
        double final_average = total_sum / total_samples;
        double final_variance = (total_squared_sum / total_samples) - 
                            (final_average * final_average);

        // Print the total run time and average iterations per second and seconds per iteration
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto time_elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0;
        std::cout << "--- Simulation Performance ---" << std::endl;
        std::cout << "total computation time: " << time_elapsed << " [s]" << std::endl;
        std::cout << "average time per iteration: " << time_elapsed / inputs.max_time << " [s]" << std::endl;
        std::cout << "average iterating frequency: " << inputs.max_time / time_elapsed << " [iter/s]" << std::endl;

        std::cout << "--- Combined Statistics Across All Processes ---" << std::endl;
        std::cout << "time on road: avg=" << final_average 
                << ", std=" << pow(final_variance, 0.5) 
                << ", N=" << static_cast<int>(total_samples) 
                << std::endl;

    }
    MPI_Finalize();

    // Return with no errors
    return 0;
}
