#include <vector>
#include <queue>
#include <stdexcept>
#include <limits>
#include <algorithm>

#ifndef SPK_TEMPQUEUES_H
#define SPK_TEMPQUEUES_H

namespace SPPARKS_NS {


class DoubleQueueContainer {
public:
    // Default constructor
    DoubleQueueContainer() = default;

    // Method to initialize the container with specified dimensions
    void initialize(size_t xSize, size_t ySize, size_t zSize) {
        temperatureQueues.resize(xSize); // Resize the outer vector (x dimension)
        for (size_t x = 0; x < xSize; ++x) {
            temperatureQueues[x].resize(ySize); // Resize the middle vector (y dimension)
            for (size_t y = 0; y < ySize; ++y) {
                temperatureQueues[x][y].resize(zSize); // Resize the inner vector (z dimension)
            }
        }
    }

    // Access operator to get a reference to a specific queue
    std::queue<double>& operator()(size_t x, size_t y, size_t z) {
        if (x >= temperatureQueues.size() || y >= temperatureQueues[x].size() || z >= temperatureQueues[x][y].size()) {
            throw std::out_of_range("Index out of range");
        }
        return temperatureQueues[x][y][z];
    }

    // Function to get the size of the x dimension
    size_t xSize() const {
        return temperatureQueues.size();
    }

    // Function to get the size of the y dimension for a given x index
    size_t ySize(size_t x) const {
        if (x >= temperatureQueues.size()) {
            throw std::out_of_range("Index out of range");
        }
        return temperatureQueues[x].size();
    }

    // Function to get the size of the z dimension for a given x and y index
    size_t zSize(size_t x, size_t y) const {
        if (x >= temperatureQueues.size() || y >= temperatureQueues[x].size()) {
            throw std::out_of_range("Index out of range");
        }
        return temperatureQueues[x][y].size();
    }

    // Function to find the smallest value at the front of the queues and synchronize it across processors
    double findAndSyncSmallestFrontValue(MPI_Comm comm) {
        double localMin = std::numeric_limits<double>::max(); // Initialize local minimum to max double

        // Iterate through all queues to find the smallest front value
        for (size_t x = 0; x < xSize(); ++x) {
            for (size_t y = 0; y < ySize(x); ++y) {
                for (size_t z = 0; z < zSize(x, y); ++z) {
                    std::queue<double>& q = (*this)(x, y, z);
                    if (!q.empty()) { // Check if the queue is not empty
                        double frontValue = q.front();
                        localMin = std::min(localMin, frontValue); // Update local minimum
                    }
                }
            }
        }

        // Synchronize the minimum value across all processors
        double globalMin;
        MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);

        return globalMin; // Return the global minimum value
    }

private:
    // 3D array of queues
    std::vector<std::vector<std::vector<std::queue<double>>>> temperatureQueues;
};
}
#endif