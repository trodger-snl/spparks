#include <vector>
#include <queue>

#ifndef SPK_TEMPQUEUES_H
#define SPK_TEMPQUEUES_H

namespace SPPARKS_NS {


class DoubleQueueContainer {
public:

    // Default constructor
    DoubleQueueContainer() = default;

    // Method to initialize the container with specified dimensions
    void initialize(size_t numQueues) {
        temperatureQueues.resize(numQueues);
    }

    // Access operator to get a reference to a specific queue
    std::queue<double>& operator[](size_t index) {
        if (index >= temperatureQueues.size()) {
            throw std::out_of_range("Index out of range");
        }
        return temperatureQueues[index];
    }

    // Function to get the number of queues
    size_t size() const {
        return temperatureQueues.size();
    }

private:
    std::vector<std::queue<double>> temperatureQueues; // Vector of queues
};
}

#endif