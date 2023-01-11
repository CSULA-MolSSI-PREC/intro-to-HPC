from mpi4py import MPI
import random
import time
import numpy as np

def monte_carlo_pi(n):
    # Monte Carlo method to approximate pi
    # n: number of random points to generate
    # Returns the approximate value of pi

    # Initialize counter for number of points inside circle
    count = 0

    # Iterate over n random points
    for _ in range(n):
        # Generate random x and y values between -1 and 1
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        # Check if the point lies inside the unit circle (distance from (0, 0) is less than 1)
        if x*x + y*y < 1:
            count += 1

    # Approximate pi by multiplying the ratio of points inside the circle to total points by 4
    return 4.0 * count / n

if __name__ == "__main__":
    # Initialize the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set the number of points to generate per process
    n_samples = 100000000
    count = int(np.floor(n_samples/size))
    remainder = n_samples % size

    if rank == 0:
        # Start time of Monte Carlo algorithm
        start_time = time.perf_counter()

    # Calculate the approximate value of pi
    if rank < remainder:
        pi_approx = monte_carlo_pi(count + 1)
    else:
        pi_approx = monte_carlo_pi(count)

    # Reduce the approximations from all processes to get the final approximation
    pi_final = comm.reduce(pi_approx, op=MPI.SUM)

    if rank == 0:
        # End time of Monte Carlo algorithm
        end_time = time.perf_counter()

        # Elapsed time of Monte Carlo algorithm
        elapsed_time = end_time - start_time

        # Print the final approximation
        print(f"Final approximation of pi: {pi_final / size:.10f}")
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        print(f"Number of processes: {size} ranks")
