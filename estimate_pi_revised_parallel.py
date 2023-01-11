from mpi4py import MPI
import random
import time
import numpy as np

def monte_carlo_revised_pi(n):
    # Monte Carlo method to approximate pi
    # n: number of random points to generate
    # Returns the approximate value of pi

    x = np.float64(np.random.uniform(size=n))
    y = np.float64(np.random.uniform(size=n))

    radii2 = x*x + y*y

    filtered = np.where(radii2<=1.0)
    count = len(radii2[filtered])

    return (4.0 * count / n)

if __name__ == "__main__":
    # Initialize the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # set the number of runs
    n_ests = 200

    # array for values of pi in each run
    pi_ests = []

    # Set the number of points to generate per process
    n_samples = 10000000
    count = int(np.floor(n_samples/size))
    remainder = n_samples % size

    if rank == 0:
        # Start time of Monte Carlo algorithm
        start_time = time.perf_counter()

    # loop over number of runs and add 
    # pi value to the pi_ests array
    for i in range(n_ests):

        # Calculate the approximate value of pi
        if rank < remainder:
            pi_approx = monte_carlo_revised_pi(count + 1)
        else:
            pi_approx = monte_carlo_revised_pi(count)

        # Reduce the approximations from all processes to get the final approximation
        pi_final = comm.reduce(pi_approx, op=MPI.SUM)

        if rank==0:
            pi_ests.append(pi_final/size)

    if rank == 0:
        # End time of Monte Carlo algorithm
        end_time = time.perf_counter()

        # Elapsed time of Monte Carlo algorithm
        elapsed_time = end_time - start_time

        pi_est_mean = np.mean(pi_ests)
        pi_est_std  = np.std(pi_ests)

        # Print the final approximation
        print(f"pi_est_mean = {pi_est_mean :2.10f}, pi_est_std = {pi_est_std :2.10f}, runs = {n_ests:d}, "\
              f"samples = {n_samples :d}, processors: {size} ranks, elapsed time: {elapsed_time:.6f} seconds")
