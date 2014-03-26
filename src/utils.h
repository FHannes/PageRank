#ifndef __UTILS_HEADER__
#define __UTILS_HEADER__

#include <mcbsp.h>

struct pr_params {
	/**
	 * True (1) if the PageRank vector has to be printed out after execution.
	 */
	unsigned int print_vector;

	/**
	 * True (1) if the measurement array has to be printed out after execution.
	 */
	unsigned int print_measurements;

	/**
	 * The number of iterations per time measurement. The average execution time of
	 * all iterations counts as one measurement.
	 */
	unsigned int iterations;

	/**
	 * The number of time measurements to capture.
	 */
	unsigned int measurements;

	/**
	 * The number of links (matrix size) to work with.
	 */
	size_t links;

	/**
	 * Path to the matrix market file which will be processed.
	 */
	char* path;

	/**
	 * The accuracy desired for convergence. Defaults to 5 digit precision.
	 */
	double epsilon;

	/**
	 * The number of threads to use to execute the algorithm.
	 */
	size_t nprocs;
};

void parse_arguments(int argc, char **argv, struct pr_params *params);

#endif
