#include <mcbsp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <utils.h>

/**
 * The program arguments.
 */
static struct pr_params params;

/**
 * Size of the link matrix
 */
static size_t matrix_size;

/**
 * 1 / matrix_size, used to avoid recalculating this static value.
 */
static double size_div;

/**
 * A vector containing the number of non-zero elements for each row.
 */
static size_t *nonzero_vector;

/**
 * Link matrix which will be transformed into the Google matrix during execution.
 * NOTE: The matrix is stored by column rather than row, to allow for easier inner-
 *       product calculation.
 */
static double *google_matrix;

/**
 * Array offsets for matrix_size over nprocs processors.
 */
static size_t *offsets;

/**
 * A constant factor (between 0 and 1) which indicates how often users will stay
 * within the hyperlink network, rather than jump to a random page.
 */
static const double ALPHA = 0.9;

/**
 * The actual PageRank vector.
 */
static double *pagerank_vector;

/**
 * The single-program multiple-data method for setting up the Google matrix. In this
 * method BSP is used as a simple multi-threaded environment without data
 * synchronization. Each thread is assigned a number of columns of the matrix to work
 * on, as a result no race conditions can exist.
 */
void gm_spmd() {
	// Start BSP program (Simple threaded program, no race conditions)
	bsp_begin(params.nprocs);

	/**
	 * 1) Normalize the link matrix to transform into the hyperlink matrix:
	 * 		The original matrix contains a 1 for each link from one page to another,
	 * 		this step of the process will transform the matrix into a representation
	 * 		of the markov chain model associated with the original link matrix. This
	 * 		newly formed matrix is referred to as the hyperlink matrix.
	 * 2) Stochasticity adjustment:
	 * 		The hyperlink matrix can contain rows with only zeros when the associated
	 * 		page does not link to any other page or at least not any page that is
	 * 		present in the matrix. When running the PageRank algorithm, these rows
	 * 		create "rank sinks". As the rank of a page is determined by the rank of
	 * 		the incoming pages, these end-nodes in the graph will accumulate more and
	 * 		more of the rank and thereby monopolize it, as they never share it to any
	 * 		of the other pages. The stochasticity adjustment counters this issue by
	 * 		emulating an outlink from these pages to all other pages, causing their
	 * 		rank to be distributed across all other pages each iteration.
	 * 3) Primitivity adjustment:
	 *		The stochastic matrix can't assure that a positive PageRank value exists
	 *		and that calculation with the power method will converge fast enough. A
	 *		value alpha was derived from the "random surfer" argument which takes
	 *		into account that a surfer may jump to a random page outside of the
	 *		current hyperlink structure. This adjustment also converts the matrix
	 *		into a dense matrix. This implementation of the PageRank algorithm uses
	 *		this dense matrix for calculations, but due to the nature of all of the
	 *		previous operations, there really is no need to store the entire matrix,
	 *		as the updated zero values can easily be determined during runtime.
	 */
	for (int x = offsets[bsp_pid()]; x < offsets[bsp_pid() + 1]; x++) {
		int pos = x * matrix_size;
		for (int y = 0; y < matrix_size; y++) {
			double value = 0;
			int elem_count = nonzero_vector[y];
			if (elem_count != 0) {
				value = google_matrix[pos] / elem_count;
			} else {
				value = size_div;
			}
			value = ALPHA * value + (1 - ALPHA) * size_div;
			google_matrix[pos++] = value;
		}
	}

	// End the BSP program
	bsp_end();
}

/**
 * The single-program multiple-data method which calculates the PageRank vector by
 * using the power method. Each thread processes the inner-product calculations for
 * a number of columns. After an iteration is complete, the new partial PageRank
 * vectors are synced to all threads for the next iteration.
 */
void pr_spmd() {
	// Start BSP program
	bsp_begin(params.nprocs);

	double *tmp_pagerank = malloc(matrix_size * sizeof(double));
	double *max_diff = malloc(params.nprocs * sizeof(double));

	// Setup the initial PageRank vector (all pages have equal rank)
	for (int idx = 0; idx < matrix_size; idx++) {
		tmp_pagerank[idx] = size_div;
	}

	// Register allocated memory in BSP
	bsp_push_reg(tmp_pagerank, matrix_size * sizeof(double));
	bsp_push_reg(max_diff, params.nprocs * sizeof(double));
	bsp_sync();

	// Power method iteration
	while (1) {
		double cur_diff = 0;
		for (int col = offsets[bsp_pid()], pos = col * matrix_size;
				col < offsets[bsp_pid() + 1]; col++) {
			// Calculate new PageRank values
			double rank = 0;
			for (int idx = 0; idx < matrix_size; idx++) {
				rank += tmp_pagerank[idx] * google_matrix[pos++];
			}
			for (unsigned int pid = 0; pid < params.nprocs; pid++) {
				bsp_put(pid, &rank, tmp_pagerank, col * sizeof(double), sizeof(double));
			}
			// Determine max difference between old and new rank to determine degree of convergence
			cur_diff = fmax(cur_diff, fabs(tmp_pagerank[col] - rank));
			for (unsigned int pid = 0; pid < params.nprocs; pid++) {
				bsp_put(pid, &cur_diff, max_diff, bsp_pid() * sizeof(double), sizeof(double));
			}
		}
		bsp_sync();
		// Check if the desired convergence has been achieved
		cur_diff = 0;
		for (unsigned int pid = 0; pid < params.nprocs; pid++) {
			cur_diff = fmax(cur_diff, max_diff[pid]);
		}
		if (cur_diff < params.epsilon) {
			break;
		}
	}

	// Copy all columns to the result vector memory location
	for (int col = offsets[bsp_pid()]; col < offsets[bsp_pid() + 1]; col++) {
		pagerank_vector[col] = tmp_pagerank[col];
	}

	// Free the locally allocated memory
	bsp_pop_reg(tmp_pagerank);
	free(tmp_pagerank);
	bsp_pop_reg(max_diff);
	free(max_diff);

	// End the BSP program
	bsp_end();
}

int main(int argc, char **argv) {
	// Parse commandline arguments
	parse_arguments(argc, argv, &params);

	// Open matrix market file
	FILE *file = fopen(params.path, "r");
	if (file == NULL) {
		printf("Unable to open specified matrix market file\n");
		exit(EXIT_FAILURE);
	}

	// Read matrix size
	if (fscanf(file, "%ld", &matrix_size) == 0) {
		printf("Unable to read matrix size\n");
		exit(EXIT_FAILURE);
	}
	if (params.links > 0) {
		// If a different matrix size was requested, use it if it's not larger than
		// the available size in the dataset.
		if (params.links < matrix_size) {
			matrix_size = params.links;
		}
	}
	printf("Hyperlink matrix size: %ld\n", matrix_size);
	double memory_size = matrix_size * matrix_size * sizeof(double);
	memory_size /= 1048576; // Convert B to MiB
	printf("Hyperlink matrix memory size: %fMiB\n", memory_size);

	// Setup constants and allocate memory for Google matrix
	size_div = (double) 1 / matrix_size;
	google_matrix = malloc(matrix_size * matrix_size * sizeof(double));
	memset(google_matrix, 0, sizeof(google_matrix));
	nonzero_vector = malloc(matrix_size * sizeof(size_t));
	memset(nonzero_vector, 0, sizeof(nonzero_vector));

	// Read matrix and setup nonzero vector
	int x, y;
	while (fscanf(file, "%d", &y) > 0 && fscanf(file, "%d", &x) > 0) {
		if (x >= matrix_size || y >= matrix_size) {
			continue;
		}
		google_matrix[x * matrix_size + y] = 1;
		nonzero_vector[y]++;
	}

	// Close matrix market file handle
	fclose(file);

	// Calculate row offsets for each thread
	offsets = malloc((params.nprocs + 1) * sizeof(size_t));
	size_t rows_min = matrix_size / params.nprocs;
	size_t rows_rem = matrix_size % params.nprocs;
	offsets[0] = 0;
	for (int idx = 0; idx < params.nprocs; idx++) {
		offsets[idx + 1] = offsets[idx] + rows_min + (idx < rows_rem ? 1 : 0);
	}

	// Print out the number of processes in use
	printf("Process count (#threads): %ld\n", params.nprocs);

	// Variables to measure elapsed time
	struct timeval start, end;

	// Start Google matrix BSP program
	gettimeofday(&start, NULL);
	bsp_init(&gm_spmd, argc, argv);
	gm_spmd();
	gettimeofday(&end, NULL);

	// Write time elapsed to build Google matrix
	double elapsed = (end.tv_sec - start.tv_sec) * 1000 +
			((double) end.tv_usec - start.tv_usec) / 1000;
	printf("Google matrix built in %fms\n", elapsed);

	// Free nonzero_vector which is no longer needed
	free(nonzero_vector);

	// Allocate memory to store the PageRank vector
	pagerank_vector = malloc(matrix_size * sizeof(double));

	// Allocate memory to store the time measuremetns for the execution of the algorithm
	double *measure_array = malloc(params.measurements * sizeof(double));

	for (unsigned int mm = 0; mm < params.measurements; mm++) {
		// Start PageRank BSP
		gettimeofday(&start, NULL);
		for (unsigned int it = 0; it < params.iterations; it++) {
			bsp_init(&pr_spmd, argc, argv);
			pr_spmd();
		}
		gettimeofday(&end, NULL);

		// Store time elapsed to calculate the PageRank vector
		measure_array[mm] = ((end.tv_sec - start.tv_sec) * 1000 +
				((double) end.tv_usec - start.tv_usec) / 1000) / params.iterations;
	}

	double measurements_mean = 0;
	for (unsigned int mm = 0; mm < params.measurements; mm++) {
		measurements_mean += measure_array[mm];
	}
	measurements_mean /= params.measurements;
	printf("Google PageRank calculated in avg %fms over %u measurements of %u iterations\n",
			measurements_mean, params.measurements, params.iterations);
	printf("Avg elapsed time per element: %fµs\n",
			(measurements_mean * 1000) / (matrix_size * matrix_size));

	// Write the measurements array
	if (params.print_measurements != 0) {
		printf("Measurements:\n(%f", measure_array[0]);
		for (int idx = 1; idx < params.measurements; idx++) {
			printf(",");
			if (idx % 5 == 0) {
				printf("\n");
			}
			printf(" %f", measure_array[idx]);
		}
		printf(")\n");
	}

	// Free the array that holds all of the time measurements
	free(measure_array);

	// Write the PageRank vector
	if (params.print_vector != 0) {
		printf("PageRank vector:\n(%.10f", pagerank_vector[0]);
		for (int idx = 1; idx < matrix_size; idx++) {
			printf(",");
			if (idx % 5 == 0) {
				printf("\n");
			}
			printf(" %.10f", pagerank_vector[idx]);
		}
		printf(")\n");
	}

	// Free memory
	free(google_matrix);
	free(pagerank_vector);
}
