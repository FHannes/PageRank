#include <mcbsp.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Size of the link matrix
static size_t matrix_size;

// 1 / matrix_size, used to avoid recalculating this static value
static double size_div;

// A vector containing the number of non-zero elements for each row
static size_t *nonzero_vector;

/**
 * Link matrix which will be transformed into the Google matrix during execution.
 * NOTE: The matrix is stored by column rather than row, to allow for easier inner-
 *       product calculation.
 */
static double *google_matrix;

// Array offsets for matrix_size over nprocs processors
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
 * The number of iterations of the power method.
 */
static const double ITERATIONS = 50;

void gm_spmd() {
	size_t nprocs = bsp_nprocs();

	// Start BSP program (Simple threaded program, no race conditions)
	bsp_begin(nprocs);

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

void pr_spmd() {
	size_t nprocs = bsp_nprocs();

	// Start BSP program
	bsp_begin(nprocs);

	double *tmp_pagerank = malloc(matrix_size * sizeof(double));

	// Setup the initial PageRank vector (all pages have equal rank)
	for (int idx = 0; idx < matrix_size; idx++) {
		tmp_pagerank[idx] = size_div;
	}

	// Register PageRank vector in BSP
	bsp_push_reg(tmp_pagerank, matrix_size * sizeof(double));
	bsp_sync();

	for (unsigned int it = 0; it < ITERATIONS; it++) {
		for (int col = offsets[bsp_pid()], pos = col * matrix_size;
				col < offsets[bsp_pid() + 1]; col++) {
			double rank = 0;
			for (int idx = 0; idx < matrix_size; idx++) {
				rank += tmp_pagerank[idx] * google_matrix[pos];
				pos++;
			}
			for (unsigned int pid = 0; pid < nprocs; pid++ ) {
				bsp_put(pid, &rank, tmp_pagerank, col * sizeof(double), sizeof(double));
			}
		}
		bsp_sync();
	}

	for (int col = offsets[bsp_pid()]; col < offsets[bsp_pid() + 1]; col++) {
		pagerank_vector[col] = tmp_pagerank[col];
	}

	bsp_pop_reg(tmp_pagerank);
	free(tmp_pagerank);

	// End the BSP program
	bsp_end();
}

int main(int argc, char **argv) {
	size_t nprocs = bsp_nprocs();

	// Setup
	matrix_size = 6;
	size_div = (double) 1 / matrix_size;
	google_matrix = malloc(matrix_size * matrix_size * sizeof(double));
	nonzero_vector = malloc(matrix_size * sizeof(size_t));
	offsets = malloc((nprocs + 1) * sizeof(size_t));
	pagerank_vector = malloc(matrix_size * sizeof(double));

	// Calculate row offsets for each thread
	size_t rows_min = matrix_size / nprocs;
	size_t rows_rem = matrix_size % nprocs;
	offsets[0] = 0;
	for (int idx = 0; idx < nprocs; idx++) {
		offsets[idx + 1] = offsets[idx] + rows_min + (idx < rows_rem ? 1 : 0);
	}

	// Temporary test matrix (transposed)
	double test_matrix[] = {
		0, 0, 1, 0, 0, 0,
		1, 0, 1, 0, 0, 0,
		1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1,
		0, 0, 1, 1, 0, 0,
		0, 0, 0, 1, 1, 0
	};
	size_t test_nonzero[] = {
		2, 0, 3, 2, 2, 1
	};

	// Read link matrix and nonzero vector (to be replaced with matrix market)
	for (int idx = 0; idx < matrix_size * matrix_size; idx++) {
		google_matrix[idx] = test_matrix[idx];
	}
	for (int idx = 0; idx < matrix_size; idx++) {
		nonzero_vector[idx] = test_nonzero[idx];
	}

	// Start Google matrix BSP program
	clock_t start = clock();
	bsp_init(&gm_spmd, argc, argv);
	gm_spmd();
	clock_t end = clock();

	// Write time elapsed to build Google matrix
	double elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Google matrix built in %lfs\n", elapsed);

	// Start PageRank BSP
	start = clock();
	bsp_init(&pr_spmd, argc, argv);
	pr_spmd();
	end = clock();

	// Write time elapsed to calculate the PageRank vector
	elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Google PageRank calculated in %lfs\n", elapsed);

	// Write the PageRank vector
	printf("PageRank vector:\n(%f", pagerank_vector[0]);
	for (int idx = 1; idx < matrix_size; idx++) {
		if (idx % 5 == 0) {
			printf("\n");
		}
		printf(", %f", pagerank_vector[idx]);
	}
	printf(")\n");
}