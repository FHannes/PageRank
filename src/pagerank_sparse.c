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
 * Vector storing 1 for dangling nodes and 1 - ALPHA for other nodes. This value is
 * afterwards multiplied with 1 / matrix_size.
 *
 * All of the adjustments made to the initial hyperlink matrix to transform it into
 * the Google matrix can be reduced to a single value for each row in the Google matrix.
 * This value is normally added to each cell of the matrix. Those values are stored
 * in this vector.
 */
static double *adjustment_vector;

/**
 * A node in a linked list to store matrix coordinates.
 */
typedef struct link_node {
	unsigned int x, y;
	struct link_node *next;
} link_node;

/**
 * An array of linked lists with coordinates of links in the hyperlink matrix. Each
 * linked list stored in this array is assigned to a single thread and contains all
 * of the links from columns handled by those threads in the order that they would
 * normally encounter them during the inner-product calculations.
 */
static link_node **sparse_link_matrix;

/**
 * An array containing the pre-calculated weight values for the rows in the link matrix.
 * This is equal to ALPHA divided by the number of outgoing nodes for any given link.
 *
 * The hyperlink matrix always contains the same values (aside from zeros on a single
 * row. Those values are stored in this vector and can be retrieved to learn the value
 * associated with a known link on those rows.
 */
static double *element_weight;

/**
 * Array offsets for matrix_size over nprocs processors.
 */
static unsigned int *offsets;

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
 * The accuracy desired for convergence. Defaults to 5 digit precision.
 */
static double epsilon = 0.00001;

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
		link_node *current_node = sparse_link_matrix[bsp_pid()];
		for (unsigned int col = offsets[bsp_pid()]; col < offsets[bsp_pid() + 1]; col++) {
			// Calculate new PageRank values
			double rank = 0;
			for (int idx = 0; idx < matrix_size; idx++) {
				// Calculate the entry in the Google matrix from precalculated sparse data
				double google_matrix_entry = adjustment_vector[idx];
				if (current_node != NULL && current_node->x == col && current_node->y == idx) {
					google_matrix_entry += element_weight[idx];
					current_node = current_node->next;
				}
				rank += tmp_pagerank[idx] * google_matrix_entry;
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
		if (cur_diff < epsilon) {
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
		return 0;
	}

	// Read matrix size
	if (fscanf(file, "%ld", &matrix_size) == 0) {
		printf("Unable to read matrix size\n");
		return 0;
	}
	if (params.links > 0) {
		// If a different matrix size was requested, use it if it's not larger than
		// the available size in the dataset.
		if (params.links < matrix_size) {
			matrix_size = params.links;
		}
	}
	size_div = (double) 1 / matrix_size;
	printf("Hyperlink matrix size: %ld\n", matrix_size);

	// Calculate row offsets for each thread
	offsets = malloc((params.nprocs + 1) * sizeof(size_t));
	size_t rows_min = matrix_size / params.nprocs;
	size_t rows_rem = matrix_size % params.nprocs;
	offsets[0] = 0;
	for (int idx = 0; idx < params.nprocs; idx++) {
		offsets[idx + 1] = offsets[idx] + rows_min + (idx < rows_rem ? 1 : 0);
	}

	// Initiate the sparse link matrix linked list
	sparse_link_matrix = malloc(params.nprocs * sizeof(link_node*));

	// Allocate memory for temporary storage of outlink count per url
	size_t *nonzero_vector = malloc(matrix_size * sizeof(size_t));
	memset(nonzero_vector, 0, sizeof(nonzero_vector));

	// Read matrix and setup nonzero vector along with linked lists for each thread
	unsigned int x, y, prev_x = 0, prev_y = 0, idx = 0;
	link_node *active_node;
	while (fscanf(file, "%d", &y) > 0 && fscanf(file, "%d", &x) > 0) {
		if (x >= matrix_size || y >= matrix_size) {
			continue;
		}
		if (prev_x * matrix_size + prev_y > x * matrix_size + y) {
			printf("Exception: MatrixMarket has to be sorted by column, not by row");
			return 0;
		}

		// Move to next linked list if column belong to next thread
		while (x >= offsets[idx + 1]) {
			idx++;
		}

		// Create new node
		link_node *node = malloc(sizeof(link_node));
		node->x = x;
		node->y = y;

		// Insert node into appropriate linked list
		if (sparse_link_matrix[idx] == NULL) {
			sparse_link_matrix[idx] = node;
		} else {
			active_node->next = node;
		}
		active_node = node;

		// Add 1 to outlink count
		nonzero_vector[y]++;

		// Store current coordinates for next iteration
		prev_x = x; prev_y = y;
	}

	// Close matrix market file handle
	fclose(file);

	// Initiate element weight vector and dangling node vector
	element_weight = malloc(matrix_size * sizeof(double));
	adjustment_vector = malloc(matrix_size * sizeof(double));
	for (unsigned int idx = 0; idx < matrix_size; idx++) {
		element_weight[idx] = (nonzero_vector[idx] == 0 ? 0 : ALPHA / (double) nonzero_vector[idx]);
		adjustment_vector[idx] = size_div * (nonzero_vector[idx] == 0 ? 1 : 1 - ALPHA);
	}

	// Free temporary storage of outlink count per url
	free(nonzero_vector);

	// Print out the number of processes in use
	printf("Process count (#threads): %ld\n", params.nprocs);

	// Variables to measure elapsed time
	struct timeval start, end;

	// Allocate memory to store the PageRank vector
	pagerank_vector = malloc(matrix_size * sizeof(double));

	// Allocate memory to store the time measurements for the execution of the algorithm
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
	for (int idx = 0; idx < params.nprocs; idx++) {
		link_node *node, *node_next = sparse_link_matrix[idx];
		while ((node = node_next) != NULL) {
			node_next = node->next;
			free(node);
		}
	}
	free(sparse_link_matrix);
	free(element_weight);
	free(adjustment_vector);
	free(pagerank_vector);
}
