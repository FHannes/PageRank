#include <mcbsp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

/**
 * The number of threads that will be used.
 */
static size_t nprocs;

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
	bsp_begin(nprocs);

	double *tmp_pagerank = malloc(matrix_size * sizeof(double));
	double *max_diff = malloc(nprocs * sizeof(double));

	// Setup the initial PageRank vector (all pages have equal rank)
	for (int idx = 0; idx < matrix_size; idx++) {
		tmp_pagerank[idx] = size_div;
	}

	// Register allocated memory in BSP
	bsp_push_reg(tmp_pagerank, matrix_size * sizeof(double));
	bsp_push_reg(max_diff, nprocs * sizeof(double));
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
			for (unsigned int pid = 0; pid < nprocs; pid++) {
				bsp_put(pid, &rank, tmp_pagerank, col * sizeof(double), sizeof(double));
			}
			// Determine max difference between old and new rank to determine degree of convergence
			cur_diff = fmax(cur_diff, fabs(tmp_pagerank[col] - rank));
			for (unsigned int pid = 0; pid < nprocs; pid++) {
				bsp_put(pid, &cur_diff, max_diff, bsp_pid() * sizeof(double), sizeof(double));
			}
		}
		bsp_sync();
		// Check if the desired convergence has been achieved
		cur_diff = 0;
		for (unsigned int pid = 0; pid < nprocs; pid++) {
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
	unsigned int print_pr_flag = 0;
	char* path;

	nprocs = bsp_nprocs();

	struct option long_options[] = {
			{"output", no_argument, &print_pr_flag, 1},
			{"epsilon", required_argument, NULL, 'e'},
			{"nprocs", required_argument, NULL, 'n'},
			{"path", required_argument, NULL, 'p'},
			{"help", no_argument, NULL, 'h'},
			{NULL, 0, NULL, 0}
	};

	// Read arguments
	int opt, option_index = 0;
	while ((opt = getopt_long(argc, argv, ":en:p:h", long_options, NULL)) != -1) {
		switch (opt) {
		case 0:
			break;
		case 'e':
			epsilon = 1 / (double) pow(10, atoi(optarg));
			break;
		case 'n':
			nprocs = atoi(optarg);
			if (nprocs <= 0 || nprocs > bsp_nprocs()) {
				nprocs = bsp_nprocs();
				printf("Invalid process count entered, switching to maximum available: %ld", nprocs);
				return 0;
			}
			break;
		case 'p':
			path = optarg;
			break;
		case 'h':
		default:
			fprintf(stderr, "Usage: %s -p path [-e epsilon_precision] [-n nprocs] [--output]\n", argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	// Open matrix market file
	FILE *file = fopen(path, "r");
	if (file == NULL) {
		printf("Unable to open specified matrix market file\n");
		return 0;
	}

	// Read matrix size
	if (fscanf(file, "%ld", &matrix_size) == 0) {
		printf("Unable to read matrix size\n");
		return 0;
	}
	size_div = (double) 1 / matrix_size;
	printf("Hyperlink matrix size: %ld\n", matrix_size);

	// Calculate row offsets for each thread
	offsets = malloc((nprocs + 1) * sizeof(size_t));
	size_t rows_min = matrix_size / nprocs;
	size_t rows_rem = matrix_size % nprocs;
	offsets[0] = 0;
	for (int idx = 0; idx < nprocs; idx++) {
		offsets[idx + 1] = offsets[idx] + rows_min + (idx < rows_rem ? 1 : 0);
	}

	// Initiate the sparse link matrix linked list
	sparse_link_matrix = malloc(nprocs * sizeof(link_node*));

	// Allocate memory for temporary storage of outlink count per url
	size_t *nonzero_vector = malloc(matrix_size * sizeof(size_t));
	memset(nonzero_vector, 0, sizeof(nonzero_vector));

	// Read matrix and setup nonzero vector along with linked lists for each thread
	unsigned int x, y, prev_x = 0, prev_y = 0, idx = 0;
	link_node *active_node;
	while (fscanf(file, "%d", &y) > 0 && fscanf(file, "%d", &x) > 0) {
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
	printf("Process count (#threads): %ld\n", nprocs);

	// Variables to measure elapsed time
	struct timeval start, end;

	// Allocate memory to store the PageRank vector
	pagerank_vector = malloc(matrix_size * sizeof(double));

	// Start PageRank BSP
	gettimeofday(&start, NULL);
	bsp_init(&pr_spmd, argc, argv);
	pr_spmd();
	gettimeofday(&end, NULL);

	// Write time elapsed to calculate the PageRank vector
	double elapsed = (end.tv_sec - start.tv_sec) * 1000 +
			((double) end.tv_usec - start.tv_usec) / 1000;
	printf("Google PageRank calculated in %fms\n", elapsed);
	printf("Elapsed time per element: %fµs\n",
			(elapsed * 1000) / (matrix_size * matrix_size));

	// Write the PageRank vector
	if (print_pr_flag != 0) {
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
	for (int idx = 0; idx < nprocs; idx++) {
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
