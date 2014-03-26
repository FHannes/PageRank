#include <utils.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void parse_arguments(int argc, char **argv, struct pr_params *params) {
	params->iterations = 1;
	params->measurements = 1;
	params->links = 0;
	params->print_measurements = 0;
	params->print_vector = 0;
	params->path = NULL;
	params->epsilon = 0.00001;
	params->nprocs = bsp_nprocs();

	struct option long_options[] = {
			{"help", no_argument, NULL, 'h'},
			{"printpr", no_argument, &params->print_vector, 1},
			{"printmm", no_argument, &params->print_measurements, 1},
			{"epsilon", required_argument, NULL, 'e'},
			{"nprocs", required_argument, NULL, 'n'},
			{"path", required_argument, NULL, 'p'},
			{"links", required_argument, NULL, 'l'},
			{"iterations", required_argument, NULL, 'i'},
			{"measurements", required_argument, NULL, 'm'},
			{NULL, 0, NULL, 0}
	};

	// Read arguments
	int opt, option_index = -1;
	while ((opt = getopt_long(argc, argv, ":e:n:p:i:m:l:h:", long_options, &option_index)) != -1) {
		switch (opt) {
		case 'e':
			params->epsilon = 1 / (double) pow(10, atoi(optarg));
			break;
		case 'i':
			params->iterations = atoi(optarg);
			break;
		case 'l':
			params->links = atoi(optarg);
			break;
		case 'm':
			params->measurements = atoi(optarg);
			break;
		case 'n':
			params->nprocs = atoi(optarg);
			if (params->nprocs <= 0 || params->nprocs > bsp_nprocs()) {
				fprintf(stderr, "Invalid process count entered\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 'p':
			params->path = optarg;
			break;
		case 0:
			break;
		case 'h':
		default:
			fprintf(stderr, "Usage: %s -p path [-e epsilon_precision] [-n nprocs] [-i iterations]\n  [-m measurements] [--printpr] [--printmm]\n", argv[0]);
			fprintf(stderr, "Maximum nprocs (threads) available: %u\n", bsp_nprocs());
			exit(EXIT_FAILURE);
		}
	}

	if (params->path == NULL) {
		fprintf(stderr, "Missing mandatory path parameter.\nTo find out more: %s --help\n", argv[0]);
		exit(EXIT_FAILURE);
	}
}
