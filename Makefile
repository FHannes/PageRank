CC=gcc -ansi -std=c99 -I./include -I./src

build: src/pagerank.c src/pagerank_sparse.c src/utils.c
	${CC} -o pagerank src/pagerank.c src/utils.c lib/libmcbsp1.1.0.a -pthread -lm -lrt
	${CC} -o pagerank_sparse src/pagerank_sparse.c src/utils.c lib/libmcbsp1.1.0.a -pthread -lm -lrt
	rm -f pagerank.o pagerank_sparse.o

clean:
	rm -f pagerank pagerank_sparse
