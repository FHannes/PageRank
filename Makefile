CC=gcc -ansi -std=c99 -I./include

build: src/pagerank.c src/pagerank_sparse.c
	${CC} -o pagerank src/pagerank.c lib/libmcbsp1.1.0.a -pthread -lm
	${CC} -o pagerank_sparse src/pagerank_sparse.c lib/libmcbsp1.1.0.a -pthread -lm
	rm -f pagerank.o pagerank_sparse.o

clean:
	rm -f pagerank pagerank_sparse
