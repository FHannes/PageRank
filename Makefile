CC=gcc -ansi -std=c99 -I./include

build: src/pagerank.c
	${CC} -o pagerank src/pagerank.c lib/libmcbsp1.1.0.a -pthread -lm
	rm -f pagerank.o

clean:
	rm -f pagerank
