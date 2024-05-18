all:
	g++ matrix.cpp main.cpp -DNDEBUG -O3
clean:
	rm -f *.o
	rm a.out
