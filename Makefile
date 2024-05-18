all:
	g++ matrix.cpp layer.cpp model.cpp main.cpp -DNDEBUG -O3
clean:
	rm -f *.o
	rm a.out
