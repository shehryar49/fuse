all:
	g++ src/matrix.cpp src/layer.cpp src/model.cpp src/main.cpp -DNDEBUG -O3 -I include
clean:
	rm -f *.o
	rm a.out
