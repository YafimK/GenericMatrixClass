CPPFLAGS=-std=c++11 -Wall -Wextra -Wvla -pthread -O
TAR=ex3.tar
FILES=Matrix.hpp Makefile README

all: clean Matrix

Matrix: Matrix.hpp
	$(CXX) $(CPPFLAGS) Matrix.hpp

GMD: GenericMatrixDriver.o Complex.o Matrix.hpp
	$(CXX) $(CPPFLAGS) $^ -o $@

PAR: ParallelChecker.o Complex.o Matrix.hpp
	$(CXX) $(CPPFLAGS) $^ -o $@

tar:
	tar cvf $(TAR) $(FILES)

clean:
	rm -r -f *.o *.gch $(TAR) GMD PAR

.PHONY: clean tar PAR GMD Matrix


.DEFUALT_GOAL := Matrix
