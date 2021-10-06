OPT = -O3 -ffast-math
FC = gfortran

.PHONY: all
all: a.out

damping.o: damping.F fsrc.fh
	$(FC) damping.F -c $(OPT) -o $@

main.o: main.c
	$(CC) $^ -c $(OPT) -o $@

%.o: %.cpp
	$(CXX) $^ -c -std=c++11 $(OPT) -o $@

a.out: main.o damp.o damping.o
	$(FC) $^ $(OPT) -o $@

.PHONY: clean
clean:
	rm -f *.o a.out
