CC = mpicc
CFLAGS = --std=c99 -g -O3
LIBS = -lm
OBJECTS = main.o hw2harness.o
EXECUTABLE = cgsolve

$(EXECUTABLE) : $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LIBS)

%.o : %.c
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm -f *.o $(EXECUTABLE)