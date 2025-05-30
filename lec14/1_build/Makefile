CC = mpicc
CFLAGS = -O2 -Wall
TARGET = mpi_hello
OBJS = main.o function.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)
