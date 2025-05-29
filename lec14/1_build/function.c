#include <stdio.h>

void print_hello(int rank, int size) {
    printf("Hello from rank %d out of %d\n", rank, size);
}
