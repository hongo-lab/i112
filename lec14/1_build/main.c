#include <mpi.h>
#include <stdio.h>

// function.c にある関数のプロトタイプ宣言
extern void print_hello(int rank, int size);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    print_hello(rank, size);

    MPI_Finalize();
    return 0;
}
