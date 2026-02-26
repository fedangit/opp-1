#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1000       // размер системы
#define EPS 1e-6     // точность

double norm(double* x_new, double* x_old, int n) {
    double sum = 0.0;
    for (int i=0; i<n; i++)
        sum += (x_new[i] - x_old[i]) * (x_new[i] - x_old[i]);
    return sqrt(sum);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    int extra = N % size;
    int start = rank * rows_per_proc + (rank < extra ? rank : extra);
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);

    double *A = malloc(local_rows * N * sizeof(double));
    double *x = malloc(N * sizeof(double));
    double *x_new = malloc(N * sizeof(double));
    double *b = malloc(N * sizeof(double));

    // Инициализация A и b одинаково на всех процессах
    srand(42);
    for(int i=0; i<N; i++) {
        b[i] = rand() % 10 + 1;
    }
    for(int i=0; i<local_rows; i++) {
        int global_i = start + i;
        for(int j=0; j<N; j++) {
            if(global_i == j)
                A[i*N + j] = 2.0 + rand() % 10;
            else
                A[i*N + j] = 1.0 + rand() % 5;
        }
    }

    for(int i=0; i<N; i++) x[i] = 0.0;

    double t0 = MPI_Wtime();
    int iter = 0;
    while(1) {
        for(int i=0; i<local_rows; i++) {
            int global_i = start + i;
            double sum = 0.0;
            for(int j=0; j<N; j++)
                if(j != global_i)
                    sum += A[i*N + j] * x[j];
            x_new[global_i] = (b[global_i] - sum) / A[i*N + global_i];
        }

        // собираем x_new со всех процессов
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x_new, local_rows, MPI_DOUBLE, MPI_COMM_WORLD);

        if(norm(x_new, x, N) < EPS) break;

        for(int i=0; i<N; i++) x[i] = x_new[i];
        iter++;
    }
    double t1 = MPI_Wtime();

    if(rank == 0)
        printf("Converged in %d iterations, time = %f s\n", iter, t1 - t0);

    free(A); free(x); free(x_new); free(b);
    MPI_Finalize();
    return 0;
}
