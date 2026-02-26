#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1000       // размер системы
#define EPS 1e-6     // точность

// вычисление евклидовой нормы разности векторов
double norm(double* x_new, double* x_old, int n) {
    double sum = 0.0;
    for (int i=0; i<n; i++)
        sum += (x_new[i] - x_old[i]) * (x_new[i] - x_old[i]);
    return sqrt(sum);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // распределение строк матрицы между процессами
    int rows_per_proc = N / size;
    int extra = N % size;
    int local_rows = rows_per_proc + (rank < extra ? 1 : 0);
    int start = rank * rows_per_proc + (rank < extra ? rank : extra);

    // локальные массивы
    double *A = malloc(local_rows * N * sizeof(double));
    double *x_local = malloc(local_rows * sizeof(double));
    double *b_local = malloc(local_rows * sizeof(double));
    double *x_global = malloc(N * sizeof(double));

    // массивы для MPI_Allgatherv
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    for(int i=0; i<size; i++){
        recvcounts[i] = rows_per_proc + (i < extra ? 1 : 0);
        displs[i] = i*rows_per_proc + (i < extra ? i : extra);
    }

    // инициализация A и b одинаково на всех процессах
    srand(42);
    for(int i=0; i<local_rows; i++){
        int global_i = start + i;
        b_local[i] = rand() % 10 + 1;
        for(int j=0; j<N; j++){
            if(global_i == j)
                A[i*N + j] = 2.0 + rand() % 10;
            else
                A[i*N + j] = 1.0 + rand() % 5;
        }
    }

    // начальное значение x_local = 0
    for(int i=0; i<local_rows; i++) x_local[i] = 0.0;

    int iter = 0;
    double t0 = MPI_Wtime();

    while(1){
        // собираем глобальный вектор x для всех процессов
        MPI_Allgatherv(x_local, local_rows, MPI_DOUBLE,
                       x_global, recvcounts, displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        double max_diff = 0.0;
        // вычисляем новые значения x_local
        for(int i=0; i<local_rows; i++){
            int global_i = start + i;
            double sum = 0.0;
            for(int j=0; j<N; j++)
                if(j != global_i)
                    sum += A[i*N + j] * x_global[j];
            double x_new = (b_local[i] - sum) / A[i*N + global_i];
            double diff = fabs(x_new - x_local[i]);
            if(diff > max_diff) max_diff = diff;
            x_local[i] = x_new;
        }

        // проверка критерия сходимости
        double global_diff;
        MPI_Allreduce(&max_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(global_diff < EPS) break;

        iter++;
    }

    double t1 = MPI_Wtime();

    if(rank == 0)
        printf("Variant 2: Converged in %d iterations, time = %f s\n", iter, t1 - t0);

    free(A); free(x_local); free(b_local); free(x_global);
    free(recvcounts); free(displs);

    MPI_Finalize();
    return 0;
}
