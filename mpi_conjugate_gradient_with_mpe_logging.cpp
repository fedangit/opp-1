#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <mpe.h> // Добавили библиотеку MPE

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPE_Init_log(); // Инициализация логирования MPE

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Регистрация событий для визуализации
    int event_calc_start = MPE_Log_get_event_number();
    int event_calc_end   = MPE_Log_get_event_number();
    int event_comm_start = MPE_Log_get_event_number();
    int event_comm_end   = MPE_Log_get_event_number();

    if (rank == 0) {
        MPE_Describe_state(event_calc_start, event_calc_end, "Computation", "blue");
        MPE_Describe_state(event_comm_start, event_comm_end, "Communication", "red");
    }

    const int N = 10000;
    const double tau = 0.00001, epsilon = 1e-5;

    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_n = rows_per_proc + (rank < remainder ? 1 : 0);
    int offset = rank * rows_per_proc + (rank < remainder ? rank : remainder);

    // В Варианте 1 векторы b и x имеют полный размер N на каждом процессе
    std::vector<double> b(N, N + 1);
    std::vector<double> x(N, 0.0);
    std::vector<double> next_x_part(local_n);

    std::vector<int> counts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        counts[i] = N / size + (i < N % size ? 1 : 0);
        displs[i] = (i > 0) ? displs[i-1] + counts[i-1] : 0;
    }

    double b_norm_sq = 0;
    for (double v : b) b_norm_sq += v * v;

    int iter = 0;
    bool converged = false;

    while (!converged) {
        iter++;

        // --- ФАЗА ВЫЧИСЛЕНИЙ ---
        MPE_Log_event(event_calc_start, rank, "Compute");
        double local_r_sq = 0;
        for (int i = 0; i < local_n; ++i) {
            int global_i = offset + i;
            double Ax_i = 0;
            for (int j = 0; j < N; ++j) {
                double A_ij = (global_i == j) ? 2.0 : 1.0;
                Ax_i += A_ij * x[j];
            }
            double r_i = Ax_i - b[global_i];
            local_r_sq += r_i * r_i;
            next_x_part[i] = x[global_i] - tau * r_i;
        }
        MPE_Log_event(event_calc_end, rank, "Compute");

        // --- ФАЗА КОММУНИКАЦИИ ---
        MPE_Log_event(event_comm_start, rank, "Sync");
        double global_r_sq;
        MPI_Allreduce(&local_r_sq, &global_r_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Сборка всего вектора x в каждом процессе
        MPI_Allgatherv(next_x_part.data(), local_n, MPI_DOUBLE,
                       x.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPE_Log_event(event_comm_end, rank, "Sync");

        if (std::sqrt(global_r_sq / b_norm_sq) < epsilon) converged = true;
        if (iter > 2000) break; // Ограничение итераций для теста
    }

    // Сохраняем лог (убрал .clog2 из имени, чтобы библиотека сама добавила расширение)
    MPE_Finish_log("solver_profile");

    if (rank == 0) std::cout << "Var 1 finished. Iterations: " << iter << std::endl;

    MPI_Finalize();
    return 0;
}
