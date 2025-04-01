#include <iostream>
#include <Windows.h>
#include "cblas/include/cblas.h"
#include <immintrin.h>



using namespace std;



double** new_matrix(const int n) {
    double** r = new double* [n];
    double* mem = new double[n * n];
    for (int i = 0; i < n; ++i) {
        r[i] = mem + i * n; // Каждый указатель ссылается на начало строки
    }
    return r;
}

void delete_matrix(double**& m) {
    delete[] m[0]; // Удаляем основной блок памяти
    delete[] m;    // Удаляем массив указателей
    m = nullptr;
}

void mat_transp(const int n, double** a, double** at) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            at[j][i] = a[i][j];
}


void matmul_naive(const int n, double** a, double** b, double** c) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += a[i][k] * b[k][j]; // Умножение строк на столбцы
            c[i][j] = s;
        }
}

/*void print_matrix(const int n, double** matrix) {
    for (int i = 0; i < n; ++i) { // Ограничение по строкам
        for (int j = 0; j < n; ++j) { // Ограничение по столбцам
            cout << matrix[i][j] << " "; // Вывод элемента с табуляцией
        }
        cout << endl; // Переход на новую строку
    }
}*/

void init(const int n, double** matrix) {
    for (int i = 0; i < n;++i) {
        for (int j = 0;j < n;++j) {
            matrix[i][j] = ((double)rand() / RAND_MAX);

        }
    }
}

void init0(const int n, double** matrix) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = 0.0;
        }
    }
}

string mat_equal(const int n, double** a, double** b, float eps = 1.e-6)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (abs(a[i][j] - b[i][j]) > eps)
                return "не равны";
    return "равны";
}
#include <immintrin.h> // Для SIMD

void matmul_transposed_blocked_optimized(const int n, double** a, double** bt, double** c, const int block_size = 64) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                // Ограничение размеров блока
                int i_end = min(i + block_size, n);
                int j_end = min(j + block_size, n);
                int k_end = min(k + block_size, n);

                // Блочное умножение
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        // Инициализация накопителя SIMD
                        __m256d sum_vec = _mm256_setzero_pd();

                        // SIMD-обработка основной части
                        int kk;
                        for (kk = k; kk <= k_end - 4; kk += 4) {
                            __m256d a_vec = _mm256_loadu_pd(&a[ii][kk]);
                            __m256d bt_vec = _mm256_loadu_pd(&bt[jj][kk]);
                            sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, bt_vec));
                        }

                        // Суммируем элементы вектора
                        double temp[4];
                        _mm256_storeu_pd(temp, sum_vec);
                        double sum = temp[0] + temp[1] + temp[2] + temp[3];

                        // Обработка остаточных элементов
                        for (; kk < k_end; ++kk) {
                            sum += a[ii][kk] * bt[jj][kk];
                        }

                        c[ii][jj] += sum; // Добавляем к текущему значению
                    }
                }
            }
        }
    }
}


int main() {
    SetConsoleOutputCP(1251);

	int n = 1024;
    
    double** a = new_matrix(n);
    double** b = new_matrix(n);
    double** bt = new_matrix(n);
    double** c1 = new_matrix(n);
    double** c2 = new_matrix(n);
    double** c3 = new_matrix(n);


    init(n,a);
    init(n,b);
    init0(n, c1);
    init0(n, c2);
    init0(n, c3);


    clock_t start, end;
    double p;

    start = clock();
    matmul_naive(n, a, b,c1);
    end = clock();

    double elapsed_secs;
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    p = 2.0 * (double)n * n * n / elapsed_secs * 1.e-6;

    cout << "Прямое умножение" << endl;
    cout << "===========================================" << endl;
    cout << "Время: " << elapsed_secs << " sec" << endl;
    cout << "Производительность: " << p << " MFlops" << endl;
    cout << "===========================================" << endl << endl;

    start = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, &a[0][0],n, &b[0][0],n, 0, &c2[0][0],n);
    end = clock();

    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    p = 2.0 * (double)n * n * n / elapsed_secs * 1.e-6;
    double q = elapsed_secs;

    cout << "Cblas" << endl;
    cout << "===========================================" << endl;
    cout << "Время: " << elapsed_secs << " sec "<< endl;
    cout << "Производительность: " << p << " MFlops" << endl;
    cout << "===========================================" << endl << endl;

    mat_transp(n, b, bt);

    start = clock();
    matmul_transposed_blocked_optimized(n, a, bt, c3);
    end = clock();

    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    double q2 = elapsed_secs;
    p = 2.0 * (double)n * n * n / elapsed_secs * 1.e-6;

    cout << "Вольный метод" << endl; 
    cout << "===========================================" << endl ;
    cout << "Время: " << elapsed_secs << " sec " << endl;
    cout << "Производительность: " << p << " MFlops" << endl;
    cout << "===========================================" << endl << endl;

    cout << "Эффективность вольного метода относительно CBlas = " << q / q2 * 100 << "%" << endl << endl;

    cout << "C1 (прямое умножение) и C2 (CBlas): " << mat_equal(n, c1, c2, 1.e-2) << endl;
    cout << "C2 (CBlas) и C3 (вольный метод): " << mat_equal(n, c2, c3, 1.e-2) << endl;
    cout << "C1 (прямое умножение) и C3 (вольный метод): " << mat_equal(n, c1, c3, 1.e-2) << endl << endl;


    cout << " -------------------------" << endl;
    cout << "| Ломинога Кирилл Юрьевич |" << endl;
    cout << "| 090301-ПОВа-О24         |" << endl;
    cout << " -------------------------" << endl;

    delete_matrix(a);
    delete_matrix(b);
    delete_matrix(bt);
    delete_matrix(c1);
    delete_matrix(c2);
    delete_matrix(c3);

    return 0;
}