#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Estructura para representar una entrada en formato COO (Coordinate List)
typedef struct {
    int fila;
    int columna;
    double valor;
} EntradaCOO;

// Estructura para representar una matriz dispersa en formato COO
typedef struct {
    int num_filas;
    int num_columnas;
    int num_entradas;
    EntradaCOO *entradas;
} MatrizDispersaCOO;

// Función para inicializar una matriz dispersa en formato COO con datos aleatorios
MatrizDispersaCOO *inicializar_matriz_coo_aleatoria(int filas, int columnas, int num_entradas) {
    MatrizDispersaCOO *matriz = (MatrizDispersaCOO *) malloc(sizeof(MatrizDispersaCOO));
    matriz->num_filas = filas;
    matriz->num_columnas = columnas;
    matriz->num_entradas = num_entradas;
    matriz->entradas = (EntradaCOO *) malloc(num_entradas * sizeof(EntradaCOO));

    // Generar entradas aleatorias
    for (int i = 0; i < num_entradas; i++) {
        matriz->entradas[i].fila = rand() % filas;
        matriz->entradas[i].columna = rand() % columnas;
        matriz->entradas[i].valor = (double) rand() / RAND_MAX; // Valor aleatorio entre 0 y 1
    }

    return matriz;
}

// Función para liberar la memoria de una matriz dispersa en formato COO
void liberar_matriz_coo(MatrizDispersaCOO *matriz) {
    free(matriz->entradas);
    free(matriz);
}

// Función para imprimir los primeros elementos de una matriz dispersa en formato COO
void imprimir_primeros_elementos(MatrizDispersaCOO *matriz, const char *nombre, int max_elementos) {
    printf("Matriz %s (Filas: %d, Columnas: %d, Entradas: %d):\n", nombre, matriz->num_filas, matriz->num_columnas, matriz->num_entradas);
    for (int i = 0; i < max_elementos && i < matriz->num_entradas; i++) {
        printf("  Entrada %d: Fila: %d, Columna: %d, Valor: %.2f\n", i + 1, matriz->entradas[i].fila, matriz->entradas[i].columna, matriz->entradas[i].valor);
    }
    if (matriz->num_entradas > max_elementos) {
        printf("  ...\n");
    }
}

// Función para comparar entradas de la matriz por filas y luego por columnas (para ordenación)
int comparar_entradas(const void *a, const void *b) {
    EntradaCOO *entradaA = (EntradaCOO *) a;
    EntradaCOO *entradaB = (EntradaCOO *) b;
    if (entradaA->fila != entradaB->fila) {
        return entradaA->fila - entradaB->fila;
    }
    return entradaA->columna - entradaB->columna;
}

// Función para imprimir la matriz completa ordenada por filas y columnas
void imprimir_matriz_ordenada(MatrizDispersaCOO *matriz, const char *nombre) {
    // Ordenar las entradas por filas y luego por columnas
    qsort(matriz->entradas, matriz->num_entradas, sizeof(EntradaCOO), comparar_entradas);
    
    printf("Matriz %s (Filas: %d, Columnas: %d):\n", nombre, matriz->num_filas, matriz->num_columnas);
    for (int i = 0; i < matriz->num_entradas; i++) {
        printf("  Fila %d Columna %d: Valor: %.2f\n", matriz->entradas[i].fila, matriz->entradas[i].columna, matriz->entradas[i].valor);
    }
}

// Función para multiplicar una fila de A con B Y Colocar el resultado a C
void multiplicar_fila(int fila, MatrizDispersaCOO *A, MatrizDispersaCOO *B, MatrizDispersaCOO *C) {
    for (int i = 0; i < A->num_entradas; i++) {
        if (A->entradas[i].fila == fila) {
            for (int j = 0; j < B->num_entradas; j++) {
                if (B->entradas[j].fila == A->entradas[i].columna) {
                    // Encontrar o añadir el resultado en C
                    int k;
                    int col_idx = B->entradas[j].columna;
                    double valor = A->entradas[i].valor * B->entradas[j].valor;

                    // Asegurar que el acceso a C sea seguro para hilos
                    #pragma omp critical
                    {
                        // Buscar si ya existe una entrada en C para esta fila y columna
                        for (k = 0; k < C->num_entradas; k++) {
                            if (C->entradas[k].fila == fila && C->entradas[k].columna == col_idx) {
                                C->entradas[k].valor += valor;
                                break;
                            }
                        }

                        // Si no se encontró, añadir una nueva entrada
                        if (k == C->num_entradas && valor != 0) {
                            C->entradas[C->num_entradas].fila = fila;
                            C->entradas[C->num_entradas].columna = col_idx;
                            C->entradas[C->num_entradas].valor = valor;
                            C->num_entradas++;
                        }
                    }
                }
            }
        }
    }
}

// Función principal para multiplicar dos matrices dispersas por filas utilizando OpenMP
MatrizDispersaCOO *multiplicacion_matriz_dispersa_paralela(MatrizDispersaCOO *A, MatrizDispersaCOO *B) {
    // Verificar la compatibilidad de las dimensiones
    if (A->num_columnas != B->num_filas) {
        printf("Error: Las dimensiones de las matrices no son compatibles para la multiplicación.\n");
        exit(EXIT_FAILURE);
    }

    // Crear matriz resultante C
    MatrizDispersaCOO *C = (MatrizDispersaCOO *) malloc(sizeof(MatrizDispersaCOO));
    C->num_filas = A->num_filas;
    C->num_columnas = B->num_columnas;
    C->num_entradas = 0;
    C->entradas = (EntradaCOO *) malloc(A->num_entradas * B->num_entradas * sizeof(EntradaCOO));

    // Multiplicar cada fila de A con B de manera paralela
    #pragma omp parallel for
    for (int i = 0; i < A->num_filas; i++) {
        multiplicar_fila(i, A, B, C);
    }

    return C;
}

int main() {
    int filas = 1000;
    int columnas = 1000;
    int num_entradas = 10000; // Número de entradas no cero (aleatorio para este ejemplo)

    // Inicializar matrices dispersas A y B en formato COO con datos aleatorios
    MatrizDispersaCOO *A = inicializar_matriz_coo_aleatoria(filas, columnas, num_entradas);
    MatrizDispersaCOO *B = inicializar_matriz_coo_aleatoria(columnas, filas, num_entradas); // B tiene dimensiones transpuestas para compatibilidad

    // Realizar la multiplicación paralela por filas utilizando OpenMP
    MatrizDispersaCOO *C = multiplicacion_matriz_dispersa_paralela(A, B);

    // Imprimir los primeros elementos de las matrices dispersas A, B y la matriz resultante C
    imprimir_primeros_elementos(A, "A", 5); // Imprimir los primeros 5 elementos de A
    imprimir_primeros_elementos(B, "B", 5); // Imprimir los primeros 5 elementos de B
    imprimir_matriz_ordenada(C, "C"); // Imprimir toda la matriz C ordenada

    // Liberar memoria
    liberar_matriz_coo(A);
    liberar_matriz_coo(B);
    liberar_matriz_coo(C);

    return 0;
}
