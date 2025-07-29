# Clasificador MNIST con Transformer Simple en C++/CUDA

Este proyecto implementa desde cero (sin frameworks de machine learning) un **clasificador de imágenes MNIST** usando una arquitectura tipo **Vision Transformer** en C++ puro con aceleración opcional en GPU vía CUDA. Incluye entrenamiento, evaluación, métricas avanzadas y guardado/recuperación de pesos.

---

## Requisitos

- **C++17** o superior
- **CUDA Toolkit** (nvcc)
- Linux recomendado (probado en Ubuntu)
- Archivos de dataset: `mnist_train.csv`, `mnist_test.csv`, `imagen_prueba.csv`

---

## Compilación

Compila el código con:

```bash
nvcc -O2 -std=c++17 -o main main.cu

./main

