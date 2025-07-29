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
```

---

## CPU

Todo el flujo de datos, entrenamiento, inferencia, normalización, retropropagación, funciones de activación y actualización de pesos se ejecuta sobre la **CPU** en C++ puro. El forward y backward del modelo, la atención, el feedforward y el optimizador Adam son completamente gestionados en el procesador principal, asegurando compatibilidad y reproducibilidad en cualquier equipo.

---

## GPU

La **aceleración GPU** se utiliza específicamente para la multiplicación de matrices, el cálculo más costoso del modelo Transformer. Esto se realiza usando CUDA (kernel `matmul_kernel`), acelerando el forward y backward de las capas lineales, mientras que el resto de operaciones del modelo se mantiene en CPU. De esta forma, el código aprovecha el paralelismo masivo de la GPU solo donde realmente se gana velocidad.

---

## Explicación de la atención (Attention) y cómo funciona

El mecanismo de **atención** permite que el modelo asigne diferentes "pesos" a cada parte de la entrada, para cada paso de la secuencia, capturando relaciones globales. En cada bloque Transformer, cada "posición" de la imagen (cada fila, en este caso) puede atender a todas las demás:

- **Q (query):** representa la pregunta de una posición.
- **K (key):** representa lo que cada posición "ofrece".
- **V (value):** la información que puede ser seleccionada.

La atención se calcula como:

Esto produce una combinación ponderada de los valores, enfocando la atención donde el modelo lo decida.  
La **multi-head attention** repite este proceso en paralelo con diferentes proyecciones, permitiendo al modelo capturar distintos tipos de relaciones espaciales.

---

## Arquitectura utilizada

El modelo implementa una arquitectura tipo **Vision Transformer** sencilla y funcional para clasificación de imágenes MNIST. El flujo principal es:

1. **Proyección lineal:** Cada fila de la imagen (28 elementos) se proyecta a un vector de dimensión `d_model`.
2. **Codificación posicional:** Se suma información de la posición a cada vector proyectado.
3. **Bloques Transformer Encoder:** Cada bloque contiene atención multi-cabeza, normalización por capas, feedforward y dropout.
4. **Pooling global:** Se promedia la salida de todas las posiciones.
5. **Capa lineal final:** Predice la clase (0-9) del dígito.

Visualmente, la arquitectura es la siguiente:

![Arquitectura](Arquitectura.png)
