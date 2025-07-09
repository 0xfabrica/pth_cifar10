# CNN para CIFAR-10 con PyTorch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-0.8%2B-green?logo=pytorch)](https://pytorch.org/vision/stable/)
[![Google Colab](https://img.shields.io/badge/Colab-GPU-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-yellowgreen?logo=matplotlib)](https://matplotlib.org/)

---

Este repositorio contiene un notebook técnico que implementa, entrena y evalúa una **red neuronal convolucional (CNN)** en PyTorch para la clasificación de imágenes del conjunto de datos **CIFAR-10**.

## Resumen Técnico

Este proyecto aborda el problema clásico de clasificación de imágenes multiclase (10 clases), utilizando una arquitectura CNN moderna y eficiente. El pipeline cubre desde la adquisición y preprocesado de datos, definición y entrenamiento de la red, hasta la evaluación cuantitativa y visualización del modelo.

### Características Principales

- **Frameworks:** PyTorch, Torchvision, Jupyter, Google Colab (soporte GPU).
- **Dataset:** CIFAR-10 (60,000 imágenes 32x32 RGB en 10 categorías).
- **Arquitectura CNN:**
  - 4 bloques convolucionales (Conv2d + BatchNorm2d + ReLU + MaxPool2d)
  - 3 capas densas (fully-connected) con ReLU
  - Regularización por batch normalization y weight decay
- **Entrenamiento:** SGD con momentum, CrossEntropyLoss, 20 épocas.
- **Evaluación:** Precisión sobre el set de test y gráfico de la arquitectura con torchviz.
- **Exportación:** Guardado del modelo entrenado en formato `.pth`.

---

## Ejemplo de Precisión

> **Accuracy of the network on the 10000 test images: 77 %**

---

## Arquitectura de la Red

A continuación se muestra la arquitectura computacional exacta de la red neuronal utilizada, visualizada con **torchviz** a partir del grafo dinámico de PyTorch:

![image1](image1)

---

## Pipeline de Trabajo

1. **Preprocesamiento y Carga de Datos**
   - Normalización estándar para CIFAR-10.
   - DataLoader con shuffling para entrenamiento.

2. **Definición de la Red**
   - Módulo personalizado `Net` con cuatro etapas convolucionales profundas y batch normalization.
   - Aplanamiento y tres capas lineales para proyección a las 10 clases.

3. **Entrenamiento**
   - SGD (momentum 0.9, weight decay 1e-3, lr 1e-3).
   - 20 épocas, impresión periódica de la función de pérdida.

4. **Evaluación**
   - Cálculo de precisión sobre el set de validación.
   - Visualización del grafo computacional (ver arriba).

5. **Exportación**
   - Guardado del modelo con `torch.save`.

---

## Requisitos y Entorno

- Python >= 3.8
- PyTorch >= 1.7
- Torchvision >= 0.8
- Matplotlib >= 3.x
- (Opcional) torchviz para visualización del grafo

En Google Colab estos requisitos están preinstalados salvo `torchviz`:
```python
!pip install torchviz
```

---

## Uso Rápido en Google Colab

- Abre el notebook [`Red_neuronal_PyTorch.ipynb`](./Red_neuronal_PyTorch.ipynb) en Colab.
- Ejecuta las celdas secuencialmente.
- El modelo será entrenado, evaluado y guardado automáticamente.

---

## Notas de Ingeniería

- La arquitectura es fácilmente modificable para experimentar con regularización, mayor profundidad, otros optimizadores, etc.
- El uso de `BatchNorm2d` en cada bloque acelera la convergencia y estabiliza el entrenamiento.
- La visualización con torchviz permite auditar la arquitectura real ejecutada, facilitando el debugging avanzado.

---

## Créditos

Desarrollado para propósitos educativos y de prototipado rápido en visión por computador, con enfoque en reproducibilidad y buenas prácticas de ingeniería de deep learning.
