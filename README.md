# Red Neuronal en PyTorch para CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/torchvision-0.8%2B-green?logo=pytorch)](https://pytorch.org/vision/stable/)
[![Google Colab](https://img.shields.io/badge/Colab-GPU-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-yellowgreen?logo=matplotlib)](https://matplotlib.org/)

Este proyecto consiste en un **notebook de Jupyter** que implementa una red neuronal convolucional (CNN) con **PyTorch** para clasificar imágenes del dataset **CIFAR-10**. El código está pensado para ejecutarse fácilmente en **Google Colab** con soporte para GPU.

---

## Contenido

- **Importación de módulos**: PyTorch, Torchvision, Matplotlib.
- **Carga y transformación de datos**: Descarga automática de CIFAR-10 y preprocesamiento estándar.
- **Definición de una arquitectura CNN moderna**: Varias capas convolucionales, batch normalization y fully connected.
- **Entrenamiento y evaluación**: Entrenamiento durante 20 épocas y reporte de precisión sobre el set de prueba.
- **Guardado del modelo**: Almacenamiento del modelo entrenado en formato `.pth`.
- **(Opcional)** Visualización de la arquitectura de la red con `torchviz`.

---

## Requisitos

- Python 3.8 o superior
- PyTorch 1.7 o superior
- Torchvision 0.8 o superior
- Matplotlib 3.x
- (Opcional) Torchviz para visualizar la arquitectura

Si usas Google Colab, todo está preinstalado excepto `torchviz` (instalable con `!pip install torchviz`).

---

## Ejecución rápida en Google Colab

1. Sube o abre el notebook [`Red_neuronal_PyTorch.ipynb`](./Red_neuronal_PyTorch.ipynb) en Google Colab.
2. Ejecuta cada celda en orden.
3. El modelo se entrenará en CIFAR-10, mostrará la precisión y guardará el modelo como `redneuronal.pth`.

---

## Notas

- La arquitectura propuesta es flexible y puede modificarse fácilmente para experimentar con capas, regularización y optimizadores.
- El modelo alcanza una precisión cercana al **77%** en el set de test tras 20 épocas (puede variar ligeramente).
- Puedes descargar el modelo entrenado desde Colab y reutilizarlo para inferencia o transferencia.

---

## Créditos

Desarrollado con fines educativos para ilustrar la implementación de redes neuronales convolucionales en PyTorch.
