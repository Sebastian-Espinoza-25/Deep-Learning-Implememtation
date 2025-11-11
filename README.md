# Deep Learning Implementation - Fashion Product Classifier

Proyecto de implementación de **redes neuronales profundas con Keras + TensorFlow**, desarrollado como parte de la Concentración de IA *.  
El objetivo es entrenar un modelo de **clasificación de prendas de vestir** utilizando *transfer learning* con **MobileNetV3Small** y el dataset **Fashion Product Images (Small)** de Kaggle.

---

## Dataset

**Nombre:** [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)  
**Autor:** Param Aggarwal  
**Licencia:** MIT  
**Tamaño:** ~600 MB  
**Contenido:** 44,000 imágenes de productos de moda con etiquetas como *Tshirts, Shirts, Shoes, Watches, Handbags, Tops, Heels, Sunglasses, Kurtas* y más.

### Descarga del dataset

Tener token de kaggle configurado (~/.kaggle/kaggle.json)

1. Instalar el cliente de Kaggle:
   ```bash
   pip install kaggle
2. Desde la raíz del proyecto, descarga el dataset:
   ```bash
   kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/raw/
3. Descomprimir
   ```bash
   Expand-Archive data/raw/fashion-product-images-small.zip -DestinationPath data/raw/fashion_small -Force
4. El notebook generara automaticamente las divisiones
    ```python
    data/processed/fashion_kaggle_small/train
    data/processed/fashion_kaggle_small/val
    data/processed/fashion_kaggle_small/test

### Instalación y requisitos

Crear y activar entorno virtual e instalar dependencias
```bash
pip install -r requirements.txt

```
### Estructura esperada
```bash
.
├── data/
│   ├── raw/fashion_small/
│   │   ├── images/
│   │   └── styles.csv
│   └── processed/fashion_kaggle_small/
│       ├── train/
│       ├── val/
│       └── test/
├── reports/
│   └── figures/confusion_matrix.png
├── runs/
│   ├── best.keras
│   ├── best_finetune.keras
│   ├── class_names.json
│   ├── train_log.csv
│   └── train_log_finetune.csv
├── deep_learning_implementation.ipynb
├── LICENSE
├── .gitignore
└── README.md
```
### Ejecución del notebook

Ejecutar secuencialmente

El notebook:

Procesa el dataset y genera splits (train/val/test)

Entrena un modelo MobileNetV3Small preentrenado en ImageNet

Aplica fine-tuning sobre el 30 % final del backbone

Evalúa el rendimiento y genera métricas y gráficos

### RESULTADOS OBTENIDOS

Modelo final: MobileNetV3Small (fine-tuned)
Accuracy (test): 0.9230
F1-macro (test): 0.9201

Los reportes y pesos se guardan automáticamente en:

runs/ → Modelos, logs y clases (best.keras, train_log.csv, etc.)

reports/figures/ → Visualizaciones (la matriz de confusión)

### Predicción en imágenes nuevas

Se puede hacer una predicción de una imágen nueva como se muestra en la última celda del notebook.



