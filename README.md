# IMAGE CAPTIONING - ANALYSIS

## 1) DESCRIPCIÓN
En este repositorio, implementé un modelo de *Image Captioning* utilizando *Inception V3* como extractor de características (*encoder*), y un *decoder model* que consta de una capa de *Embeddings*, seguida de una *GRU layer* y una capa de clasificación con las palabras del vocabulario, empleando PyTorch.

El objetivo de la implementación se enfoca en evaluar y comparar dos enfoques sobre la generación de captions:
- 1\) La modificación de hiperparámetros de la arquitectura conjunta (Encoder-Decoder).
- 2\) El análisis del efecto de la frecuencia de palabras y su aplicación mediante Class Weights.
  
Además, utilicé estrategias de *decoding* como *Greedy Approach*, *Beam Search* con normalización por longitud y *Diverse Beam Search*, evaluadas a través de la métrica BLEU Score.

----

## 2) DATASET
- El *dataset* utilizado fue **Flickr 8k Dataset**, disponible en *Kaggle*.
- El conjunto de datos contiene 8091 imágenes, cada una anotada con 5 *captions*.


### 2.1) DIVISIÓN DE DATOS
#### 2.1.1) ESTRATEGIA
La división de datos se realizó en base a la cantidad de imágenes para evitar *Data Leakage*, el cual podía ocurrir si la partición recaía sobre el total de *captions*.

#### 2.1.2) DISTRIBUCIÓN DE SPLITS
Las particiones se distribuyeron de la siguiente manera:

<div align="center">

|    Split   | Porcentaje | Imágenes | Captions |
|:----------:|:----------:|:--------:|:--------:|
|    Train   |     70%    |   5564   |   28320  |
| Validation |     20%    |   1618   |   8090   |
|    Test    |     10%    |    809   |   4045   |

</div>

----

## 3) ARQUITECTURA
### 3.1) ENCODER

### 3.2) DECODER
