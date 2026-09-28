# IMAGE CAPTIONING - ANALYSIS

## 1) <ins>DESCRIPCIÓN</ins>
En este repositorio, implementé un modelo de *Image Captioning* utilizando *Inception V3* como extractor de características (*encoder*), y un *decoder model* que consta de una capa de *Embeddings*, seguida de una *GRU layer* y una capa de clasificación con las palabras del vocabulario, empleando PyTorch.

El objetivo de la implementación se enfoca en evaluar y comparar dos enfoques sobre la generación de captions:
- 1\) La modificación de hiperparámetros de la arquitectura conjunta (Encoder-Decoder).
- 2\) El análisis del efecto de la frecuencia de palabras y su aplicación mediante Class Weights.
  
Además, utilicé estrategias de *decoding* como *Greedy Approach*, *Beam Search* con normalización por longitud y *Diverse Beam Search*, evaluadas a través de la métrica BLEU Score.

----

## 2) <ins>DATASET</ins>
- El *dataset* utilizado fue **Flickr 8k Dataset**, disponible en *Kaggle*.
- El conjunto de datos contiene 8091 imágenes, cada una anotada con 5 *captions*.


### <ins>2.1) DIVISIÓN DE DATOS</ins>
#### - ESTRATEGIA
La división de datos se realizó en base a la cantidad de imágenes para evitar *Data Leakage*, el cual podía haber ocurrido si la partición recaía sobre el total de *captions*.

#### - DISTRIBUCIÓN DE SPLITS
Las particiones se distribuyeron de la siguiente manera:

<div align="center">

|    Split   | Porcentaje | Imágenes | Captions |
|:----------:|:----------:|:--------:|:--------:|
|    Train   |     70%    |   5564   |   28320  |
| Validation |     20%    |   1618   |   8090   |
|    Test    |     10%    |    809   |   4045   |

<small>**NOTA:** A cada imagen le corresponde 5 *captions*.</small>

</div>

----

## 3) <ins>PREPROCESAMIENTO DE DATOS</ins>

### <ins>3.1) IMAGEN</ins>
El preprocesamiento de las imágenes se realizó de acuerdo al modelo *Inception V3*, y se describe a continuación:

#### 1) **Resize 299**:
  - Redimensionado del lado más pequeño de la imagen para mantener el *ratio* ancho*alto.
  
#### 2) **Center Crop 299**:
  - Recorte central de la imagen para obtener una matriz cuadrada sin alterar el *ratio*.
  
#### 3) **Conversión a tensor**:
  - Transformación de imágenes PIL a tensores de PyTorch.
  
#### 4) **Normalización**:
  - Aplicación de normalización usando *mean*=[0.485,0.456,0.406] y *std*=[0.229,0.224,0.225] sobre los tres canales de la imagen, adaptado al modelo *Inception V3*.

====

### <ins>3.2) TEXTO</ins>

----

## 4) ARQUITECTURA
### 4.1) ENCODER

### 4.2) DECODER
