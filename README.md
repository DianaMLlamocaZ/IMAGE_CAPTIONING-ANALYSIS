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

### <ins>3.2) CAPTIONS</ins>

#### <ins>3.2.1) VOCABULARIO</ins>
- El vocabulario está compuesto por palabras que tienen una frecuencia de aparición mayor o igual a cinco en el conjunto de datos de entrenamiento.
- El vocabulario se encarga de mapear cada palabra con su ID Token.
- Se asignan valores predeterminados para los tokens especiales:<br>
        ```{"<pad>": 0, "<unk>": 1, "<start_seq>": 2, "<end_seq>": 3}```
- Tamaño final del vocabulario: 2463

#### <ins>3.2.2) PREPROCESAMIENTO DE CAPTIONS</ins>
El preprocesamiento de *captions* comprende las etapas de normalización de texto y su conversión a secuencia de ID Tokens como *input* para el *decoder*:

- Conversión a minúsculas
- Eliminación de signos de puntuación
- Limpieza de espacios al inicio y fin del *caption* preprocesado.
- Mapeo del *caption* preprocesado (*string*) a una secuencia de *ID Tokens* (tensor numérico).<br>


> **NOTA**:<br>
> - Si una palabra se encuentra en el *caption* preprocesado y NO en el vocabulario, se le asigna el token "\<unk>".<br>
> - Cada tensor numérico de *ID Tokens* inicia y finaliza con los tokens "<start_seq>" y "<end_seq>", respectivamente.

----

## 4) ARQUITECTURA
La arquitectura del modelo consta de los siguientes componentes:<br>

- **Encoder**:
    - Utiliza el modelo *Inception V3*, como extractor de características, para generar el *embedding* de la imagen.<br>
- **Decoder**:
    - El *decoder* toma como *hidden state* inicial el *embedding* de la imagen generada por el *encoder* y genera el *caption* iterativamente, actualizando el *hidden state* en cada paso.

A continuación, se describe cada componente detalladamente: 

### 4.1) ENCODER
- Primero, se congelan todas las capas del modelo *Inception V3*, ya que se utiliza únicamente como *feature extractor* para generar los *embeddings* de las imágenes.
  
- Se remueve la última capa de clasificación del modelo, y se cambia por una *Identity Layer* que mantiene las 2048 dimensiones resultantes de la transformación de la capa anterior.
  
- Se conecta la *Identity Layer* a una capa lineal que mapea las 2048 dimensiones a '*embed_img_size*' dimensiones, donde '*embed_img_size*' se definió con un valor de 256.

> **NOTA:** El *embedding* de la imagen debe tener la misma cantidad de dimensiones que el *embedding* de cada palabra.

====

### 4.2) DECODER
#### 4.2.1) CAPAS:
- *Embedding Layer*:
    - **Función:** Convertir el tensor de secuencia de *ID tokens* a tensores de secuencia de *embeddings*.
      
    - **Input Size:** Tamaño del vocabulario --> 2463
  
    - **Output Size:** '*emb_text_size*' --> Definido con un valor de 256<br>

> **NOTA:** El *embedding* de la imagen ('*embed_img_size*') y el *embedding* de cada palabra ('*emb_text_size*') tienen el mismo valor porque se concatenan sobre la dimensión de secuencia (dim=1) antes de pasar a la capa GRU.

---
    
- *GRU*:
    - **Función:** Generar el *caption* actualizando iterativamente su *hidden state*.
      
    - **Input Size:** '*embed_img_size*' o *emb_text_size* (tienen el mismo valor).
      
    - **Hidden Size:** '*hidden_size*' --> Definido con un valor de 128 y 256 (misma arquitectura, diferentes hiperparámetros).
        
> **NOTA:** La generación de *captions* es iterativa y a nivel de palabra, iniciando con el *embedding* de la imagen como tensor inicial en el *time step* 0. 

---

- *Classification Layer*:
    - **Función**: Predicción de la siguiente palabra en la generación del *caption*.

    - **Input Size:** '*hidden_size*' --> 128 o 256 (diferentes hiperparámetros).
 
    - **Output Size:** Tamaño del vocabulario --> 2463

> **NOTA:** La generación de *captions* es a nivel de palabra. En ese sentido, la capa de clasificación tiene 2463 neuronas, correspondientes a las palabras del vocabulario.


#### 4.2.2) FLUJO DE GENERACIÓN DE *CAPTIONS*:
- **Dinámica temporal ('n' pasos)**: Para cada *caption*, el proceso se ejecuta mediante un bucle iterativo de 'n' pasos, donde 'n' representa la longitud de la secuencia (o la longitud máxima del *caption* si se utiliza procesamiento en *batches*).
