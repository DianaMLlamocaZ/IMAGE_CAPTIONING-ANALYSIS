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

## 4) <ins>ARQUITECTURA</ins>
La arquitectura del modelo consta de los siguientes componentes:<br>

- **Encoder**:
    - Utiliza el modelo *Inception V3*, como extractor de características, para generar el *embedding* de la imagen.<br>
- **Decoder**:
    - El *decoder* toma como *hidden state* inicial el *embedding* de la imagen generada por el *encoder* y genera el *caption* iterativamente, actualizando el *hidden state* en cada paso.

A continuación, se describe cada componente detalladamente: 

### <ins>4.1) ENCODER</ins>
- Primero, se congelan todas las capas del modelo *Inception V3*, ya que se utiliza únicamente como *feature extractor* para generar los *embeddings* de las imágenes.
  
- Se remueve la última capa de clasificación del modelo, y se cambia por una *Identity Layer* que mantiene las 2048 dimensiones resultantes de la transformación de la capa anterior.
  
- Se conecta la *Identity Layer* a una capa lineal que mapea las 2048 dimensiones a '*embed_img_size*' dimensiones, donde '*embed_img_size*' se definió con un valor de 256.

> **NOTA:** El *embedding* de la imagen debe tener la misma cantidad de dimensiones que el *embedding* de cada palabra.

====

### <ins>4.2) DECODER</ins>
#### <ins>4.2.1) CAPAS:</ins>
- ***Embedding Layer***:
    - **Función:** Convertir el tensor de secuencia de *ID tokens* a tensores de secuencia de *embeddings*.
      
    - **Input Size:** Tamaño del vocabulario --> 2463
  
    - **Output Size:** '*emb_text_size*' --> Definido con un valor de 256<br>

> **NOTA:** El *embedding* de la imagen ('*embed_img_size*') y el *embedding* de cada palabra ('*emb_text_size*') tienen el mismo valor porque se concatenan sobre la dimensión de secuencia (dim=1) antes de pasar a la capa GRU.

---
    
- ***GRU***:
    - **Función:** Generar el *caption* actualizando iterativamente su *hidden state*.
      
    - **Input Size:** '*embed_img_size*' o *'emb_text_size'* (tienen el mismo valor).
      
    - **Hidden Size:** '*hidden_size*' --> Definido con un valor de 128 y 256 (misma arquitectura, diferentes hiperparámetros).
        
> **NOTA:** La generación de *captions* es iterativa y a nivel de palabra, iniciando con el *embedding* de la imagen como tensor inicial en el *time step* 0. 

---

- ***Classification Layer***:
    - **Función**: Predicción de la siguiente palabra en la generación del *caption*.

    - **Input Size:** '*hidden_size*' --> 128 o 256 (diferentes hiperparámetros).
 
    - **Output Size:** Tamaño del vocabulario --> 2463

> **NOTA:** La generación de *captions* es a nivel de palabra. En ese sentido, la capa de clasificación tiene 2463 neuronas, correspondientes a las palabras del vocabulario.


#### <ins>4.2.2) FLUJO DE GENERACIÓN DE *CAPTIONS*:</ins>
- **Dinámica temporal ('n' pasos)**:
    - Para cada *caption*, el proceso se ejecuta mediante un bucle iterativo de 'n' pasos, donde 'n' representa la cantidad de pasos definidos en la inferencia, o la longitud máxima entre todos los *captions* si se utiliza procesamiento en *batches* para el entrenamiento.
 
- **Paso inicial (t=0)**:
    - Se utiliza el *embedding* de la imagen como el *input* inicial de la capa GRU para establecer el contexto visual y realizar la predicción del primer *token* (que corresponde al token '\<start_seq>' debido a la forma de inferencia definida en la implementación).

- **Propagación y actualización de memoria**:
    - Para t>0, la GRU procesa el *input* actual y actualiza su *hidden state*, manteniendo la memoria activa desde t=0 hasta el paso actual para la generación del *caption*.

- **Criterio de terminación**:
    - **Inferencia:**
        - Las iteraciones continúan de forma secuencial hasta que el modelo prediga el token '\<end_seq>' o si alcanza el límite máximo de pasos definidos.
    - **Entrenamiento:**
        - El bucle se ejecuta durante los 'n' pasos, que representa la longitud máxima entre todos los *captions* del *batch*, aplicando un manejo de *padding* en la función de pérdida (*ignore_padding*) para evitar que el *padding token* afecte el cálculo del gradiente en las secuencias.

----

## 5) <ins>DATASET</ins>
Se creó un *custom dataset*, utilizando la clase predeterminada de PyTorch.

- **Almacenamiento (en \_\_init_\_\):**
    - Carga y almacena, en dos listas, los ID de las imágenes y los *captions*.
      
- **Valores de retorno (en \_\_getitem_\_\):**
    - Preprocesa la imagen y el *caption* con las [funciones de preprocesamiento de datos definidas](#3-preprocesamiento-de-datos).
    - Retorna el tensor de la imagen y el tensor de secuencia de ID Tokens del *caption*, listos para utilizarse en el *Encoder* y *Decoder*.

----

## 6) <ins>DATALOADER</ins>
Se utiliza la clase *DataLoader* de PyTorch para permitir el entrenamiento mediante *batches*, empleando la función auxiliar *collate_fn*:

- **Función:**
    - Permite el entrenamiento paralelo en *batches*.
      
- **Alineación de secuencias (*collate_fn*):**
    - 1\) Se calcula la longitud exacta, de cada tensor de *caption* dentro del *batch*, para determinar la longitud máxima de secuencia.
      
    - 2\) Para cada muestra en el *batch*, se obtiene la diferencia entre la longitud máxima y el tamaño de su *caption*.
      
    - 3\) Se genera un tensor de ceros, equivalente a la diferencia calculada, que se concatena al tensor original de la muestra para aplicar *padding* si la secuencia actual es menor a la longitud máxima detectada en el *batch*.

----

## 7) <ins>ENTRENAMIENTO</ins>
El entrenamiento optimiza conjuntamente los parámetros del *Encoder* y *Decoder*, a través del mismo *optimizer*, cada uno con un *learning rate* individual.

#### 7.1) <ins>ESTRATEGIA DE SECUENCIA Y TEACHER FORCING:</ins>
- **Embedding visual (t=0):**
    - Al inicio del bucle iterativo, el tensor de la imagen se concatena como el primer elemento de la secuencia de entrada (dim=1), estableciendo el contexto visual como primer paso (*step*) en la *GRU Layer*.

- **Implementación de *teacher forcing*:**
    - Durante la fase de entrenamiento, el modelo no utiliza sus propias predicciones anteriores como entrada para el siguiente paso. En cambio, se utiliza la estrategia *teacher forcing* para utilizar directamente los *embeddings* del *caption* real en cada paso del bucle para estabilizar el aprendizaje y convergencia.
 
- **Criterio de terminación:**
    - El bucle se ejecuta durante '*seq_length-1*' pasos. Esto evita que el *token* *'\<end_seq>'* se procese como *input* para generar un paso posterior, permitiendo que el *decoder* aprenda a predecir cuándo finalizar la generación del *caption*.


#### 7.2) <ins>FUNCIÓN DE PÉRDIDA Y PADDING:</ins>
- **Cálculo de logits:**
    - En cada iteración, el *hidden state* de la *GRU Layer* pasa por la capa de clasificación para generar *logits* (que representa la distribución de probabilidad no normalizada) sobre el espacio total del vocabulario (2463 clases/palabras).<br>
      Los tensores resultantes se concatenan y permutan a las siguientes dimensiones:<br>
      <div align="center">
        
      ```[batch_size,vocab_size,sequence_length]```
      
      </div>
      
      para evaluarse de forma multidimensional utilizando la *Cross Entropy Loss Function* de PyTorch.

- **Ignore index - Loss function:**
    - Debido a que las secuencias en un mismo *batch* contienen diferentes longitudes, se utiliza la técnica *padding* para permitir el entrenamiento paralelo.<br> Con la finalidad de evitar que el modelo calcule gradientes sobre estos valores 'vacíos', se emplea el parámetro "*ignore_index=0*" en la *Cross Entropy loss function*, haciendo referencia al *padding token* para que no afecte el entrenamiento.

----

## 8) <ins>MODELO ENTRENADO INICIAL</ins>
- Inicialmente, se entrenó el modelo con los siguientes hiperparámetros:
  
| **Hiperparámetros** | **Valor** |
|:-------------------:|:---------:|
|    embed_img_size   |    256    |
|  embed_caption_size |    256    |
|     hidden_size     |    128    |
|      num_layers     |     1     |
|    l_r_img_model    |    1e-5   |
|    l_r_dec_model    |    5e-4   |
|    épocas máximas   |    100    |
|    Early Stopping   |     Sí    |

- Encoder: *Inception V3* --> *Linear Layer*: 2048 dims --> 256 dims
- Decoder: *Embedding Layer* --> *GRU Layer* --> *Classification Layer*: 256 dims --> 128 dims --> 2463 dims
