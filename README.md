
# Proyecto_MLOps_Steam

![Proyecto PLataforma Steam (2)](https://github.com/MariJo91/Proyecto_MLOps_Steam/assets/117179993/96e23554-d447-4c92-ac56-6bf28b0a7d3e)


### Sistema de Recomendacion para Usuarios de la Plataforma Multinacional de Videojuegos Steam


Este proyecto presenta una API implementada en un servicio en la nube y la aplicación de dos modelos de aprendizaje automático. Simula el rol de un ingeniero MLOps, que combina las habilidades de un ingeniero de datos y un científico de datos, para la plataforma multinacional de videojuegos Steam.

El proyecto involucró realizar análisis de sentimiento sobre los comentarios de los usuarios de los juegos y recomendar juegos en función de un título o de las preferencias de un usuario en particular. Se desarrolló un caso de negocio del mundo real utilizando conjuntos de datos públicos de la industria de los videojuegos.

#### Objetivo

El objetivo principal era crear el primer modelo de aprendizaje automático de extremo a extremo a través de un enfoque que abarca tareas de ingeniería de datos (ETL, EDA, API) hasta la implementación del aprendizaje automático. Se buscaba lograr un desarrollo rápido y tener un Producto Mínimo Viable (MVP).

El ciclo de vida del proyecto abarca todo el proceso de aprendizaje automático, desde la recopilación y el procesamiento de datos hasta el entrenamiento y mantenimiento del modelo a medida que llegan nuevos datos.

Puntos Clave:

- Rol del Ingeniero MLOps: El proyecto destaca las habilidades y responsabilidades de un ingeniero MLOps, que fusiona la ingeniería de datos y la ciencia de datos.

- Análisis de Sentimiento y Recomendaciones de Juegos: El proyecto demuestra aplicaciones prácticas del aprendizaje automático en la industria de los videojuegos, incluido el análisis de sentimiento y las recomendaciones de juegos personalizadas.

- Caso de Negocio del Mundo Real: El enfoque en un caso de negocio real de la industria de los videojuegos agrega relevancia práctica y muestra el impacto potencial del aprendizaje automático en este dominio.

- Modelo de Aprendizaje Automático de Extremo a Extremo: El proyecto enfatiza la creación de un pipeline completo de aprendizaje automático, desde la ingeniería de datos hasta la implementación y el mantenimiento del modelo.

- Enfoque de Producto Mínimo Viable (MVP): El proyecto prioriza el desarrollo rápido y la entrega de un Producto Mínimo Viable, asegurando un enfoque práctico y eficiente.

- Ciclo de Vida del Aprendizaje Automático: El proyecto abarca todo el ciclo de vida del aprendizaje automático, enfatizando la importancia del entrenamiento y mantenimiento continuo del modelo.

### Etapas del Proyecto

![Gráfico de Etapas o Pasos de un Proyecto Elementos Relacionados Multicolor  (1)](https://github.com/MariJo91/Proyecto_MLOps_Steam/assets/117179993/61239200-624f-436b-8fbd-2231f86093db)

#### 1. - Inmersión en los Datos: Exploratory Data Analysis (EDA)

El proyecto se inició con un análisis exploratorio profundo de tres archivos JSON alojados en la carpeta https://drive.google.com/drive/folders/1yhjOWPTD0bZb3RP6qgIq9vlo0vYQnqzA de un repositorio público en Google Drive. Esta fase de EDA tenía como objetivo obtener información valiosa de los conjuntos de datos y guiar los pasos posteriores de ETL (Extracción, Transformación y Carga). El objetivo final era optimizar tanto el rendimiento de la API como el entrenamiento del modelo.

Herramientas y técnicas clave:

Pandas: Empleado para la manipulación y limpieza de datos.

Matplotlib y Seaborn: Utilizados para la visualización y exploración de datos.

##### Proceso de EDA:

- Examen inicial de datos: Inspección inicial de los conjuntos de datos para comprender su estructura, contenido y posibles problemas.

- Limpieza de datos: Abordar valores faltantes, valores atípicos e inconsistencias para garantizar la calidad de los datos.

- Transformación de datos: Transformar los datos en un formato adecuado para su análisis y modelado.

- Análisis exploratorio: Visualizar y analizar datos para descubrir patrones, tendencias y relaciones entre variables.

###### Resultados:

- Comprensión mejorada de los conjuntos de datos: Una comprensión más profunda de las características, la distribución y los desafíos potenciales de los conjuntos de datos.

- Decisiones de ETL informadas: Decisiones bien informadas para el proceso de ETL posterior, asegurando que los datos estén preparados para su análisis y modelado.

- Rendimiento optimizado de API y modelo: Una base sólida para construir una API de alto rendimiento y un modelo de aprendizaje automático eficaz.

#### 2. - Ingeniería de Datos (ETL y API)

###### 2.1 Transformación de Datos: Preparando los Datos para un Rendimiento Óptimo

Realicé transformaciones esenciales en los conjuntos de datos para garantizar un formato adecuado, optimizando tanto el rendimiento de la API como el entrenamiento del modelo. Cada conjunto de datos pasó por un proceso minucioso para optimizar su estructura y mejorar su utilidad.

El conjunto de datos original lo podemos encontrar en el siguiente enlace. https://drive.google.com/drive/folders/1yhjOWPTD0bZb3RP6qgIq9vlo0vYQnqzA

australian_user_reviews.json: Este archivo contiene reseñas de juegos escritas específicamente por usuarios australianos. Para extraer información valiosa, procesé cuidadosamente las reseñas, generando un archivo limpio y estructurado llamado user_reviews_limpio.parquet. 

output_steam_games.json: Este archivo proporciona información detallada sobre los juegos disponibles en la plataforma Steam. Incluye detalles como géneros, etiquetas, especificaciones, desarrolladores, año de lanzamiento, precio y otros atributos relevantes de cada juego. El notebook de ETLprocess_steam_game profundiza en los pasos de procesamiento necesarios para preparar estos datos para su análisis.

australian_users_items.json: Este archivo contiene información sobre elementos relacionados con usuarios australianos. Para mejorar su usabilidad, apliqué técnicas de Extracción, Transformación y Carga (ETL), como se describe en el notebook ETLprocess_users_items. 

Al transformar meticulosamente estos conjuntos de datos, garantizamos su idoneidad tanto para el rendimiento de la API como para el entrenamiento del modelo. Las estructuras de datos resultantes proporcionaron una base sólida para construir un sistema de recomendación efectivo.

###### 2.2 Ingeniería de Características: Revelando el Sentimiento de las Reseñas de Usuarios

Para enriquecer el conjunto de datos con información valiosa, incorporé análisis de sentimiento a las reseñas de los usuarios, creando una nueva columna llamada sentiment_analysis. Esto involucró el uso de la biblioteca Natural Language Toolkit (NLTK) y su analizador de sentimientos Vader. Vader asigna una puntuación compuesta a cada reseña, lo que nos permite clasificar su polaridad como negativa (valor '0'), neutral (valor '1') o positiva (valor '2'). Para las reseñas faltantes, asigné un valor de '1' para garantizar la coherencia.

La implementación detallada de este paso de ingeniería de características se puede encontrar en el notebook ETLprocess_user_reviews. Para profundizar en el proceso de análisis de sentimiento, consulta el notebook EDAprocess.

Esta incorporación de análisis de sentimiento transformó las reseñas de los usuarios en una fuente de información más rica, permitiéndonos capturar las emociones y actitudes subyacentes expresadas por los usuarios, lo que resultó invaluable para construir un sistema de recomendación efectivo.


###### 2.3 Desarrollando la API: Liberando el Poder de la Información sobre Videojuegos

Para brindar a los usuarios información relacionada con los videojuegos, implementé una API utilizando FastAPI y la implementé en Render. Esta API funciona como una puerta de entrada a un tesoro de información, ofreciendo cinco puntos finales que atienden diversas necesidades de los usuarios.

Punto Final 1: Conocimiento sobre Desarrolladores (developer)

Adéntrate en el mundo de los desarrolladores de videojuegos con este punto final. Revela la cantidad de artículos y el porcentaje de contenido gratuito lanzado por cada desarrollador cada año.

Punto Final 2: Datos Centrados en el Usuario (userdata)

Descubre los hábitos de juego de un usuario con este punto final. Revela la cantidad de dinero gastado, el porcentaje de recomendaciones basadas en las reseñas de los usuarios y la cantidad total de artículos jugados.

Punto Final 3: Maestros del Género (userforgenre)

Descubre el usuario que ha acumulado más horas jugadas en un género específico. Este punto final también proporciona un desglose del tiempo de juego por año de lanzamiento.

Punto Final 4: Desarrolladores Más Recomendados (best_developer_year)

Descubre los 3 principales desarrolladores con los juegos más recomendados por los usuarios para un año determinado.

Punto Final 5: Análisis de Reseñas por Desarrollador (developer_reviews_analysis)

Obtén información sobre el sentimiento del usuario hacia un desarrollador en particular. Este punto final devuelve un diccionario con el nombre del desarrollador como clave y una lista que contiene el número total de registros de reseñas de usuarios que se encuentran categorizados con un análisis de sentimiento como positivo o negativo.

Aprovecha el Poder de la Información sobre Videojuegos

Para explorar la funcionalidad completa de la API y embarcarte en un viaje de descubrimiento de juegos, visita la URL de la https://obertomaria91.wixsite.com/myweb. Allí encontrarás los diversos puntos finales (endoints) listos para satisfacer tus necesidades


#### 3. - Desvelando el Modelo de Recomendación Basado en Ítems

Para impulsar el motor de recomendaciones, me adentré en el reino del filtrado colaborativo basado en ítems, una técnica que identifica ítems similares en función de sus características compartidas.

###### 3.1 Adoptando la Similitud del Coseno para las Recomendaciones de Juegos

En el núcleo del Sistema de Recomendación Basado en Ítems se encuentra el concepto de similitud del coseno. Esta métrica mide el ángulo entre dos vectores, con valores más cercanos a 1 que indican una mayor similitud. Al analizar la similitud del coseno entre juegos, pude identificar elementos que comparten rasgos y preferencias de usuario similares.

#### 4. Implementación de MLOps: Dando Vida al Modelo

Una vez que el modelo de recomendación estuvo listo, llegó el momento de integrarlo a la API para que los usuarios pudieran aprovechar su poder.

4.1 Implementación del Modelo de Recomendación

Implementé sin problemas el modelo de recomendación como parte integral de la API, permitiendo a los usuarios aprovechar sus capacidades a través de los puntos finales de la API. Puedes acceder a la API en la siguiente URL: https://mlps-pi-steam-soyhenry.onrender.com

4.2 Adopción de Render para una Implementación Fluida

Para optimizar el proceso de implementación, opté por Render, una plataforma en la nube unificada diseñada específicamente para crear y ejecutar aplicaciones y sitios web. Las capacidades de implementación automatizada de Render, junto con su integración directa con GitHub, demostraron ser invaluables.

4.3 Aprovechando GitHub para una Integración Fluida

Dadas las limitaciones de almacenamiento del servicio gratuito de Render, creé un repositorio dedicado específicamente para fines de implementación. Este repositorio se puede encontrar en el siguiente enlace: https://github.com/MariJo91/Proyecto_MLOps_Steam.git

#### 5. Video Explicativo: Desmitificando la Magia

Para mejorar la comprensión y brindar una descripción general completa, elaboré un video explicativo. Este video profundiza en las complejidades de la API, mostrando sus funcionalidades, demostrando consultas de ejemplo y ofreciendo una explicación concisa de los principios subyacentes del aprendizaje automático. Puedes ver el video en YouTube en el siguiente enlace: 


