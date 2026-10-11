# Cuotas de fútbol en tiempo real con un Axial Transformer

Trabajo Fin de Máster del Máster en Análisis de Datos Masivos (Big Data), Universidad Europea de Madrid.
Autor: Mateo Martínez Sanz.

El proyecto predice lo que queda de un partido de fútbol después de cada evento importante (tiros, faltas, córners, cambios...) y lo convierte en cuotas de apuestas que se actualizan en directo. El modelo sigue el Axial Transformer de Horton y Lucey (2025), y el sistema en tiempo real está desplegado en AWS.

- **Datos:** eventos de [StatsBomb Open Data](https://github.com/statsbomb/open-data) (temporada 2015/16 de las cinco grandes ligas, 1.551 partidos) y resultados y cuotas de [football-data.co.uk](https://www.football-data.co.uk).
- **Modelo:** rejilla de jugadores convocados, equipos y partido por eventos clave; 4 capas de atención axial, 4 cabezas, D = 128.
- **Salidas:** probabilidades del 1X2 y tasas de Poisson de lo que le queda a cada jugador y equipo en 12 acciones (goles, tiros, córners, tarjetas...).
- **Cuotas:** 1X2, próximo gol y Más/Menos de goles, tiros, córners y tarjetas, con calibración por temperatura, sesgo favorito-outsider y margen calibrados con casas reales.
- **AWS:** Kinesis, Lambda, DynamoDB, SageMaker (endpoint serverless), SNS/SQS y API Gateway, todo con CloudFormation.

## Estructura

| Fichero | Para qué sirve |
|---|---|
| `procesado.py` | Convierte los eventos de StatsBomb en las entradas del modelo. Solo usa numpy, así que también lo usan las Lambdas. |
| `data_pipeline.py` | Carga los partidos, monta la convocatoria, calcula el nivel previo y hace la partición train/val/test. |
| `historial.py` | Elo y datos de la temporada anterior con los resultados de football-data. |
| `axial_transformer.py` | El modelo. |
| `train.py` | Entrenamiento, calibración de la temperatura (`--calibrar`) y demo de un partido (`--demo`). |
| `odds_generator.py` | De predicciones a cuotas. |
| `evaluate.py` | Métricas del paper y comparación con las casas de apuestas. |
| `calibrar_cuotas.py` | Ajuste del margen y de alfa con las cuotas de seis casas. |
| `comparar_v2_v3.py` | Comparación del modelo con y sin la convocatoria completa. |
| `figuras_memoria.py`, `figura_arquitectura.py` | Figuras de la memoria. |
| `entrenar_v3.cmd` | Todo seguido: entrenar, calibrar, evaluar, demo y figuras (Windows). |
| `checkpoints/best_model.pt` | Modelo entrenado (versión final, con la convocatoria completa). |
| `results/` | Métricas, figuras y la prueba de extremo a extremo en AWS. |
| `infra/` | Despliegue en AWS (ver más abajo). |

## Instalación

Python 3.12 o superior.

```bash
pip install -r requirements.txt
```

Los datos de StatsBomb no están en el repositorio (unos 7 GB). Hay que descargarlos y decir dónde están:

```bash
git clone https://github.com/statsbomb/open-data.git
set STATSBOMB_DATA=C:\ruta\a\open-data        # Windows
export STATSBOMB_DATA=/ruta/a/open-data       # Linux / macOS
```

Los ficheros de football-data se descargan solos la primera vez y se guardan en `cache/`.

## Uso

```bash
python train.py                  # entrenamiento (unas 9 horas en CPU)
python train.py --calibrar       # temperatura del 1X2 con validación
python evaluate.py               # evaluación en test frente a Pinnacle
python train.py --demo           # cuotas de un partido de test
```

Si solo se quiere evaluar, se puede usar directamente `checkpoints/best_model.pt` y saltarse el entrenamiento.

## Despliegue en AWS

Hace falta la AWS CLI configurada con un usuario de IAM (`aws configure`). Las claves no se guardan en el código.

```bash
# Presupuesto con avisos por correo (una sola vez)
aws cloudformation deploy --template-file infra/presupuesto.yaml --stack-name tfm-presupuesto --parameter-overrides EmailAvisos=tu@correo.com

python infra/desplegar.py                          # despliega todo (unos 10 minutos)
python infra/simulador.py --match_id 3754206 --velocidad 60
python infra/exportar_prueba.py --match_id 3754206 # guarda las cuotas publicadas en results/aws
python infra/comprobar_nube.py                     # compara la nube con el modelo local
python infra/borrar.py                             # lo borra todo menos el presupuesto
```

`desplegar.py` imprime la dirección de la web, donde se ven las cuotas cambiando mientras corre el simulador.

| Pieza en AWS | Código |
|---|---|
| Lambda `tfm-procesador` | `infra/lambdas/procesador/handler.py` + `procesado.py` |
| Endpoint de SageMaker | `infra/modelo/inference.py` + `axial_transformer.py` + `procesado.py` + pesos |
| Lambda `tfm-cuotas` | `infra/lambdas/cuotas/handler.py` + `odds_generator.py` + `procesado.py` |
| Lambda `tfm-plataforma` | `infra/lambdas/plataforma/handler.py` |
| Lambda `tfm-api` y web | `infra/lambdas/api/handler.py` + `index.html` |

## Resultados principales

En los 232 partidos de test (abril y mayo de 2016):

| 1X2 antes del partido | Log-loss | Brier | Acierto |
|---|---|---|---|
| Modelo | 0,983 | 0,580 | 52,2 % |
| Pinnacle (cierre) | 0,921 | 0,544 | 55,6 % |

El modelo mejora con claridad a las referencias sencillas, pero no llega al mercado antes del partido. En AWS, cada actualización de cuotas tarda una mediana de 135 ms en el modelo.

## Referencia

Horton, M. y Lucey, P. (2025). *Large-Scale In-Game Outcome Forecasting for Match, Team and Players in Football using an Axial Transformer Neural Network.* arXiv:2511.18730.
