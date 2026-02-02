# Fine-tuning Classification - Español

Este notebook implementa un pipeline completo de fine-tuning para clasificación de textos en español, específicamente para detectar intención suicida en textos. Utiliza técnicas modernas como LoRA (Low-Rank Adaptation) y la plataforma Hugging Face.

## 📋 Descripción General

El notebook `Fine_tuning_Clasification.ipynb` realiza las siguientes tareas:

1. **Pre-requisitos**: Instalación de dependencias necesarias
2. **Configuración**: Setup de modelos, dispositivos (GPU/CPU) y variables de entorno
3. **Carga de Dataset**: Obtiene el dataset de Hugging Face
4. **Tokenización**: Preprocesamiento de textos
5. **Configuración LoRA**: Ajuste de parámetros para fine-tuning eficiente
6. **Entrenamiento**: Training del modelo con callbacks y monitoreo
7. **Evaluación**: Comparación de rendimiento base vs fine-tuned
8. **Guardado**: Exportación del modelo a Hugging Face Hub

## 🚀 Requisitos Previos

- Python 3.8+
- CUDA (opcional, para GPU)
- Cuenta en Hugging Face Hub
- Token de Hugging Face

## 📦 Dependencias Principales

```
transformers
accelerate
datasets
peft
torch
wandb
evaluate
rouge-score
pandas
numpy
scikit-learn
```

## 🛠️ Instalación

Las dependencias se instalan automáticamente en las primeras celdas del notebook:

```python
!pip install transformers
!pip install accelerate
!pip install wandb
!pip install evaluate==0.4.0 rouge-score==0.1.2
!pip install peft==0.13.2 datasets==3.0.1 wandb==0.13.1
```

## 📊 Dataset

El notebook utiliza el dataset `spanish-suicide-intent` disponible en Hugging Face Hub:
- **Estructura**: Train, Validation y Test splits
- **Tarea**: Clasificación binaria (intención suicida: Sí/No)
- **Idioma**: Español

```python
dataset = load_dataset("PrevenIA/spanish-suicide-intent")
```

## 🤖 Modelos Soportados

El notebook está configurado para trabajar con diferentes modelos base:
- `Qwen/Qwen2.5-0.5B` (por defecto)
- `Qwen/Qwen3-0.6B`
- `dccuchile/bert-base-spanish-wwm-cased`
- `UMUTeam/emotions-DistilBETO`
- `meta-llama/Llama-3.2-1B`

## ⚙️ Parámetros de Configuración

### Configuración LoRA
```python
LORA_RANK = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
TASK_TYPE = TaskType.SEQ_CLS
```

### Configuración de Entrenamiento
```python
num_train_epochs=2
learning_rate=0.001
per_device_train_batch_size=16
gradient_accumulation_steps=16
eval_steps=4
fp16=True  # Mixed precision training
```

## 🔄 Flujo del Notebook

### 1. Pre-requisitos (Sección 1)
- Instalación de dependencias
- Importación de librerías necesarias

### 2. Configuración (Sección 2)
- Variables globales y nombres de modelos
- Autenticación con Hugging Face Hub
- Carga del modelo y tokenizador

### 3. Dataset (Sección 3)
- Carga del dataset desde HF Hub
- Exploración de datos
- Pruebas de inferencia con el modelo base

### 4. Entrenamiento (Sección 4)
- Tokenización y preparación de datos
- Configuración de LoRA
- Training con monitoreo en W&B
- Callbacks para early stopping

### 5. Evaluación (Sección 5)
- Comparación modelo base vs PEFT
- Métricas: Accuracy, F1, Precision, Recall
- Análisis caso por caso

### 6. Guardado (Sección Final)
- Guardado local del modelo
- Carga de credenciales HF
- Upload a Hugging Face Hub

## 📈 Monitoreo y Logging

El notebook integra **Weights & Biases (W&B)** para monitoreo:

```python
PROJECT_NAME_WANDB = "FinetuningClasificacion"
wandb.init(project=PROJECT_NAME_WANDB, name=f"Run_{model_name}_{current_datetime}")
```

Se registran las siguientes métricas:
- Loss (entrenamiento y validación)
- Accuracy
- F1, Precision, Recall
- Learning rate
- Epoch y steps

## 💾 Guardado y Carga de Modelos

### Guardar en Hugging Face Hub
```python
save_model_locally(peft_model, REPO_LOCAL_NAME)
create_or_get_repo(REPO_HF_NAME, HF_TOKEN)
upload_lora_adapters(REPO_HF_NAME, REPO_LOCAL_NAME, HF_TOKEN)
```

### Cargar desde Hub
```python
from peft import PeftConfig, PeftModel
config = PeftConfig.from_pretrained(REPO_HF_NAME)
optimized_model = PeftModel.from_pretrained(base_model, REPO_HF_NAME)
```

## 🔑 Variables de Entorno

```bash
HUGGING_FACE_HUB_TOKEN=<your_token>
HF_HUB_ENABLE_HF_TRANSFER=1
```

## 📝 Funciones Principales

### `load_model_and_tokenizer(base_model_name)`
Carga el modelo y tokenizador preentrenados.

### `tokenize_preprocess(examples, tokenizer, max_length=64)`
Tokeniza los textos y prepara las etiquetas.

### `prepare_datasets(dataset, tokenizer, test_size=0.1)`
Divide el dataset en train, validation y test.

### `configure_lora(...)`
Configura los parámetros de LoRA.

### `compute_metrics(eval_pred)`
Calcula métricas de evaluación.

### `train_model(...)`
Ejecuta el training con callbacks.

## 🎯 Resultados Esperados

El modelo fine-tuned con LoRA típicamente:
- Reduce significativamente los parámetros entrenables
- Mejora el accuracy respecto al modelo base
- Mantiene o mejora F1, Precision y Recall
- Requiere menos memoria y tiempo de entrenamiento

## 🐛 Troubleshooting

### Error de CUDA/Memoria
```python
import torch
torch.cuda.empty_cache()
```

O ajustar en variables de entorno:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Token de HF no encontrado
Ejecutar la celda de login y proporcionar el token cuando se solicite.

### Problemas con Pad Token
El notebook maneja automáticamente la configuración del pad_token para diferentes tokenizadores.

## 📚 Referencias

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Weights & Biases](https://wandb.ai/)

## 👤 Autor

Judith

## 📄 Licencia

Este proyecto utiliza datasets y modelos disponibles bajo licencias de Hugging Face.

---

**Última actualización**: Febrero 2026
