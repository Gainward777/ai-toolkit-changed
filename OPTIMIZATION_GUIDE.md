## Руководство по оптимизациям обучения LoRA для Flux-Kontext (low-VRAM)

Ниже описаны оптимизации, перенесённые в `ai-toolkit-changed-main`, и как ими пользоваться.

### 1) Кэш `control_latent` (самая важная оптимизация для Flux-Kontext)

**Проблема**: Flux-Kontext использует paired control-изображение и в базовой реализации делает VAE-энкод control-изображения на каждом train-step → это дорого по VRAM и времени.

**Решение**: при включённом `cache_latents_to_disk` мы теперь **однократно** считаем и сохраняем `control_latent` рядом с обычным `latent` в `_latent_cache/*.safetensors`. На train-step модель берёт `batch.control_latents` и не трогает VAE для control.

**Как включить** (в датасете, Flux-Kontext):

```yaml
datasets:
  - folder_path: "..."
    control_path: "..."
    cache_latents_to_disk: true
```

**Как это работает**:
- Во время этапа “Caching latents” создаются поля:
  - `latent`
  - `control_latent` (если есть `control_path`)
- В train-loop `DataLoaderBatchDTO` формирует `control_latents`.
- `FluxKontextModel.condition_noisy_latents()` использует `control_latents` (fast-path) и не делает VAE-encode.

### 2) Запрет caching latents для multi-frame датасетов (по умолчанию включён)

**Почему**: для видео/мультифреймных датасетов формат латентов/кадров и аугментации часто несовместимы с простым кэшом “один latent на файл”.

**Поведение по умолчанию**: если `num_frames > 1` и включён `cache_latents` или `cache_latents_to_disk`, то будет ошибка.

**Флаг** (в датасете):

```yaml
datasets:
  - folder_path: "..."
    num_frames: 16
    cache_latents_to_disk: true
    forbid_cache_latents_multi_frame: true   # default
```

Если вы точно знаете, что делаете (и ваш пайплайн это поддерживает), можно временно отключить:

```yaml
    forbid_cache_latents_multi_frame: false
```

### 3) Кэш `img_ids/txt_ids` для Flux-Kontext

**Что даёт**: уменьшает количество крупных аллокаций на каждом шаге, снижает фрагментацию VRAM, помогает избежать OOM на небольших GPU.

**Настроек не требует** — работает автоматически.

### 4) Уменьшение лишних `.to(device, dtype)` в hot-path

**Что даёт**: меньше временных копий тензоров в `get_noise_prediction()`, меньше пиков VRAM.

**Настроек не требует** — работает автоматически.

### 5) `layer_offloading` для Flux-Kontext (сильно снижает VRAM, но медленнее)

**Что делает**: “оборачивает” Linear/Conv слои MemoryManager’ом и по ходу forward/backward выгружает часть параметров на CPU.

**Как включить**:

```yaml
model:
  layer_offloading: true
  layer_offloading_transformer_percent: 1.0
  layer_offloading_text_encoder_percent: 1.0
```

### 6) `network.lora_weight_dtype` (LoRA веса не обязательно FP32)

**Что даёт**: LoRA веса в bf16/fp16 могут заметно снизить VRAM, особенно на больших rank/многих слоях.  
**Риск**: в некоторых случаях может ухудшить стабильность/качество, поэтому дефолт остаётся `float32`.

**Как включить**:

```yaml
network:
  type: lora
  linear: 16
  linear_alpha: 16
  lora_weight_dtype: bf16   # или float16, либо оставить float32
```

### Рекомендуемый “low VRAM” пресет для Flux-Kontext

```yaml
datasets:
  - folder_path: "..."
    control_path: "..."
    cache_latents_to_disk: true

train:
  batch_size: 1
  gradient_checkpointing: true
  optimizer: adamw8bit

network:
  type: lora
  linear: 16
  linear_alpha: 16
  # lora_weight_dtype: bf16

model:
  quantize: true
  # layer_offloading: true
  # layer_offloading_transformer_percent: 1.0
  # layer_offloading_text_encoder_percent: 1.0
```

