## Перенос оптимизаций (AI-Toolkit → ai-toolkit-changed-main)

Этот файл — **чистый список правок**, которые были перенесены из вашей ветки оптимизации в репозиторий коллеги `ai-toolkit-changed-main`, при этом сохранены его логические изменения (в т.ч. запрет latent caching для multi-frame по умолчанию).

### Изменённые файлы (с привязкой к строкам)

#### 1) `toolkit/dataloader_mixins.py`
- **Импорт**: добавлен `torch.nn.functional as F` для ресайза control-тензоров при кэшировании.
- **Latent cache расширен**: добавлен `_cached_control_latent`, чтение/очистка и запись `control_latent` в `.safetensors`.
- **Инвалидация кэша**: добавлены поля в `get_latent_info_dict()` для учёта `control_path` и параметров control-изображений.
- **Multi-frame защита**: запрет caching latents для `num_frames > 1` теперь управляется флагом `datasets.forbid_cache_latents_multi_frame` (по умолчанию True, т.е. поведение коллеги сохраняется).

Ключевые места: строки примерно **L14–L20**, **L1630–L1725**, **L1731–L1845**.

#### 2) `toolkit/data_transfer_object/data_loader.py`
- **Добавлено поле** `control_latents` в `DataLoaderBatchDTO`.
- **Сборка из кэша**: если latents закэшированы и есть `_cached_control_latent`, батч формирует `control_latents` (с заполнением нулями для отсутствующих).

Ключевые места: строки примерно **L138–L181**.

#### 3) `extensions_built_in/diffusion_models/flux_kontext/flux_kontext.py`
- **Fast-path без VAE**: если в батче есть `control_latents`, используется конкатенация без VAE-энкода control-картинки.
- **Кэш `img_ids/txt_ids`**: добавлены `_img_ids_cache/_txt_ids_cache` и методы `_get_cached_img_ids/_get_cached_txt_ids` для снижения аллокаций и фрагментации VRAM.
- **Условные `.to()`**: добавлен `_ensure_device_dtype()` — перенос/каст делается только при необходимости.
- **Layer offloading**: добавлены `MemoryManager.attach(...)` для transformer/T5/CLIP по флагам `model.layer_offloading_*_percent`.

Ключевые места: строки примерно **L20–L120**, **L314–L425**, **L455–L505**.

#### 4) `toolkit/config_modules.py`
- `NetworkConfig`: добавлен `network.lora_weight_dtype` (по умолчанию `float32`).
- `DatasetConfig`: добавлен `datasets.forbid_cache_latents_multi_frame` (по умолчанию `true`).

Ключевые места: строки примерно **L165–L220** и **L950–L960**.

#### 5) `jobs/process/BaseSDTrainProcess.py`
- LoRA/Network переводится в dtype из `network.lora_weight_dtype` вместо жёсткого `float32`.

Ключевые места: строки примерно **L1745–L1755**.

#### 6) `config/examples/train_lora_flux_kontext_24gb.yaml`
- Добавлены подсказки для:
  - `network.lora_weight_dtype`
  - `model.layer_offloading*`

---

### Что НЕ переносили (чтобы сохранить логику коллеги)
- Любые изменения, не относящиеся к VRAM/скорости/памяти (UI/функциональные логические фичи и т.п.), кроме тех, что требовались для включения оптимизационных флагов/параметров.

