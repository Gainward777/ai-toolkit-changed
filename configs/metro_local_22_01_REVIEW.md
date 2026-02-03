# Обзор конфига: metro_local_22_01

## ⚠️ КРИТИЧЕСКАЯ ОПЕЧАТКА!

```yaml
local_hard_k_franc: 0.3   # ❌ ОПЕЧАТКА! 
local_hard_k_frac: 0.3    # ✅ Правильно (frac, не franc)
```

**Исправь это перед запуском!**

---

## Анализ настроек текстурного лосса

### ✅ Хорошие настройки

| Параметр | Значение | Комментарий |
|----------|----------|-------------|
| `texture_loss` | `"custom"` | ✅ Правильный режим |
| `loss_local_coef` | `1.0` | ✅ Основной лосс включён |
| `loss_Spect_coef` | `0.5` | ✅ Хорошо для плитки |
| `loss_AFC_coef` | `0.3` | ✅ Нормально |
| `loss_cross_patch_coef` | `0.3` | ✅ Для перспективы |
| `gate` | `0.6` | ✅ Чуть активнее стандартного (0.65) |
| `v_scale` | `0.2` | ✅ Стандартное значение |
| `min_beta` | `0.03` | ✅ Хорошо — сразу немного текстуры |
| `max_beta` | `0.5` | ✅ Стандарт |
| `local_patch_size` | `192` | ✅ Оптимально для 1024px |
| `local_stride` | `96` | ✅ Половина patch_size |
| `cross_patch_size` | `256` | ✅ Большие патчи для тайлов |
| `cross_patch_stride` | `128` | ✅ 50% overlap |

### ⚠️ Кастомные параметры (отличаются от дефолтов)

```yaml
# Спектральные параметры (изменены)
tau: 0.08              # default: 0.06 — чуть мягче soft-peak
band_sigma: 0.22       # default: 0.15 — шире полоса частот  
w_peak: 0.8            # default: 1.0 — слабее акцент на пике
w_band: 0.35           # default: 1.0 — слабее band matching

# ACF параметры (изменены)
afc_tau: 0.1           # default: 0.08
afc_w: 0.25            # default: 1.0 — СИЛЬНО ослаблен!

# Phase параметры (изменены)
pce_weight: 0.05       # default: 0.1 — ослаблен

# Local weights (стандартные)
local_w_spectral: 0.5  # default: 0.5 ✅
local_w_afc: 1.5       # default: 1.5 ✅
local_w_phase: 1.2     # default: 1.2 ✅
```

**Вопрос**: Откуда эти значения? Если это результат предыдущих экспериментов — ок. Если случайные — лучше начать с дефолтов.

### ❌ Не использованные/выключенные

```yaml
loss_Phase_coef: 0     # Выключен (уже есть в local)
loss_Log_coef: 0       # Выключен (обычно не нужен)
local_w_log: 0         # Выключен
```

Это нормально — они обычно не нужны.

---

## Общие настройки обучения

| Параметр | Значение | Комментарий |
|----------|----------|-------------|
| `steps` | `2500` | ✅ Достаточно для материала |
| `lr` | `0.0001` | ✅ Стандарт для LoRA |
| `batch_size` | `1` | ✅ Для low VRAM |
| `gradient_accumulation` | `4` | ✅ Эффективный batch = 4 |
| `dtype` | `bf16` | ✅ Для Flux |
| `optimizer` | `adamw8bit` | ✅ Экономит память |
| `ema_decay` | `0.999` | ✅ EMA включён |
| `resolution` | `1024` | ✅ |

### LoRA настройки

```yaml
linear: 32
linear_alpha: 32
conv: 16
conv_alpha: 16
```

Это достаточно высокий rank. Для материалов может быть избыточно, но не повредит.

---

## 📋 Итоговая оценка

### Исправить обязательно:
1. **`local_hard_k_franc` → `local_hard_k_frac`**

### Рекомендую проверить:
1. Кастомные `w_peak: 0.8`, `w_band: 0.35`, `afc_w: 0.25` — сильно ослаблены. Если не уверен откуда они — верни к дефолтам.

### Всё остальное выглядит хорошо! 👍

---

## Исправленный конфиг (ключевые параметры)

```yaml
train:
  texture_loss: "custom"
  
  # Основные коэффициенты (ок)
  loss_local_coef: 1
  loss_Spect_coef: 0.5
  loss_AFC_coef: 0.3
  loss_cross_patch_coef: 0.3
  loss_Phase_coef: 0
  loss_Log_coef: 0
  
  # Активация (ок)
  gate: 0.6
  v_scale: 0.2
  min_beta: 0.03
  max_beta: 0.5
  
  # Local loss params (ок)
  local_patch_size: 192
  local_stride: 96
  local_min_coverage: 0.5
  local_w_spectral: 0.5
  local_w_afc: 1.5
  local_w_phase: 1.2
  local_w_log: 0
  
  # ⚠️ ИСПРАВЛЕНО:
  local_hard_k_frac: 0.3    # Было: local_hard_k_franc
  local_hard_k_min: 4
  
  # CrossPatch (ок)
  cross_patch_size: 256
  cross_patch_stride: 128
  
  # Спектральные (кастомные — проверь нужны ли)
  tau: 0.08
  band_sigma: 0.22
  w_peak: 0.8           # default: 1.0
  w_band: 0.35          # default: 1.0
  min_bin: 3
  
  # ACF (кастомные)
  afc_tau: 0.1
  afc_min_rel: 0.04
  afc_max_rel: 0.55
  afc_w: 0.25           # default: 1.0 — очень слабый!
```

---

## Что смотреть в логах

```
[step 250] tex: beta=0.080 gate=0.45 base=0.0250 add=0.0080
```

**Хорошо если:**
- `gate > 0.2` (текстурный лосс активен)
- `add` = 10-30% от `base`
- `beta` растёт со временем

**Плохо если:**
- `add = 0.0000` постоянно
- `add >> base` (текстура доминирует)
