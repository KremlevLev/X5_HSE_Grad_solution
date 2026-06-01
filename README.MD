# README.md

```markdown
# X5 Retail Forecasting — HSE Hackathon Solution

> **Финальный скор: 90.45 / 100**  
> Хакатон: X5 Retail Group × НИУ ВШЭ  
> Задача: прогноз РТО (выручки) 18,657 магазинов на март 2025

---

## 🏆 Результаты

| Решение | LB Score |
|---------|----------|
| LightGBM baseline | 89.74 |
| LightGBM + MLP blend | 90.34 |
| + Ridge Regression | 90.44 |
| **Финальный ансамбль** | **90.45** |

---

## 📋 Задача

**Тип:** Panel Time Series Forecasting  
**Данные:** 18,657 магазинов × 26 месяцев (янв 2023 — фев 2025)  
**Таргет:** РТО (розничный товарооборот, рубли)  
**Горизонт:** март 2025  
**Метрика:** `score = 100 × ((100 - MAPE) / 100)²`

---

---

## 🔧 Компоненты решения

### 1. Feature Engineering (71 признак)

```python
# Лаги
lag_1, lag_2, lag_3, lag_6, lag_12, lag_13, lag_14, lag_24

# Скользящие статистики (leakage-free, shift=1)
roll_mean_3, roll_mean_6, roll_mean_12
roll_std_3,  roll_std_6,  roll_std_12

# EMA и EWM
ema_3, ema_6, ema_12
ewm_alpha=[0.95, 0.90, 0.70, 0.50]

# Weighted lag (M5 стиль)
weighted_lag = 0.5*lag1 + 0.3*lag2 + 0.15*lag3 + 0.05*lag6

# YOY паттерны
yoy_growth, yoy_diff, mar_feb_ratio_prev

# Momentum
mom_1m, mom_3m, mom_12m

# Нормализация к магазину
lag1_vs_median, lag1_vs_mean, lag1_vs_cluster, lag1_vs_region

# Сезонность
month_sin, month_cos, is_march, is_q1, is_dec

# Агрегаты (leakage-free, сдвиг +1 месяц)
cluster_median_prev, cluster_mean_prev
region_median_prev,  region_mean_prev
```

### 2. Кластеризация магазинов

Двухуровневая иерархическая кластеризация:

```
Уровень 1: квартили по log(mean_RTO) → 4 группы
           (малые / средние / крупные / очень крупные)

Уровень 2: Ward linkage по профилю:
           [log_mean, cv, zero_ratio, trend_6m, mar_vs_feb]
           → 10 итоговых кластеров
```

### 3. LightGBM (основная модель)

Параметры найдены через **Optuna (100 trials)**:

```python
{
    'objective':         'regression_l1',
    'num_leaves':        170,
    'max_depth':         13,
    'learning_rate':     0.01368,
    'feature_fraction':  0.4134,
    'bagging_fraction':  0.9771,
    'reg_alpha':         5.054,
    'min_gain_to_split': 0.563,
    'max_bin':           398,
    'path_smooth':       0.851,
}
# best_iteration = 1618
# Локально: MAPE 4.54% = 91.13 балла
```

**Топ признаков:**
```
#1  ewm_alpha_0.90    — краткосрочная память
#2  lag_1             — предыдущий месяц
#3  ewm_alpha_0.70    — среднесрочная EWM
#4  log_mean_rto      — размер магазина
#5  month             — сезонность
```

### 4. MLP

Полносвязная нейросеть на тех же признаках.  
Соло результат: **89.00 на LB**.

### 5. Ridge Regression

Простая линейная модель как регуляризатор ансамбля:

```python
Ridge(alpha=1.0)
# Признаки: лаги + EMA + сезонность
# Обучение на log1p(target)
# Роль: снижает дисперсию ансамбля
```

### 6. N-BEATS

Нейросеть специально для временных рядов (архитектура из M5):

```python
# 4 блока, lookback=12 месяцев, hidden=256
# Нормализация: деление на среднее магазина
# Роль: улавливает паттерны ряда без табличных признаков
```

---

## 🎯 Финальный ансамбль

```python
# Пропорции (применяем apply_coef ОДИН РАЗ в конце)
blend = 0.44 * lgb_raw + 0.51 * mlp_raw + 0.05 * ridge_raw

# Постобработка — сезонная коррекция март/февраль
final = np.where(blend < 70_000_000,
                 blend * 1.044,   # малые магазины (+4.4%)
                 blend * 1.038)   # крупные магазины (+3.8%)
final *= 0.992  # коррекция bias (MAPE штрафует перепрогноз)
```

**Почему блендинг работает:**
```
LightGBM  — табличные паттерны, лаги, кластеры
MLP       — нелинейные комбинации признаков
Ridge     — линейный сигнал, снижает дисперсию

Корреляция ошибок LGB↔MLP < 1.0
→ усреднение снижает общую ошибку
```

---

## ❌ Что не сработало

| Метод | Результат | Причина |
|-------|-----------|---------|
| Mean Encoding (M5) | 88.38 | Leakage в expanding window |
| Seasonal coef из истории | 85.05 | март/февраль ratio нестабилен |
| Auto-подбор коэф (Nelder-Mead) | 84.54 | Февраль ≠ март сезонно |
| Стекинг 120 моделей | хуже | Усредняет хорошее с плохим |
| Loss Multiplier Ensemble | не помог | LightGBM 4.x не поддерживает fobj |
| Seed=102 вместо 42 | 88.75 | Февральская валидация врёт |

---

## 💡 Ключевые инсайты

**1. Февральская валидация нерелевантна для марта**
```
Лучшая локальная модель (91.13б) → хуже на LB
Худшая локальная модель (90.97б) → лучше на LB
Причина: февраль ≠ март сезонно
```

**2. Блендинг разных архитектур критичен**
```
LGB соло:      89.74
LGB + MLP:     90.34  (+0.60)
+ Ridge:       90.44  (+0.10)
Финал:         90.45  (+0.01)
```

**3. Коэффициенты важнее чем кажется**
```
Модель обучена на всех месяцах
Март = систематический скачок вверх
1.044/1.038/0.992 компенсируют этот bias
```

**4. apply_coef() должен применяться ОДИН РАЗ**
```
Неправильно: apply_coef(lgb) × w1 + apply_coef(mlp) × w2
Правильно:   apply_coef(lgb × w1 + mlp × w2)
```

---

## 📦 Зависимости

```
python>=3.10
lightgbm==4.6.0
optuna==3.x
torch==2.x
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
scipy>=1.10
```

---

## 👥 Команда

| Участник |
|----------|
| https://github.com/KremlevLev |
| https://github.com/Vladislav585 |

---

## 📚 Референсы

- [M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [Matthias 2nd place solution (N-BEATS + LightGBM)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/163684)
- [Ekrem Bayar — Store Sales Guide](https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide)
- [N-BEATS: Neural basis expansion analysis](https://arxiv.org/abs/1905.10437)
```
