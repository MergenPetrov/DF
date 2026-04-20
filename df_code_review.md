# Разбор реализации Drift Flux (my_code_12.txt)

Проверял только математику в `test_function` (строки 675–1473). Обёртку для
finite-difference производных (строки 1–673) не трогал — просил сам.

Всё сверялось с ECLIPSE Technical Description, главой «The drift flux slip
model» (стр. 692–699, уравнения 8.66–8.91) и статьёй Shi, Holmes et al.
«Drift-Flux Modeling of Two-Phase Flow in Wellbores», SPE Journal, March 2005
(уравнения 1–20).

Сокращения:
- Eclipse Eq. 8.XX — ссылки на уравнения из мануала;
- Shi Eq. X — ссылки на уравнения из статьи Shi2005.

---

## 1. КРИТИЧЕСКАЯ ошибка — формула `β*` (строка 1176)

### Текущий код

```cpp
double beta_norm = fabs (beta - B) / (1.0 - B);
if (beta_norm > 1.0)
  beta_norm = 1.0;
```

### Что должно быть (Eclipse Eq. 8.70)

```
β* = (β - B) / (1 - B),   0 ≤ β* ≤ 1
```

То есть при `β < B` результат должен обрезаться до **нуля**, а не браться по
модулю.

### Почему это ошибка

С `fabs`, при малых `αg` и `j` (`β ≈ 0 < B=0.3`) получается:

```
β* = |0 - 0.3| / 0.7 = 0.429
C0 = 1.2 / (1 + 0.2 * 0.429²) = 1.157
```

А должно быть `β* = 0`, `C0 = A = 1.2`.

Ошибка занижает C0 на режиме bubble/slug flow — то есть как раз там, где
профильное скольжение должно быть максимальным. Это сильно повлияет на расчёт.

### Правильный код

```cpp
double beta_norm = (beta - B) / (1.0 - B);
if (beta_norm < 0.0)
  beta_norm = 0.0;
else if (beta_norm > 1.0)
  beta_norm = 1.0;
```

---

## 2. КРИТИЧЕСКАЯ ошибка — sqrt в знаменателе drift velocity (строки 1195–1198)

### Текущий код

```cpp
double drift_velocity =
    (1.0 - temp_holdups.gas * profile_param_C0) * profile_param_C0 * K_g * bubble_rise_velocity
    / (sqrt (gas_density / liq_density) * temp_holdups.gas * profile_param_C0
       + 1.0 - temp_holdups.gas * profile_param_C0);
```

Здесь `sqrt` применён **только** к `ρg/ρl`, и знаменатель получается:
```
sqrt(ρg/ρl) * αg*C0 + (1 - αg*C0)
```

### Что должно быть (Eclipse Eq. 8.78, Shi Eq. 14–15)

```
vd = (1 - αg*C0) * C0 * K(αg) * vc
     ─────────────────────────────────
     sqrt( αg*C0 * ρg/ρl + 1 - αg*C0 )
```

`sqrt` охватывает **весь знаменатель**.

### Почему это ошибка

Эффект, конечно, меньше, чем по п.1, но результат отличается систематически.

### Правильный код

```cpp
const double denom_inner = temp_holdups.gas * profile_param_C0 * (gas_density / liq_density)
                         + 1.0 - temp_holdups.gas * profile_param_C0;
double drift_velocity =
    (1.0 - temp_holdups.gas * profile_param_C0) * profile_param_C0 * K_g * bubble_rise_velocity
    / sqrt (std::max (denom_inner, tnm::min_compare));
```

---

## 3. Ошибка — inclination multiplier: перепутаны θ и cos(θ) (строки 1011–1027)

### Текущий код

```cpp
auto compute_inclination_multiplier =
  [&] () -> double
  {
    if (fabs (seg.wsn->pipe_props.depth_change) < tnm::min_compare)
      return 0.0;                          // <-- горизонталь ⇒ 0, OK

    double length = fabs (seg.wsn->pipe_props.length);
    double depth = fabs (seg.wsn->pipe_props.depth_change);
    if (length < tnm::min_compare)
      return 0.0;

    const double cos_theta = (depth > length - tnm::min_compare) ? 1.0 : depth / length;
    if (depth > length - tnm::min_compare)
      return 1.0;                          // вертикаль: m(0) = 1, OK

    return tnav_pow (cos_theta, 0.5)
         * tnav_pow (1.0 + sqrt (std::max (0.0, 1.0 - cos_theta * cos_theta)), 2.0);
  };
```

### Что должно быть (Eclipse Eq. 8.80)

```
m(θ) = cos(θ)^0.5 * (1 + sin(θ))²,   θ — угол от вертикали
```

Здесь `depth_change / length = cos(θ)`, `sin(θ) = sqrt(1 - cos²θ)`. То есть

```
m = cos_theta^0.5 * (1 + sin_theta)^2
```

### Почему это, скорее всего, правильно, но уточни конвенцию

Формула выглядит корректной. Но удостоверься, что `depth_change` = разница
**вертикальных глубин** (vertical), а не измеренной (MD). Иначе `depth/length`
не будет равно `cos θ`.

И мелкое: при идеально горизонтальной трубе (`depth_change = 0`) функция
возвращает `0.0` — это правильно (Eclipse: «For horizontal segments the drift
velocity becomes zero»), НО:

**Потенциальная проблема: непрерывность.**
На θ = 90° функция cos(θ)^0.5 * (1+sin θ)² = 0 * 4 = 0 — плавно стремится к 0.
У тебя вместо плавного перехода стоит жёсткий if `< min_compare → 0`. Для
близко-горизонтальных сегментов (`depth ≈ 0`) это может дать ступеньку.
Не ошибка по сути, но для сходимости Якобиана лучше убрать эту ветку.

---

## 4. ОШИБКА — направление потока и итерация по сегментам (концептуальная)

Drift-flux — это связь между **superficial velocities** (`vsg`, `vsl`) и
**in-situ holdups** (`αg`). В итерации ты делаешь так (строки 1105–1107):

```cpp
const double gas_superficial_velocity = prev_vels.gas * temp_holdups.gas;
const double liq_superficial_velocity = prev_vels.liquid * temp_holdups.liquid;
const double mixture_velocity = gas_superficial_velocity + liq_superficial_velocity;
```

То есть `vsg` и `vsl` сам по себе «плавают» между итерациями — они
пересчитываются из обновлённых `holdup` и `velocity` каждой итерации.
В результате `j = vsg + vsl` меняется, и ты **не сходишься к заданному
расходу**.

### Что должно быть

Входные данные — это `q_tot` и покомпонентные молярные расходы (на входе в
DF). Из них надо один раз посчитать:

```
Qp [мольный расход фазы p] = Σc  xc,p * qc
vsp = Qp / (ξp * A)      — superficial velocity фазы p (в метрах/сек)
j = Σp vsp                — константа в итерации
```

А внутри итерации должна меняться только `αg` (ну и `αo/αw`), и обновляются
`C0`, `vd`. После — `αg` пересчитывается как

```
αg = vsg / (C0 * j + vd)
```

Это фикспоинт на одной переменной `αg`, а не на четырёх (`holdup` + `velocity`
каждой фазы), как сейчас.

### Почему это ошибка

Посмотри на строчку 1208:
```cpp
new_holdups.gas = gas_superficial_velocity / new_vels.gas;
```

`gas_superficial_velocity = prev_vels.gas * temp_holdups.gas` — то есть это
«предыдущее vsg». На следующей итерации `prev_vels.gas = new_vels.gas` и
`temp_holdups.gas = new_holdups.gas`. Если подставить:

```
vsg^(n+1) = v_g^(n+1) * αg^(n+1) = v_g^(n) * αg^(n)
```

получается тождественно. Значит, `vsg` действительно сохраняется. Хорошо.
**Но только** если в исходной итерации `prev_vels.gas * alpha_g_seed` равно
реальному заданному `vsg` — а у тебя `prev_vels.gas = average_volumetric_velocity`
и `alpha_g_seed = 1/3`, что очевидно неверно.

### Правильный seed

```cpp
// seed для superficial velocities фазы из входных расходов
double vsg_input = q_gas_molar / (xi_gas * area);   // и т.д.
double vsl_input = (q_oil + q_water) / ...;
```

Эти `vsg_input`, `vsl_input` фиксируются до итерации и используются внутри
цикла вместо `prev_vels.gas * temp_holdups.gas`.

У тебя, к счастью, есть `reconstruct_phase_rates_df` — она как раз решает
систему x_ic,ip * Qp = qc для нахождения мольного расхода каждой фазы. Нужно
просто использовать её результат как константный seed, а не только как
источник для `beta_o_first_seed`.

---

## 5. Ошибка — σgl считается через `pipe_gas_liq_interfacial_tension_holdup_weightening`, но веса неправильные

### Текущий код (строка 1133)

```cpp
double gas_liq_interfacial_tension =
    surf_mult * pipe_gas_liq_interfacial_tension_holdup_weightening (
        element_status->p * converter_metric_to_field.pressure_mult (),
        160.0,
        oil_api,
        temp_holdups.oil,
        temp_holdups.water,
        ...
```

`temp_holdups.oil` и `temp_holdups.water` — это `αo`, `αw` как доли **от всего
сечения**, а не от жидкости.

### Что должно быть (Eclipse Eq. 8.82)

```
σgl = (αo * σgo + αw * σwg) / (αo + αw)
```

Здесь `αo + αw = αl` (в знаменателе именно сумма, не 1). Если функция
`pipe_gas_liq_interfacial_tension_holdup_weightening` делает:
```
σgl = (αo * σgo + αw * σwg) / (αo + αw)
```
— тогда ОК, можно подавать `αo/αw` как угодно, главное одинаково (нормируется).
Если же функция использует напрямую `αo + αw = 1` как предположение — тогда
баг.

**Что надо сделать**: загляни внутрь этой функции и проверь знаменатель.
Из имени неочевидно.

---

## 6. Мелкая ошибка — ветка `A == 1.0` (строки 1166–1183)

```cpp
double profile_param_C0 = 1.0;
const double A = 1.2;
const double B = 0.3;
const double Fv = 1.0;
if (fabs (A - 1.0) > tnm::min_compare)   // всегда true при A=1.2
  {
    ...
    profile_param_C0 = A / (1.0 + (A - 1.0) * beta_norm * beta_norm);
  }
```

С A=1.2 ветка всегда исполняется, так что ошибки нет. Но это мёртвый код —
если когда-нибудь решишь использовать SHI-04 (A=1.0), `profile_param_C0`
останется 1.0 корректно. Но нужно иметь в виду: когда A=1.0, формулой можно
всё равно считать — получится C0≡1, и дополнительная ветка не нужна. Лучше
убрать проверку, формула сама даст 1.0.

И ещё: `if (fabs (profile_param_C0) < tnm::min_division) profile_param_C0 = tnm::min_division;`
— бессмысленно. C0 = A/(1 + (A-1)*β*²) ≥ A/(1 + (A-1)) = 1 при A ≥ 1. Никогда
не станет меньше 1, обрезка не нужна.

---

## 7. Ошибка — flooding_velocity не считается при малом ρg (строки 1162–1164)

```cpp
double flooding_velocity = 0.0;
if (gas_density > tnm::min_compare_well_limit)
  flooding_velocity = Kut_number * sqrt (liq_density / gas_density) * bubble_rise_velocity;
```

Если `gas_density` маленькое (что странно физически, но бывает при
флэш-инициализации), `flooding_velocity = 0`. Тогда в β-формуле (строки
1172–1174):

```cpp
double beta = temp_holdups.gas;
if (flooding_velocity > tnm::min_compare)
  beta *= std::max (1.0, Fv * fabs (mixture_velocity) / flooding_velocity);
```

Ветка не берётся, и `β = αg`. С учётом бага #1 это заметно исказит C0.

Физически это OK (при ρg→0 газ «невесомый» → vsgf→∞ → β→αg), но если по
совокупности ошибок σgl уже искажён, это может цепляться. Советую залогировать
это условие.

---

## 8. Замечание — `run_flash` внутри `test_function` внутри FD-цикла (строки 763–772)

```cpp
if (element_status->component_N_tot > tnm::min_compare)
  {
    if (auto err = run_flash <true> (rep, ...); err != ...) return;
  }
```

FLASH выполняется на каждой FD-оценке (1 + nseg_vars * 2 раз). Это дорого
(на 10-компонентной задаче — x21 вызовов флэша на сегмент). Для больших
симуляций это критично, но это не про корректность, а про скорость.
Для аналитических производных — FLASH можно вызвать один раз, и получить
`∂ξ/∂z`, `∂ρ/∂z` из самого флэша (у тебя в `avg_D_xi`, `avg_D_rho` как раз
они хранятся).

---

## 9. Замечание — `dbg_R_g`, `dbg_R_o` (строки 1421–1422)

```cpp
dbg_R_g = prev_holdups.gas * prev_vels.gas - prev_vels.gas * prev_holdups.gas;  // = 0
dbg_R_o = prev_holdups.oil * prev_vels.oil - prev_vels.oil * prev_holdups.oil;  // = 0
```

Явные нули. Скорее всего, забыл дописать. Если это тест-невязка, она
бессодержательна.

---

## 10. Oil-Water stage — несколько вопросов

### 10.1 σow вычисляется не по ECLIPSE (строки 1317–1328)

```cpp
double gas_oil_interfacial_tension = surf_mult * pipe_gas_oil_interfacial_tension (...);
double gas_wat_interfacial_tension = surf_mult * pipe_gas_wat_interfacial_tension (...);
double wat_oil_interfacial_tension =
    fabs (gas_oil_interfacial_tension * ow_prev.oil
          - gas_wat_interfacial_tension * ow_prev.water);
```

**Это не совпадает с ECLIPSE.** Мануал (стр. 697): «σow can either be supplied
as a tabulated function of pressure with the keyword STOW (in ECLIPSE 100) or
**calculated internally as the difference between the values of σgo and σwg**
obtained from the above correlations».

То есть:
```
σow = |σgo - σwg|
```
Без весов `ow_prev.oil` и `ow_prev.water`. Это явная ошибка.

### 10.2 Формула для vd' (строка 1336) — ОК для ORIGINAL

```cpp
double drift_velocity_OW = 1.53 * bubble_rise_velocity_OW * tnav_pow (1.0 - ow_prev.oil, 2.0);
```

Соответствует Eclipse Eq. 8.86 `vd' = 1.53 vc' (1-αol)²` для дефолтного
набора. OK.

### 10.3 C0' для ORIGINAL

```cpp
double profile_param_C0_OW = 1.0;
if (ow_prev.oil < 0.4)
  profile_param_C0_OW = 1.2;
else if (ow_prev.oil > 0.7)
  profile_param_C0_OW = 1.0;
else
  profile_param_C0_OW = interpolate_y_against_x (ow_prev.oil, 0.4, 0.7, 1.2, 1.0);
```

Соответствует Eclipse Eq. 8.84. OK.

### 10.4 vc' (строки 1330–1334)

```cpp
double bubble_rise_velocity_OW = tnav_pow (
    wat_oil_interfacial_tension * internal_const::grav_metric ()
        * fabs (element_status->phase_rho[PHASE_WATER] - element_status->phase_rho[PHASE_OIL])
        / (element_status->phase_rho[PHASE_WATER] * element_status->phase_rho[PHASE_WATER]),
    0.25);
```

Eclipse Eq. 8.87: `vc' = [σow*g*(ρw-ρo)/ρw²]^(1/4)`. OK.

---

## 11. Композиционная запись расхода — что не так (строки 1441–1472)

### Текущий код

```cpp
for (auto ic = mp.nc0; ic < mp.nc; ++ic)
  wsncs->wsn_component_rate[ic] = 0.0;

for (unsigned int ip = 0; ip < mp.np; ++ip)
  {
    ...
    const double phase_superficial_flux = phase_velocity * phase_holdup * area;
    for (auto ic = mp.nc0; ic < mp.nc; ++ic)
      {
        wsncs->wsn_component_rate[ic] +=
            phase_superficial_flux
            * element_status->component_phase_x[ic * mp.np + ip]
            * element_status->phase_xi[ip];
      }
  }
```

- `phase_superficial_flux = αp * vp * A` — это **объёмный расход фазы** в
  пласте (м³/с). ОК.
- Умножаем на `ξp` (моль/м³) — получаем мольный расход фазы (моль/с). ОК.
- Умножаем на мольную долю компоненты в фазе → мольный расход компоненты. ОК.

Схема верная. **НО**: после всей итерации ты пересчитываешь `wsn_component_rate`
по своим `prev_holdups`, которые сами зависели от seed. Сумма по всем
компонентам должна дать обратно `q_tot` только если DF-итерация сошлась
правильно. Если не сошлась — `Σ component_rate ≠ q_tot`. Это и будет основным
симптомом бага #4.

**Проверка (надо добавить в тест)**:
```cpp
double q_tot_check = 0.0;
for (auto ic = mp.nc0; ic < mp.nc; ++ic)
  q_tot_check += wsncs->wsn_component_rate[ic];
// должно равняться wsncs->wsn_mixture_molar_rate
```

Если не равно — признак, что DF-итерация пересобирает расход.

---

## 12. Итог — что чинить в первую очередь

Ранжирую по влиянию на результат:

| # | Приоритет | Что          | Эффект                                           |
|---|-----------|--------------|-------------------------------------------------|
| 1 | КРИТ      | `fabs(β-B)` → clamp  [0,1] | C0 занижается для β<B, всегда     |
| 2 | КРИТ      | sqrt покрывает весь знаменатель vd   | vd искажён  |
| 4 | КРИТ      | seed для vsg/vsl из заданных расходов| DF не сходится к заданному потоку |
| 5/10.1 | ВЫСОК   | σgl, σow — проверить формулу | ошибка в σ тянет vc, Ku, vd |
| 3 | СРЕДН     | Плавный переход m(θ) → 0 на θ=90° | сходимость якобиана |
| 6–9| НИЗК     | Мертвый код, логирование, эффективность | |

Рекомендую:
1. Исправить #1 и #2 — это тривиальные правки, эффект проверяется руками
   на одном наборе (αg=0.1, j=1 м/с, ρl=1000, ρg=50, σgl=0.02, D=0.1).
2. Затем #4 — реструктуризовать итерацию на одной переменной αg.
3. Затем #5 и #10.1 — залогировать промежуточные σ, сравнить с Eclipse.

---

## Вопросы, на которые хочу ответ от тебя

1. **Что хранится в `seg.wsn->pipe_props.depth_change`** — вертикальная разница
   глубин или длина по стволу? Нужно, чтобы `depth/length = cos(θ)`.
2. **Функция `pipe_gas_liq_interfacial_tension_holdup_weightening`** — как
   она нормирует? Делит ли на `αo + αw`?
3. **Переменная `element_status->phase_xi[ip]`** — это молярная плотность
   фазы (моль/м³)? Или что-то другое? От этого зависит правильность расчёта
   `vsp = Qp_molar / (ξp * A)`.
4. **Есть ли у тебя тестовый кейс с ECLIPSE** (исход ref-теста), чтобы можно
   было сравнивать αg/αo/αw? Хоть один сегмент с известными входами и
   ожидаемым αg будет огромным подспорьем.
5. **В каком режиме сейчас расходится с Eclipse?** Вертикальная труба? Наклон?
   Двухфазный газ-жидкость или трёхфазный?
