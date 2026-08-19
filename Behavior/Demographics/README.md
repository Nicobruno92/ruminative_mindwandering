# Demographics — comparación de grupos (demografía y psicometría)

Pipeline chico de dos scripts que compara **Controles** vs. **Riesgo de depresión**
en variables demográficas y psicométricas de la cohorte CyberSART. Es el insumo de
la muestra que se reporta en la **Sección 1 (Behavior)** del Paper 2 y en el
apartado de participantes del Paper 1 — ver `CLAUDE.md` en la raíz del repo.

Snapshot anclado a `results/demographics/` a fecha **2026-07-27**.

---

## 1. Qué hace

| Script | Qué produce | Entorno |
|---|---|---|
| `group_comparison_demographics_psychometrics.py` | `results/demographics/group_comparison_demographics_psychometrics.csv` — una fila por variable, con test, estadístico, p-valor y tamaño de efecto | `plots` (pandas, scipy, openpyxl) |
| `demographics_raincloud_plots.py` | `results/demographics/raincloud_plots/raincloud_grid_demographics.png` y `..._psychometrics.png` | `plots` (requiere `ptitprince`) |

Fuente de datos: `metadata_CyberSART.xlsx` (raíz del repo), columna `group` (1/2).
Ambos scripts corren tal cual, sin argumentos — toda la configuración está al
principio del archivo.

```bash
conda activate plots
python Behavior/Demographics/group_comparison_demographics_psychometrics.py
python Behavior/Demographics/demographics_raincloud_plots.py   # requiere el CSV de arriba
```

---

## 2. Muestra

`metadata_CyberSART.xlsx` tiene **43 filas** (una por sujeto reclutado), agrupadas
por la columna `group`:

| Grupo | Etiqueta | N | % mujeres | Edad (media ± DE, rango) |
|---|---|---|---|---|
| 1 | Controles | 31 | 51.6 % (16/31) | 48.5 ± 11.4 (29–65) |
| 2 | Riesgo de depresión | 12 | 75.0 % (9/12) | 53.8 ± 7.5 (39–64) |
| **Total** | | **43** | 58.1 % (25/43) | |

Ninguna de las tres variables demográficas difiere significativamente entre
grupos al nivel convencional (edad p=.090, sexo χ²(1)=1.10, p=.294, año de
nacimiento p=.065) — los grupos están razonablemente balanceados en estos ejes,
aunque la diferencia de edad es marginal y va en la dirección de controles más
jóvenes.

> **Atención — discrepancia con la muestra de EEG**: `results/ERPs_new/participants/`
> (y el resto del pipeline de EEG/comportamiento) tiene **42 sujetos**, `sub-02` a
> `sub-43` (sin `sub-01`, per la regla de IDs del `CLAUDE.md` raíz). Este script de
> demographics compara los **43** sujetos de la planilla de metadata sin aplicar
> ese filtro. No encontré en el código la razón por la que `sub-01` está en la
> metadata pero no en las derivadas de EEG (¿piloto? ¿exclusión no documentada?).
> Antes de reportar el N final en el paper, confirmar si `sub-01` debe excluirse
> también de esta tabla de demographics para que el N coincida con el resto del
> pipeline.

---

## 3. Variables psicométricas

| Columna | Instrumento | Notas |
|---|---|---|
| `bdi` | Beck Depression Inventory | severidad de síntomas depresivos |
| `rrs_d`, `rrs_r`, `rrs_b`, `rrs_tot` | Ruminative Response Scale (RRS) | subescalas *Depression-related*, *Reflection*, *Brooding*, y total |
| `mwq` | Mind-Wandering Questionnaire | frecuencia de mind-wandering autorreportada |
| `sris` | Self-Reflection and Insight Scale | |
| `fne` | Fear of Negative Evaluation | |
| `self-esteem` | Rosenberg Self-Esteem Scale (asumido) | |
| `a_rsq` | Amsterdam Resting-State Questionnaire (ARSQ) | |
| `ctq_ae`, `ctq_ap`, `ctq_as`, `ctq_ne`, `ctq_np`, `ctq_tot` | Childhood Trauma Questionnaire (CTQ) | *Emotional/Physical/Sexual Abuse*, *Emotional/Physical Neglect*, total |
| `qf_3`, `qd_4`, `qf_7b` | **sin identificar** | por el patrón de faltantes (ver §4) y el nombre (`qf`/`qd` ≈ "quantité/fréquence"), parecen ítems de un cuestionario de consumo de alcohol, pero no hay codebook en el repo que lo confirme — **verificar contra el instrumento original antes de interpretarlas o publicarlas** |

---

## 4. Resultados y faltantes (`group_comparison_demographics_psychometrics.csv`)

Test usado: Welch's t (continuas) / χ² de independencia (`gender`, categórica).
**Los p-valores no están corregidos por comparaciones múltiples** (22 tests en
total) — tratar como exploratorio, no confirmatorio, salvo que se aplique FDR
antes de reportar en el paper.

Significativos a p<.05 sin corregir (todos con n=30 controles / 12 riesgo salvo
que se indique):

| Variable | Controles | Riesgo de depresión | p | d |
|---|---|---|---|---|
| `bdi` | 2.7 ± 3.5 | 11.8 ± 7.5 | .0013 | −1.86 |
| `rrs_d` | 17.2 ± 4.1 | 28.1 ± 7.0 | <.001 | −2.15 |
| `rrs_tot` | 33.0 ± 9.4 | 47.6 ± 11.0 | <.001 | −1.48 |
| `mwq` | 11.5 ± 4.4 | 19.7 ± 6.1 | <.001 | −1.67 |
| `self-esteem` | 34.6 ± 4.7 | 29.2 ± 6.2 | .0140 | 1.06 |
| `qd_4` (n=14/12) | 3.1 ± 0.8 | 2.1 ± 1.2 | .0196 | 1.05 |

Marginales (p<.10): `age` (.090), `year_birth` (.065), `ctq_ae` (.092, n=21/12),
`rrs_b` (.096). El resto (`a_rsq`, `rrs_r`, `sris`, `fne`, `ctq_ap/as/ne/np/tot`,
`qf_3`, `qf_7b`) no muestra diferencias.

**Faltantes concentrados en el grupo Controles**: las subescalas CTQ solo tienen
21/31 controles (10 faltantes) vs. 12/12 completos en Riesgo de depresión; `qd_4`
es peor todavía (14/31 vs. 12/12, 17 faltantes en total). `qf_3` y `qf_7b` también
faltan exclusivamente en Controles (5 y 3, respectivamente). Esto sugiere que esos
cuestionarios se agregaron o completaron tarde en el protocolo para una parte de
los controles — con faltantes tan desbalanceados entre grupos, `ctq_*`, `qd_4`,
`qf_3` y `qf_7b` deberían leerse con cautela adicional (n efectivo bajo y no
necesariamente representativo del grupo Controles completo).

---

## 5. Salidas

```
results/demographics/
├── group_comparison_demographics_psychometrics.csv   # una fila por variable
└── raincloud_plots/
    ├── raincloud_grid_demographics.png                # age, year_birth
    └── raincloud_grid_psychometrics.png                # las 19 variables psicométricas continuas
```

El asterisco en los raincloud plots marca `p < .05` **sin corregir** (ver §4).

---

## 6. Nota de estilo pendiente

`demographics_raincloud_plots.py` usa `sns.color_palette("Set2", 2)` en vez de
`color_palette.yaml` (fuente única de colores del proyecto, ver `CLAUDE.md` raíz).
Si estos plots van al paper, migrar la paleta a `color_palette.yaml` para
mantener consistencia con el resto de las figuras (`onoff`=rojo, etc. — aunque
acá el eje relevante es grupo, no dimensión, así que probablemente convenga
definir explícitamente qué color va a Controles vs. Riesgo de depresión en la
paleta compartida en vez de dejarlo en manos de Seaborn).
