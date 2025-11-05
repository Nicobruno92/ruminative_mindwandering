# Spectral Topography from PKL Markers

## Descripción

Este script (`explore_spectral_topography_from_pkl.py`) lee marcadores espectrales pre-computados desde archivos PKL (creados por el convertidor `h5_to_pkl`) y genera topoplots promediando los valores a través de las épocas.

**Diferencia clave con otros scripts:**
- `explore_spectral_topography.py`: Calcula los marcadores on-the-fly desde datos MNE
- `explore_spectral_topography_junifer.py`: Calcula los marcadores usando Junifer on-the-fly
- `explore_spectral_topography_from_pkl.py`: **Lee marcadores ya computados desde PKL** (útil para debugging del pipeline)

## Estructura del PKL

Los archivos PKL contienen marcadores organizados jerárquicamente:

```
pkl_data/
├── markers/
│   ├── EEG_psd_bands_spectralpower/          # Potencia absoluta
│   │   └── _epoch_data/
│   │       └── [epoch_idx]/
│   │           └── _channel_data/
│   │               └── [channel_name]/
│   │                   └── data: array([delta, theta, alpha, beta, gamma])
│   │
│   └── EEG_psd_relative_spectralpower/       # Potencia relativa
│       └── (misma estructura)
│
├── metadata/
│   └── fif_info/
│       ├── n_epochs
│       ├── n_channels
│       ├── channel_names
│       └── sfreq
│
└── epoch_metadata/
    └── [epoch_idx]/
        ├── event_id
        └── onset_sample
```

## Configuración

Edita las variables al inicio del script:

```python
# Path al archivo PKL
PKL_PATH = "/path/to/sub-XX_ses-XX_task-SartX_desc-state_markers.pkl"

# Info del sujeto/sesión (para títulos)
SUBJECT_ID = "07"
SESSION = "Sart1"
DATA_TYPE = "state"  # 'state' o 'evoked'

# Directorio de salida
OUTPUT_DIR = "./spectral_topography_plots_from_pkl"

# Bandas de frecuencia (índices en el array de datos)
FREQ_BANDS = {
    'delta': 0,   # 1-4 Hz
    'theta': 1,   # 4-8 Hz
    'alpha': 2,   # 8-13 Hz
    'beta': 3,    # 13-30 Hz
    'gamma': 4    # 30-45 Hz
}

# Qué marcador usar
MARKER_TO_PLOT = 'absolute'  # 'absolute' o 'relative'

# Método de agregación entre épocas
AGGREGATION = 'trim_mean'  # 'mean', 'median', 'trim_mean'
TRIM_PERCENT = 10.0  # Para trim_mean: % a recortar de cada extremo
```

## Uso

```bash
# Activar el entorno apropiado
conda activate plots  # o el entorno que tenga mne y matplotlib

# Ejecutar el script
python explore_spectral_topography_from_pkl.py
```

## Salida

El script genera:

1. **Plot combinado**: Todos los topoplots en una figura
   - `sub-XX_task-SartX_desc-state_absolute_spectral_topography_from_pkl.png`

2. **Plots individuales**: Un topoplot por banda
   - `sub-XX_task-SartX_desc-state_absolute_delta_topography_from_pkl.png`
   - `sub-XX_task-SartX_desc-state_absolute_theta_topography_from_pkl.png`
   - etc.

## Métodos de Agregación

- **`mean`**: Promedio simple de todas las épocas
- **`median`**: Mediana (robusto a outliers)
- **`trim_mean`**: Promedio recortado (remueve top/bottom X%, default 10%)

El trim_mean es el recomendado ya que coincide con el método usado en el pipeline de Junifer.

## Debugging

El script incluye mensajes verbose que muestran:
- Marcadores disponibles en el PKL
- Número de épocas y canales
- Estadísticas de potencia por banda (mean, std, min, max)

Para desactivar: `VERBOSE = False`

## Comparación con Otros Métodos

Este script es útil para:
1. **Verificar el pipeline**: Comparar los marcadores guardados en PKL con los calculados on-the-fly
2. **Debugging**: Inspeccionar qué valores están realmente en los PKL
3. **Rapidez**: No recalcula, solo lee y promedia

## Notas Técnicas

- Los marcadores espectrales usan **64 canales EEG** (sin EOG)
- Cada canal tiene **5 valores** (uno por banda de frecuencia)
- El script maneja NaNs automáticamente (los excluye del promedio)
- Usa montaje estándar `biosemi64` para las posiciones de los electrodos
