# Solución: Problemas de Coherencia Espacial en Topografías

## Diagnóstico: DOS PROBLEMAS OPUESTOS

**El problema NO son los valores t** - están en rangos razonables (-6 a +6).

### Problema 1: "BUBBLES" (Fragmentación)
- 7 marcadores con múltiples clusters pequeños y aislados (≥3 clusters)
- 17 marcadores sin clusters a pesar de t-values razonables
- **Causa**: TFCE E=0.5 demasiado bajo → favorece picos focales

### Problema 2: "TODO SIGNIFICATIVO" (Inflación de Falsos Positivos) ⚠️
- Topografías completamente azules (todos los canales significativos)
- **Estadísticamente improbable** que TODOS los canales sean significativos
- **Causas posibles**:
  1. **n_permutations=100 DEMASIADO BAJO** → p-values inflados
  2. **Threshold demasiado liberal** (si usando método threshold)
  3. **Sin corrección de comparaciones múltiples** o corrección inadecuada
  4. **Normalización problemática** → reduce varianza real

## Causa Raíz Principal

### ⚠️ CRÍTICO: n_permutations=100 es INSUFICIENTE

Con solo 100 permutaciones:
- **Resolución mínima de p-value**: 1/100 = 0.01
- **No puedes detectar p < 0.01** → todo parece significativo
- **P-values inflados** → falsos positivos masivos
- **Clusters inestables** → resultados no reproducibles

**Mínimo absoluto**: 1000 permutaciones (resolución p=0.001)
**Recomendado**: 5000 permutaciones (resolución p=0.0002)
**Para publicación**: 10000 permutaciones

## Solución Integral

### Cambios CRÍTICOS en `config.yaml`:

```yaml
clustering:
  method: 'tfce'
  
  tfce:
    E: 1.0  # ← AUMENTAR de 0.5 (mejora coherencia espacial)
    H: 2.0  # Mantener estándar
    n_steps: 300
  
  n_permutations: 5000  # ← CRÍTICO: AUMENTAR de 100
  alpha: 0.05
  
  # Threshold para formación de clusters (si usas método threshold)
  threshold: 2.5  # ← Considerar AUMENTAR a 3.0 si todo es significativo
```

### Verificar Corrección de Comparaciones Múltiples:

```yaml
multiple_comparisons:
  evoked: "fdr_bh"  # ← Verificar que esté activado
  state: "fdr_bh"   # ← Verificar que esté activado
  alpha: 0.05
```

## Estrategia de Solución por Etapas

### ETAPA 1: Aumentar Permutaciones (OBLIGATORIO)

```yaml
clustering:
  n_permutations: 5000  # Mínimo para resultados confiables
```

**Esto debería resolver el problema de "todo significativo"**

### ETAPA 2: Ajustar TFCE para Coherencia Espacial

**Opción A - BALANCED (Recomendado):**
```yaml
tfce:
  E: 1.0  # Balance entre sensibilidad y coherencia
  H: 2.0
```

**Opción B - SMOOTH (Si persisten bubbles):**
```yaml
tfce:
  E: 1.5  # Mayor énfasis en extensión espacial
  H: 2.0
```

**Opción C - CONSERVATIVE (Si todo sigue significativo):**
```yaml
tfce:
  E: 1.0
  H: 2.5  # Aumentar H → más conservador
```

### ETAPA 3: Ajustar Threshold (Solo si usas method='threshold')

Si todavía todo es significativo:
```yaml
clustering:
  method: 'threshold'
  threshold: 3.0  # ← AUMENTAR de 2.5 (más conservador)
  stat_fun: 'sum'
```

### ETAPA 4: Verificar Normalización

```yaml
preprocessing:
  normalize_by_subject: true
  normalization_method: "zscore"  # ← Verificar que sea zscore
  channel_wise: false  # ← DEBE ser false
```

⚠️ **Si channel_wise=true**: Cambiar a false inmediatamente
- channel_wise=true destruye patrones espaciales
- Puede causar tanto bubbles como falsos positivos

## Análisis del Problema "Todo Significativo"

### ¿Por qué ocurre?

1. **Permutaciones insuficientes (100)**:
   - P-value mínimo = 0.01
   - Con α=0.05, casi todo parece significativo
   - Solución: 5000+ permutaciones

2. **Threshold muy bajo**:
   - threshold=2.0 es muy liberal
   - Muchos canales superan este umbral por azar
   - Solución: threshold=2.5 o 3.0

3. **Normalización excesiva**:
   - Reduce varianza real → infla t-values
   - channel_wise=true es especialmente problemático
   - Solución: channel_wise=false, método robust

4. **Sin corrección múltiple**:
   - 64 canales × múltiples marcadores = muchas comparaciones
   - Sin corrección → inflación masiva de falsos positivos
   - Solución: Verificar que FDR esté activado

## Configuración Recomendada Final

```yaml
# Preprocessing
preprocessing:
  normalize_by_subject: true
  normalization_method: "zscore"  # O "robust" si hay outliers
  channel_wise: false  # ← CRÍTICO: false para coherencia espacial

# LMM
lmm:
  formula: "power ~ onoff + time_on_task + valence + selfother + time + (1|subject)"
  predictor_of_interest: "onoff"
  method: "powell"
  maxiter: 2000

# Clustering
clustering:
  method: "tfce"
  
  tfce:
    E: 1.0  # Balance coherencia/sensibilidad
    H: 2.0  # Estándar
    n_steps: 300
  
  # CRÍTICO: Aumentar permutaciones
  n_permutations: 5000  # Mínimo para p-values confiables
  alpha: 0.05
  seed: 42
  n_jobs: -1
  
  # Threshold alternativo (si TFCE no funciona)
  threshold: 2.5  # O 3.0 si todo es significativo
  tail: 0
  stat_fun: "sum"

# Corrección comparaciones múltiples
multiple_comparisons:
  evoked: "fdr_bh"  # Benjamini-Hochberg FDR
  state: "fdr_bh"
  alpha: 0.05
```

## Plan de Acción Paso a Paso

### 1. Verificar Configuración Actual

```bash
# Ver configuración actual
cat Statistics/config.yaml | grep -A 5 "tfce:"
cat Statistics/config.yaml | grep "n_permutations"
cat Statistics/config.yaml | grep "channel_wise"
cat Statistics/config.yaml | grep -A 3 "multiple_comparisons"
```

### 2. Aplicar Cambios Críticos

**Cambios OBLIGATORIOS:**
- ✅ `n_permutations: 100 → 5000` (YA HECHO)
- ✅ `tfce.E: 0.5 → 1.0` (YA HECHO)
- ⚠️ Verificar `channel_wise: false`
- ⚠️ Verificar `multiple_comparisons` activado

### 3. Re-ejecutar Análisis

```bash
# Opción A: Un marcador de prueba (rápido)
python Statistics/run_pipeline.py

# Opción B: Todos los marcadores (cluster)
sbatch Statistics/submit_marker_array.sh
```

### 4. Evaluar Resultados

**Buscar:**
- ✅ Menos canales significativos (no todo azul)
- ✅ Clusters más grandes y coherentes (no bubbles)
- ✅ Patrones fisiológicamente plausibles
- ✅ P-values más conservadores

**Si todavía todo es significativo:**
- Aumentar threshold a 3.0
- Aumentar H a 2.5
- Verificar que FDR esté funcionando

**Si persisten bubbles:**
- Aumentar E a 1.5 o 2.0
- Probar stat_fun='sum' en lugar de 'max'

## Diagnóstico de Resultados

### Señales de ÉXITO:
- ✅ 20-40% de canales significativos (no 100%)
- ✅ 1-2 clusters grandes por marcador (no 4+)
- ✅ Clusters contiguos espacialmente
- ✅ Patrones simétricos o anatómicamente coherentes

### Señales de PROBLEMA:
- ❌ >80% de canales significativos → threshold muy bajo
- ❌ 3+ clusters pequeños → E muy bajo
- ❌ Patrones aleatorios → permutaciones insuficientes
- ❌ Todo no-significativo → threshold muy alto

## Casos Especiales

### Si TFCE no funciona bien:

**Alternativa 1: Threshold-based clustering**
```yaml
clustering:
  method: 'threshold'
  threshold: 2.5  # Ajustar según necesidad
  stat_fun: 'sum'  # Favorece extensión espacial
  n_permutations: 5000
```

**Alternativa 2: Aumentar threshold TFCE**
```yaml
clustering:
  method: 'tfce'
  tfce:
    E: 1.0
    H: 2.5  # ← Aumentar H (más conservador)
```

### Si hay outliers en los datos:

```yaml
preprocessing:
  normalization_method: "robust"  # En lugar de zscore
```

### Si hay pocos sujetos con varianza:

```yaml
project:
  min_predictor_variability: 3  # Reducir de 5
  # O
  min_predictor_variability: "auto"  # Solo excluir varianza cero
```

## Tiempo de Ejecución

Con `n_permutations=5000`:
- **Por marcador**: ~1-2 horas (con n_jobs=-1)
- **52 marcadores (SLURM array)**: ~2-3 horas total
- **52 marcadores (secuencial)**: ~52-104 horas

**Recomendación**: Usar SLURM array para paralelizar

## Scripts de Diagnóstico

### 1. Analizar Coherencia Espacial
```bash
python Statistics/analyze_spatial_coherence.py
```
- Identifica bubbles vs todo-significativo
- Genera visualizaciones
- Recomienda ajustes

### 2. Verificar P-values
```python
# En Python
import pandas as pd
summary = pd.read_csv("results/.../SUMMARY_REPORT_*.csv")

# Ver distribución de p-values
print(summary[['Marker Name', 'Sig Clusters (uncorr)', 'Sig Clusters (corr)']].head(20))

# Contar marcadores con todo significativo
print(f"Marcadores con >50% canales sig: {(summary['Sig Clusters (uncorr)'] > 32).sum()}")
```

## Resumen Ejecutivo

### 🔴 Problema Principal: n_permutations=100
- **Causa**: P-values inflados, resolución insuficiente
- **Efecto**: Todo parece significativo O nada es significativo
- **Solución**: n_permutations=5000 (YA IMPLEMENTADO)

### 🟡 Problema Secundario: TFCE E=0.5
- **Causa**: Favorece picos focales sobre extensión
- **Efecto**: Bubbles (clusters fragmentados)
- **Solución**: E=1.0 (YA IMPLEMENTADO)

### ✅ Próximos Pasos
1. Verificar que cambios estén en config.yaml
2. Re-ejecutar análisis completo
3. Evaluar si resultados son más razonables
4. Ajustar E o threshold si es necesario

### 📊 Expectativa de Resultados
- **Antes**: 100% canales significativos O bubbles fragmentadas
- **Después**: 20-40% canales significativos, 1-2 clusters coherentes
- **Interpretación**: Patrones fisiológicamente plausibles

## Referencias

**TFCE:**
- Smith & Nichols (2009). NeuroImage, 44(1), 83-98
- Parámetros EEG: E=1.0-1.5, H=2.0

**Permutation Testing:**
- Maris & Oostenveld (2007). J Neurosci Methods, 164(1), 177-190
- Mínimo: 1000, Recomendado: 5000+

**Multiple Comparisons:**
- Benjamini & Hochberg (1995). J Royal Stat Soc B, 57(1), 289-300
- FDR control para múltiples marcadores
