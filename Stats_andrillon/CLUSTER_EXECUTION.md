# Ejecución en Cluster - Pipeline Andrillon

## ✅ Correcciones Aplicadas

### Problemas Resueltos
1. **Conda activation** - Agregado `eval "$(conda shell.bash hook)"`
2. **Import errors** - Corregidos todos los imports de `reader.py` y `cluster_test.py`
3. **Array job** - Sistema automático de detección de markers

---

## 🚀 Ejecución Recomendada

### Opción 1: Automática (RECOMENDADA)

```bash
# Conectar al cluster
ssh cluster

# Ir al directorio
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

# Ejecutar script helper (detecta markers automáticamente)
bash Stats_andrillon/submit_parallel_andrillon.sh
```

**Esto hará**:
- Detectar automáticamente el número de markers (ej: 49)
- Enviar job con `--array=0-48`
- Cada task procesa un marker diferente

---

### Opción 2: Manual

Si necesitas control manual:

```bash
# 1. Ver cuántos markers hay
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Stats_andrillon
python get_marker_list.py --config config_andrillon.yaml --format count

# Output esperado: "Total markers: 49" (o similar)

# 2. Enviar job con el rango correcto (0 a N-1)
sbatch --array=0-48 submit_andrillon_array.sh
```

---

## 📊 Monitoreo

```bash
# Ver estado de los jobs
squeue -u $USER

# Ver logs en tiempo real
tail -f logs/andrillon_JOBID_*.out

# Ver errores
tail -f logs/andrillon_JOBID_*.err

# Contar cuántos completaron
ls logs/andrillon_JOBID_*.out | wc -l
```

---

## 🔍 Verificación de Resultados

```bash
# Ver estructura de resultados
tree results/andrillon_cluster/

# Contar markers procesados
ls results/andrillon_cluster/*/EEG_*/*.pkl | wc -l

# Ver clusters encontrados
grep "Found.*clusters" logs/andrillon_JOBID_*.out
```

---

## ⚠️ Troubleshooting

### Error: "No marker found for array task X"
**Causa**: Array range mayor que número de markers  
**Solución**: Usar `submit_parallel_andrillon.sh` en lugar de enviar manualmente

### Error: "cannot import name 'load_marker_data'"
**Causa**: Import incorrecto (ya corregido)  
**Verificar**: Los cambios están aplicados en el cluster

### Error: "CondaError: Run 'conda init'"
**Causa**: Conda no inicializado (ya corregido)  
**Verificar**: `submit_andrillon_array.sh` tiene `eval "$(conda shell.bash hook)"`

### Job se queda en PENDING
**Causas posibles**:
- Cluster ocupado (normal, esperar)
- Recursos no disponibles (reducir `--mem` o `--cpus-per-task`)
- Límite de jobs alcanzado

```bash
# Ver razón del pending
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %.20R"
```

---

## 📝 Archivos Importantes

### Scripts de Ejecución
- `submit_parallel_andrillon.sh` - **Usar este** (automático)
- `submit_andrillon_array.sh` - Array job (no usar directamente)
- `run_andrillon_pipeline.py` - Script Python principal

### Configuración
- `config_andrillon.yaml` - Parámetros del pipeline
- `get_marker_list.py` - Helper para listar markers

### Logs
- `logs/andrillon_JOBID_TASKID.out` - Output por task
- `logs/andrillon_JOBID_TASKID.err` - Errores por task

---

## 🎯 Workflow Completo

```bash
# 1. Conectar y navegar
ssh cluster
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

# 2. (Opcional) Verificar configuración
cat Stats_andrillon/config_andrillon.yaml

# 3. Enviar jobs
bash Stats_andrillon/submit_parallel_andrillon.sh

# 4. Monitorear
watch -n 10 'squeue -u $USER'

# 5. Cuando termine, verificar resultados
ls -lh results/andrillon_cluster/
```

---

## 💡 Tips

1. **Primera vez**: Ejecuta con un solo marker para verificar
   ```bash
   python Stats_andrillon/run_andrillon_pipeline.py \
       --config Stats_andrillon/config_andrillon.yaml \
       --marker-index 0
   ```

2. **Recursos**: Si hay problemas de memoria, ajusta en `submit_andrillon_array.sh`:
   ```bash
   #SBATCH --mem=32G  # Aumentar si es necesario
   ```

3. **Tiempo**: Si los jobs se cancelan por timeout, aumenta:
   ```bash
   #SBATCH --time=48:00:00  # 48 horas
   ```

4. **Logs**: Crear directorio de logs antes de enviar:
   ```bash
   mkdir -p logs
   ```

---

## ✅ Checklist Pre-Ejecución

- [ ] Configuración revisada (`config_andrillon.yaml`)
- [ ] Directorio de logs existe (`mkdir -p logs`)
- [ ] Conda environment `ML` disponible
- [ ] Módulo `proxy` cargable (`module load proxy`)
- [ ] Path de montage correcto en config
- [ ] Features root accesible desde cluster

---

**Última actualización**: 2025-11-13  
**Estado**: ✅ Todos los errores corregidos, listo para producción
