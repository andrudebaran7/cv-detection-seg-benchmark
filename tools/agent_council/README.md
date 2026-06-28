# Agent Council 🧑‍⚖️🤝

Herramientas multi-agente (LangGraph) para apoyar **decisiones técnicas** del
benchmark sin atarse a un solo modelo. Dos patrones, ambos **cross-family**
(modelos de proveedores distintos vía OpenRouter → menos sesgo de confirmación):

| Comando | Patrón | Para qué |
|---|---|---|
| `moa`       | **Mixture-of-Agents** (consenso) | N expertos responden en paralelo → un sintetizador combina. Decisiones de diseño/arquitectura. |
| `audit`     | **Worker + Auditor con retry** | Un worker propone, un auditor de otra familia revisa (APROBADO/CAMBIOS) y se reintenta o escala. Verificar código/decisiones antes de aceptarlas. |
| `peer-rank` | **Evaluación cruzada** (Borda) | N expertos responden y cada uno rankea a los demás (anonimizado, sin auto-voto) → ranking de consenso + ganador. Saber QUÉ respuesta es mejor. |

> Es una herramienta **auxiliar/experimental** del repo (no toca el pipeline de
> modelos). Vive aislada en `tools/agent_council/` con sus propias dependencias.

## Requisitos

- Python 3.10+
- Una API key de [OpenRouter](https://openrouter.ai/keys) en el entorno:
  ```bash
  export OPENROUTER_API_KEY="sk-or-..."
  ```

## Instalación (entorno aislado, recomendado)

No mezclar con las deps de CV del repo. Crear un venv propio:

```bash
cd tools/agent_council
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Uso

```bash
# Consenso (MoA) sobre una decisión de arquitectura
.venv/bin/python council.py moa "¿YOLO11n o RF-DETR-nano para edge sin GPU?"

# Auditoría con retry de una implementación
.venv/bin/python council.py audit "Implementa NMS vectorizado en numpy para cajas xyxy."

# Evaluación cruzada (peer-rank) de una decisión
.venv/bin/python council.py peer-rank "¿Mejor estrategia de data augmentation para YOLO-nano?"
```

Sin argumentos, cada comando usa una pregunta/tarea de ejemplo del dominio del repo.

## Modelos (editables en `council.py`)

- **MoA workers:** `deepseek/deepseek-v3.2`, `nvidia/nemotron-3-super-120b-a12b`,
  `qwen/qwen3-235b-a22b-2507` · **sintetizador:** `google/gemini-2.5-flash`
- **Auditor:** worker `deepseek/deepseek-v3.2` + auditor `anthropic/claude-sonnet-4.6`
  (familia distinta), `max_retries=2`

Todos son baratos por token; el coste por consulta es de centavos. Revisa precios
actuales en OpenRouter antes de cambiarlos.

## Notas

- `audit` puede terminar en **ESCALADO** si el auditor no aprueba tras los
  reintentos — es el comportamiento seguro (no aprobar trabajo sin verificar).
- El catálogo de modelos de OpenRouter cambia; si un modelo deja de existir,
  edítalo en `council.py`.
