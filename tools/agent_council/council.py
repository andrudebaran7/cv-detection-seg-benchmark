"""
Agent Council — herramientas multi-agente para decisiones técnicas del benchmark.

Dos patrones, ambos cross-family (modelos de proveedores distintos vía OpenRouter,
para reducir el sesgo de confirmación):

  • moa       — Mixture-of-Agents: N workers responden en paralelo y un sintetizador
                combina sus respuestas. Útil para decisiones de diseño/arquitectura.
  • audit     — Worker + Auditor con retry: un worker propone, un auditor de otra
                familia revisa (APROBADO/CAMBIOS) y se reintenta o escala.
  • peer-rank — Evaluación cruzada: N workers responden y cada uno rankea a los demás
                (anonimizado, sin auto-voto); agregación Borda -> mejor respuesta.

Requiere OPENROUTER_API_KEY en el entorno. Entorno aislado recomendado (ver README).

Uso:
  python council.py moa       "¿YOLO11n o RF-DETR-nano para detección en edge sin GPU?"
  python council.py audit     "Implementa NMS (non-max suppression) vectorizado en numpy."
  python council.py peer-rank "¿Mejor estrategia de data augmentation para YOLO-nano?"
"""
from __future__ import annotations

import argparse
import operator
import os
import re
import sys
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        sys.exit("ERROR: define OPENROUTER_API_KEY en el entorno.")
    return k


def llm(model: str, temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(model=model, api_key=_key(), base_url=OPENROUTER_BASE,
                      temperature=temperature, timeout=120, max_retries=3)


# ----------------------------------------------------------------------------
# Patrón B — Mixture-of-Agents (consenso)
# ----------------------------------------------------------------------------
MOA_WORKERS = {
    "DeepSeek (v3.2)": "deepseek/deepseek-v3.2",
    "Nvidia (Nemotron-120B)": "nvidia/nemotron-3-super-120b-a12b",
    "Qwen (3-235B)": "qwen/qwen3-235b-a22b-2507",
}
MOA_SYNTH = "google/gemini-2.5-flash"

_MOA_WORKER_SYS = (
    "Eres un experto independiente en visión por computadora y ML. Responde de "
    "forma directa, concisa y con tu mejor razonamiento. Da TU respuesta."
)
_MOA_SYNTH_SYS = (
    "Eres el sintetizador de un Mixture-of-Agents. Combina las respuestas de los "
    "expertos en UNA final superior; resuelve contradicciones e indica brevemente "
    "en qué coincidieron y en qué difirieron."
)


class MoAState(TypedDict):
    question: str
    responses: Annotated[list, operator.add]
    synthesis: str


def _moa_worker(name: str, model: str):
    def node(state: MoAState) -> dict:
        m = llm(model).invoke([("system", _MOA_WORKER_SYS), ("user", state["question"])])
        return {"responses": [(name, model, (m.content or "").strip())]}
    return node


def _moa_synth(state: MoAState) -> dict:
    bloques = "\n\n".join(
        f"### Experto {i+1}: {n} [{mdl}]\n{r}"
        for i, (n, mdl, r) in enumerate(state["responses"]))
    prompt = (f"PREGUNTA:\n{state['question']}\n\nRESPUESTAS:\n{bloques}\n\n"
              f"Sintetiza la respuesta final.")
    m = llm(MOA_SYNTH, 0.2).invoke([("system", _MOA_SYNTH_SYS), ("user", prompt)])
    return {"synthesis": (m.content or "").strip()}


def build_moa():
    g = StateGraph(MoAState)
    for name, model in MOA_WORKERS.items():
        nid = "w_" + name.split()[0].lower()
        g.add_node(nid, _moa_worker(name, model))
        g.add_edge(START, nid)
        g.add_edge(nid, "synth")
    g.add_node("synth", _moa_synth)
    g.add_edge("synth", END)
    return g.compile()


def run_moa(question: str) -> dict:
    return build_moa().invoke({"question": question, "responses": [], "synthesis": ""})


# ----------------------------------------------------------------------------
# Patrón A — Auditor con retry
# ----------------------------------------------------------------------------
AUDIT_WORKER = "deepseek/deepseek-v3.2"
AUDIT_AUDITOR = "anthropic/claude-sonnet-4.6"
AUDIT_MAX_RETRIES = 2

_AUD_WORKER_SYS = (
    "Eres un ingeniero competente (visión/ML). Resuelve la tarea correcta y "
    "completamente. Si recibes feedback de un auditor sobre un intento previo, "
    "corrige TODOS los puntos y entrega la versión mejorada."
)
_AUD_AUDITOR_SYS = (
    "Eres un auditor técnico riguroso. Revisa la solución frente a la tarea: "
    "correctitud, edge cases, errores sutiles, afirmaciones sin justificar. "
    "Empieza EXACTAMENTE con 'VEREDICTO: APROBADO' o 'VEREDICTO: CAMBIOS', y "
    "después lista los problemas concretos (o por qué es correcta). Sé específico."
)


class AuditState(TypedDict):
    task: str
    draft: str
    audit: str
    approved: bool
    retries: int
    trace: list


def _aud_worker(state: AuditState) -> dict:
    if state.get("audit"):
        user = (f"TAREA:\n{state['task']}\n\nINTENTO PREVIO:\n{state['draft']}\n\n"
                f"FEEDBACK DEL AUDITOR:\n{state['audit']}\n\n"
                f"Entrega la versión corregida que resuelva TODOS los puntos.")
    else:
        user = f"TAREA:\n{state['task']}\n\nEntrega tu solución."
    m = llm(AUDIT_WORKER, 0.3).invoke([("system", _AUD_WORKER_SYS), ("user", user)])
    return {"draft": (m.content or "").strip()}


def _aud_auditor(state: AuditState) -> dict:
    user = (f"TAREA:\n{state['task']}\n\nSOLUCIÓN A AUDITAR:\n{state['draft']}\n\n"
            f"Audita según los criterios.")
    m = llm(AUDIT_AUDITOR, 0.1).invoke([("system", _AUD_AUDITOR_SYS), ("user", user)])
    audit = (m.content or "").strip()
    vline = audit.splitlines()[0].upper() if audit else ""
    approved = "APROBADO" in vline and "CAMBIOS" not in vline
    trace = state.get("trace", []) + [{
        "intento": state.get("retries", 0) + 1, "draft": state["draft"],
        "audit": audit, "approved": approved}]
    return {"audit": audit, "approved": approved, "trace": trace}


def _aud_route(state: AuditState) -> str:
    if state["approved"]:
        return "accept"
    return "retry" if state.get("retries", 0) < AUDIT_MAX_RETRIES else "escalate"


def _aud_retry(state: AuditState) -> dict:
    return {"retries": state.get("retries", 0) + 1}


def build_auditor():
    g = StateGraph(AuditState)
    g.add_node("worker", _aud_worker)
    g.add_node("auditor", _aud_auditor)
    g.add_node("retry", _aud_retry)
    g.add_edge(START, "worker")
    g.add_edge("worker", "auditor")
    g.add_conditional_edges("auditor", _aud_route,
                            {"accept": END, "retry": "retry", "escalate": END})
    g.add_edge("retry", "worker")
    return g.compile()


def run_auditor(task: str) -> dict:
    return build_auditor().invoke({"task": task, "draft": "", "audit": "",
                                   "approved": False, "retries": 0, "trace": []})


# ----------------------------------------------------------------------------
# Patrón C — Peer-Rank (evaluación cruzada + Borda)
# ----------------------------------------------------------------------------
PEERRANK_WORKERS = {
    "DeepSeek": "deepseek/deepseek-v3.2",
    "Nvidia": "nvidia/nemotron-3-super-120b-a12b",
    "Qwen": "qwen/qwen3-235b-a22b-2507",
    "Gemini": "google/gemini-2.5-flash",
}

_PR_WORKER_SYS = (
    "Eres un experto independiente en visión por computadora y ML. Responde de "
    "forma directa y con tu mejor razonamiento. Da TU respuesta."
)
_PR_EVAL_SYS = (
    "Eres un evaluador imparcial. Te dan varias respuestas ANÓNIMAS a una pregunta. "
    "Júzgalas por correctitud, rigor, completitud y utilidad. No sabes quién las "
    "escribió. Razona brevemente y termina con UNA línea exactamente así:\n"
    "ORDEN: X,Y,Z\n"
    "listando las etiquetas de mejor a peor (p.ej. 'ORDEN: B,A,C')."
)


class PeerRankState(TypedDict):
    question: str
    answers: Annotated[list, operator.add]   # (name, model, answer)
    rankings: list
    scores: dict
    report: str


def _pr_worker(name: str, model: str):
    def node(state: PeerRankState) -> dict:
        m = llm(model).invoke([("system", _PR_WORKER_SYS), ("user", state["question"])])
        return {"answers": [(name, model, (m.content or "").strip())]}
    return node


def _pr_parse_orden(text: str, valid: list) -> list:
    m = re.search(r"ORDEN\s*:\s*([A-Za-z0-9 ,]+)", text)
    seq = []
    if m:
        for tok in m.group(1).split(","):
            t = tok.strip().upper()
            if t in valid and t not in seq:
                seq.append(t)
    for v in valid:                      # robustez: completar las que falten
        if v not in seq:
            seq.append(v)
    return seq


def _pr_rank(state: PeerRankState) -> dict:
    answers = {name: ans for name, _m, ans in state["answers"]}
    names = list(answers.keys())
    scores = {n: 0.0 for n in names}
    rankings = []
    for evaluator in names:
        peers = [n for n in names if n != evaluator]          # excluye la propia
        labels = [chr(ord("A") + i) for i in range(len(peers))]
        label2name = dict(zip(labels, peers))
        bloque = "\n\n".join(
            f"### Respuesta {lab}\n{answers[label2name[lab]]}" for lab in labels)
        prompt = (f"PREGUNTA:\n{state['question']}\n\nRESPUESTAS ANÓNIMAS:\n{bloque}\n\n"
                  f"Ranquéalas de mejor a peor.")
        out = llm(PEERRANK_WORKERS[evaluator], 0.1).invoke(
            [("system", _PR_EVAL_SYS), ("user", prompt)])
        order = _pr_parse_orden(out.content or "", labels)
        m = len(order)
        ranked_names = []
        for i, lab in enumerate(order):
            nm = label2name[lab]
            scores[nm] += (m - 1 - i)                          # Borda
            ranked_names.append(nm)
        rankings.append({"evaluator": evaluator, "order": ranked_names})
    return {"scores": scores, "rankings": rankings}


def _pr_report(state: PeerRankState) -> dict:
    ranking = sorted(state["scores"].items(), key=lambda kv: kv[1], reverse=True)
    lines = ["RANKING DE CONSENSO (Borda):"]
    for pos, (name, pts) in enumerate(ranking, 1):
        lines.append(f"  {pos}. {name}  —  {pts:.0f} pts")
    lines.append(f"\nGANADOR: {ranking[0][0]}")
    return {"report": "\n".join(lines)}


def build_peer_rank():
    g = StateGraph(PeerRankState)
    for name, model in PEERRANK_WORKERS.items():
        nid = "w_" + name.lower()
        g.add_node(nid, _pr_worker(name, model))
        g.add_edge(START, nid)
        g.add_edge(nid, "rank")
    g.add_node("rank", _pr_rank)
    g.add_node("report", _pr_report)
    g.add_edge("rank", "report")
    g.add_edge("report", END)
    return g.compile()


def run_peer_rank(question: str) -> dict:
    return build_peer_rank().invoke({"question": question, "answers": [],
                                     "rankings": [], "scores": {}, "report": ""})


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _cli():
    p = argparse.ArgumentParser(description="Agent Council (MoA + Auditor) para decisiones técnicas.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("moa", help="Mixture-of-Agents (consenso)")
    pm.add_argument("question", nargs="*", help="Pregunta de decisión")
    pa = sub.add_parser("audit", help="Worker + Auditor con retry")
    pa.add_argument("task", nargs="*", help="Tarea a resolver y auditar")
    pp = sub.add_parser("peer-rank", help="Evaluación cruzada (Borda)")
    pp.add_argument("question", nargs="*", help="Pregunta de decisión")
    args = p.parse_args()

    if args.cmd == "moa":
        q = " ".join(args.question).strip() or (
            "¿YOLO11n o RF-DETR-nano para detección en tiempo real en edge sin GPU?")
        print(f"PREGUNTA: {q}\nWorkers: {', '.join(MOA_WORKERS)} | Síntesis: {MOA_SYNTH}\n")
        out = run_moa(q)
        for i, (n, mdl, r) in enumerate(out["responses"]):
            print("=" * 70 + f"\n--- Worker {i+1}: {n} [{mdl}] ---\n{r[:700]}"
                  + ("..." if len(r) > 700 else ""))
        print("\n" + "#" * 70 + "\n### SÍNTESIS FINAL ###\n\n" + out["synthesis"])

    elif args.cmd == "audit":
        t = " ".join(args.task).strip() or (
            "Implementa NMS (non-maximum suppression) vectorizado en numpy para cajas "
            "xyxy con un umbral IoU. Maneja entrada vacía. Incluye 2 asserts.")
        print(f"TAREA: {t}\nWorker: {AUDIT_WORKER} | Auditor: {AUDIT_AUDITOR} | "
              f"max_retries={AUDIT_MAX_RETRIES}\n")
        out = run_auditor(t)
        for s in out["trace"]:
            print("=" * 70 + f"\n--- INTENTO {s['intento']} · WORKER ---\n"
                  + s["draft"][:800] + ("..." if len(s["draft"]) > 800 else ""))
            print(f"\n--- AUDITOR ({'APROBADO' if s['approved'] else 'CAMBIOS'}) ---\n"
                  + s["audit"][:600] + ("..." if len(s["audit"]) > 600 else "") + "\n")
        status = ("APROBADO" if out["approved"]
                  else f"ESCALADO (sin aprobar tras {AUDIT_MAX_RETRIES} reintentos)")
        print("#" * 70 + f"\nRESULTADO FINAL: {status} · intentos: {len(out['trace'])}")

    elif args.cmd == "peer-rank":
        q = " ".join(args.question).strip() or (
            "¿Cuál es la mejor estrategia de data augmentation para entrenar YOLO-nano "
            "en ~2000 imágenes? Justifica.")
        print(f"PREGUNTA: {q}\nWorkers/evaluadores: {', '.join(PEERRANK_WORKERS)}\n")
        out = run_peer_rank(q)
        for n, mdl, ans in out["answers"]:
            print("=" * 70 + f"\n--- {n} [{mdl}] ---\n"
                  + ans[:450] + ("..." if len(ans) > 450 else ""))
        print("\n" + "=" * 70 + "\nVOTOS CRUZADOS (cada evaluador rankea a los demás):")
        for r in out["rankings"]:
            print(f"  {r['evaluator']:9} →  {' > '.join(r['order'])}")
        print("\n" + "#" * 70 + "\n" + out["report"])


if __name__ == "__main__":
    _cli()
