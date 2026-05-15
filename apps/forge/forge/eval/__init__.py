"""Forge eval.

Two surfaces:

* ``runner`` — gold-set CLI (``forge eval``). Runs the bundled tasks
  through a fresh engine and grades each answer with an outcome rubric
  (Pass when score >= 4).
* ``thread_eval`` — per-thread auto-eval. Fires when a chat thread ends;
  scores the outcome **and** the trajectory with separate LLM-judge
  rubrics. Surfaced in the Electron Eval tab.
"""
from .runner import GoldTask, TaskScore, run_eval, run_eval_cli, score_task

__all__ = ["GoldTask", "TaskScore", "run_eval", "run_eval_cli", "score_task"]
