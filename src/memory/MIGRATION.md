# Memory module migration notes

## Semantic memory: triples → natural language

Older versions of this codebase represented semantic memory as
`(subject, predicate, object, confidence)` triples persisted to SQLite,
written automatically by an LLM extractor on every turn. Two problems:

1. **Garbage in.** The auto-extractor pulled triples out of small talk,
   pricing claims the assistant invented, and other low-signal turns. The
   `_is_small_talk_turn` heuristic was a band-aid.
2. **Wrong abstraction.** The agent never thinks in triples. Forcing it to
   read back `("Acme Corp", "headquartered_in", "Berlin")` rows costs tokens
   and loses nuance.

| Aspect | Before | After |
|---|---|---|
| Storage | SQLite `facts` table | Chroma collection of NL strings |
| Schema | `(subject, predicate, object, confidence)` | `(text, thread_id, created_at)` |
| Writes | Automatic per-turn LLM extractor | Explicit `semantic_write` tool call |
| Retrieval | `query(subject=...)` exact match | `search(query, k, min_score?)` similarity |
| Failure mode | "garbage in memory" | "agent forgot to write" |

### Code-level mapping

| Old call | New call |
|---|---|
| `SemanticMemory(path/"semantic.sqlite")` | `SemanticMemory(path/"semantic_chroma")` |
| `semantic.write_from_turn(...)` | _Removed._ Bind `semantic_write` tool to the agent and let it decide. |
| `semantic.upsert(SemanticFact(subject=..., predicate=..., object=...))` | `semantic.write("the user prefers X")` |
| `semantic.query(subject="Acme Corp")` | `semantic.search("Acme Corp", k=5)` |
| `f.subject`, `f.predicate`, `f.object` | `r.text` |
| `len(semantic.all())` | `semantic.count()` |

The old `semantic.sqlite` is **not** read by the new code. If you have one
lying around it is safe to delete; nothing migrates automatically.

## Reflection: outcome field removed

Earlier versions of `reflect_on_thread` returned an `outcome` string
(`success` / `failure` / `unknown` plus whatever else the LLM editorialized).
Nothing downstream actually used it for filtering or ranking — `score`
already serves the "should I trust this learning" role on both episodic
entries and procedural skills. The field has been removed everywhere
(reflection schema, episodic entry, Forge engine events, monitor renderer,
notebook prints).

## Week 2 → Week 3 boundary

DSPy MIPROv2 optimization of the old triple extractor moved to Week 3.
Triple-F1 was the wrong objective for natural-language memory; Week 3 will
reintroduce DSPy with metrics aligned to NL memory quality and retrieval.
