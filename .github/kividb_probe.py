"""Throwaway: measure KiviDB 1.1.2 TAG value encoding (PR #375). Not for merge."""
import string
import sys
import time

import redis

r = redis.Redis(host="127.0.0.1", port=6386, protocol=2, decode_responses=True)
for _ in range(60):
    try:
        r.ping()
        break
    except Exception:
        time.sleep(1)
info = r.execute_command("INFO", "server")
print("kividb_version:", info.get("kividb_version"), "redis_version:", info.get("redis_version"))
r.flushall()
r.execute_command(
    "FT.CREATE", "ix", "ON", "HASH", "PREFIX", "1", "d:", "SCHEMA",
    "t", "TAG", "SEPARATOR", ";", "u", "TAG", "SEPARATOR", ";",
)

PUNCT = list(string.punctuation) + [" "]
values = []
for c in PUNCT:
    values += [f"zz{c}zz", f"zz{c}", f"{c}zz"]
values += [
    "zz", "zzzz", "zza", "zzab", "ab", "a\\b", "C:\\dir", "Cdir", "C:dir",
    'J\\"org Hennig', 'J"org Hennig', "Jorg Hennig", "Rasmus Bj{\\o}rk", "Rasmus Bj{o}rk",
    "Smith, John", "Smith", "John", "Francisco (Paco) Guinea", "Francisco",
    "x|y", "x", "y", "c@d", "c", "d", "a,b", "a", "b", "f(x)", "fx",
    "Ulf-G. Mei{\\ss}ner", "Jos\\'e A. Caballero",
    "sp", " sp", "sp ", "MiXeD", "mixed",
    "\u2019q", "q\u2013r", "\uff08w\uff09", "caf\u00e9", "\uff5cz", "\uff20z", "\uff0az", "\uff04z",
    "*", "$", "zz*zz", "$zz", "zz$",
]
seen = set()
uniq = []
for v in values:
    if v not in seen:
        seen.add(v)
        uniq.append(v)
docs = {}
for i, v in enumerate(uniq):
    r.hset(f"d:{i}", mapping={"t": v, "u": "blue"})
    docs[f"d:{i}"] = v
r.hset("d:other", mapping={"t": "zzother", "u": "red"})
time.sleep(1)
print("docs:", len(docs) + 1, "info num_docs:", dict(zip(*[iter(r.execute_command("FT.INFO", "ix"))] * 2)).get("num_docs"))


def esc(v):
    return "".join(("\\" + c) if (c in string.punctuation or c == " ") else c for c in v)


def q(expr):
    try:
        out = r.execute_command("FT.SEARCH", "ix", expr, "RETURN", "0", "LIMIT", "0", "1000", "DIALECT", "2")
    except Exception as e:  # noqa: BLE001
        return "ERR " + str(e)[:60]
    return sorted(out[1:]) if isinstance(out, list) else out


def expected(v):
    # Exact stored value, case-folded (TAG folds case), `;`-split.
    hits = []
    for k, s in docs.items():
        toks = [t.strip().lower() for t in s.split(";")]
        if v.strip().lower() in toks:
            hits.append(k)
    return sorted(hits)


def run(label, enc, vals):
    bad = 0
    for v in vals:
        e = enc(v)
        want = expected(v)
        res = []
        for form in ("@t:{%s}" % e, "(@t:{%s} | @t:{nomatch}) @u:{blue}" % e, "@u:{blue} @t:{%s}" % e):
            got = q(form)
            res.append(got)
        ok = all(g == want for g in res)
        if not ok:
            bad += 1
        def show(g):
            if isinstance(g, str):
                return g
            if g == want:
                return "exact(%d)" % len(g)
            return "%d %s" % (len(g), [docs.get(k, k) for k in g][:6])
        print(f"{label} {v!r:28} want={len(want)} bare={show(res[0])} | grouped={show(res[1])} | and={show(res[2])}{'' if ok else '   <-- WRONG'}")
    print(f"== {label}: {bad} wrong of {len(vals)}")


probe_vals = [v for v in uniq]
run("ESC", esc, probe_vals)
run("RAW", lambda v: v, probe_vals)
print("lone escaped star:", q("@t:{\\*}"))
print("lone escaped dollar:", q("@t:{\\$}"))
print("escaped trailing star zz\\*:", [docs.get(k, k) for k in q("@t:{zz\\*}")] if isinstance(q("@t:{zz\\*}"), list) else q("@t:{zz\\*}"))
print("escaped leading dollar \\$zz:", q("@t:{\\$zz}"))
print("escaped semicolon zz\\;zz:", q("@t:{zz\\;zz}"))
print("case: @t:{mixed} ->", [docs.get(k, k) for k in q("@t:{mixed}")])
print("ws: @t:{\\ sp} ->", [docs.get(k, k) for k in q("@t:{\\ sp}")] if isinstance(q("@t:{\\ sp}"), list) else q("@t:{\\ sp}"))
print("ws: @t:{sp} ->", [repr(docs.get(k, k)) for k in q("@t:{sp}")])
# Same check in the engine's actual hybrid form (KNN over a vector field).
sys.exit(0)
