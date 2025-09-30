# opt_type_scale.py
# -----------------------------------------------------------------------------
# Tier-4 IDE Upgrade:
#   - Strong typing & docstrings
#   - Objective split with proper λ weighting
#   - Adam / GD optimizers, gradient-norm early stop, clipping
#   - Deterministic RNG and reproducible init
#   - Two calibration modes: 'topo' (sklearn isotonic along topo order),
#                           'proj' (iterative isotonic projection on pairs)
#   - Robust DAG build (weak-edge breaking) kept from original
#   - Rich diagnostics and CLI + optional plotting
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np
from collections import defaultdict, deque
from sklearn.isotonic import IsotonicRegression

try:  # pragma: no cover - optional dependency when used standalone
    from scales import SynonymGraph
except Exception:  # pragma: no cover
    SynonymGraph = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from scales import SynonymGraph as SynonymGraphType

# -----------------------------------------------------------------------------#
# Logging
# -----------------------------------------------------------------------------#
logger = logging.getLogger("opt_type_scale")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

Pair = Tuple[str, str, float]


# -----------------------------------------------------------------------------#
# Config
# -----------------------------------------------------------------------------#
@dataclass
class ScaleConfig:
    """Scale definition (label space + anchors)."""
    name: str
    anchors_ordered: List[str]
    y_min: float = -1.0
    y_max: float = 1.0


@dataclass
class OptConfig:
    """Optimization and regularization hyperparameters."""
    margin: float = 0.2                  # hinge margin
    lr: float = 0.05                     # learning rate (GD/Adam)
    steps: int = 4000                    # max iterations
    l2: float = 1e-4                     # L2 regularization
    lambda_graph: float = 1.0            # syn/ant weighting
    lambda_embed: float = 0.5            # anchor similarity prior weighting
    seed_weight: float = 4.0             # multiplier on seed pairs
    clip: float = 5.0                    # grad clip (L2)
    verbose_every: int = 500             # console progress (0 disables)
    rng: int = 7                         # reproducible RNG seed

    # Tier-4 additions:
    optimizer: str = "adam"              # {"adam","gd"}
    beta1: float = 0.9                   # Adam β1
    beta2: float = 0.999                 # Adam β2
    eps: float = 1e-8                    # Adam ε

    # Early stopping:
    early_stop_patience: int = 200       # min steps between improvements
    early_stop_tol: float = 1e-6         # relative objective tolerance


@dataclass
class CalibConfig:
    """Calibration options after raw optimization."""
    method: str = "topo"                 # {"topo","proj","none"}
    proj_max_iter: int = 2000            # for 'proj' method
    proj_tol: float = 1e-9               # stop when max violation < tol


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _vocab_from(
    anchors: Iterable[str],
    pairs: Iterable[Pair],
    syn_edges: Iterable[Pair],
    ant_edges: Iterable[Pair],
    anchor_sims: Optional[Dict[str, Dict[str, float]]],
) -> List[str]:
    V: Set[str] = set(anchors)
    for a, b, _ in pairs:
        V.add(a)
        V.add(b)
    for a, b, _ in syn_edges:
        V.add(a)
        V.add(b)
    for a, b, _ in ant_edges:
        V.add(a)
        V.add(b)
    if anchor_sims:
        V.update(anchor_sims.keys())
        for sims in anchor_sims.values():
            V.update(sims.keys())
    return sorted(V)


def _init_scores(anchors_ordered: List[str], vocab: List[str], y_min: float, y_max: float) -> np.ndarray:
    """
    Initialize by placing anchors linearly on [y_min,y_max] and others near 0.
    """
    init = {w: 0.0 for w in vocab}
    if anchors_ordered:
        if len(anchors_ordered) == 1:
            init[anchors_ordered[0]] = 0.0
        else:
            for i, w in enumerate(anchors_ordered):
                init[w] = y_min + i * (y_max - y_min) / (len(anchors_ordered) - 1)
    return np.array([init[w] for w in vocab], dtype=np.float32)


def _index_map(vocab: List[str]) -> Dict[str, int]:
    return {w: i for i, w in enumerate(vocab)}


def _build_topo_from_pairs(pairs: List[Pair]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Build a DAG (best-effort) from (a<b). If cycles appear, break lightest edges.
    Returns a topological order (or approximate) and adjacency (for diagnostics).
    """
    adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    indeg: Dict[str, int] = defaultdict(int)
    nodes: Set[str] = set()
    for a, b, w in pairs:
        nodes.add(a)
        nodes.add(b)
        adj[a].append((b, w))
        indeg[b] += 1
        indeg.setdefault(a, 0)

    order: List[str] = []
    q: deque[str] = deque([n for n in nodes if indeg[n] == 0])

    indeg_c = dict(indeg)
    adj_c: Dict[str, List[Tuple[str, float]]] = {u: list(vs) for u, vs in adj.items()}

    while q:
        u = q.popleft()
        order.append(u)
        for v, _ in adj_c.get(u, []):
            indeg_c[v] -= 1
            if indeg_c[v] == 0:
                q.append(v)

    if len(order) < len(nodes):
        # cycles exist → remove a fraction of smallest-weight edges among remaining nodes, then recurse
        remaining = [n for n in nodes if n not in order]
        edges: List[Tuple[str, str, float]] = []
        for u, vs in adj_c.items():
            for v, w in vs:
                if u in remaining and v in remaining:
                    edges.append((u, v, w))
        if edges:
            edges.sort(key=lambda x: x[2])
            to_remove = edges[: max(1, len(edges) // 10)]
            remove_set = {(u, v) for u, v, _ in to_remove}
            new_pairs = [(a, b, w) for (a, b, w) in pairs if (a, b) not in remove_set]
            return _build_topo_from_pairs(new_pairs)

    adj_simple: Dict[str, List[str]] = {u: [v for v, _ in vs] for u, vs in adj.items()}
    return order, adj_simple


def _objective_components(
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Wp: np.ndarray,
    is_seed: np.ndarray,
    margin: float,
    S_u: np.ndarray,
    S_v: np.ndarray,
    S_w: np.ndarray,
    T_u: np.ndarray,
    T_v: np.ndarray,
    T_w: np.ndarray,
    has_embed: bool,
    P: Optional[np.ndarray],
    t_anchor: Optional[np.ndarray],
    l2: float,
) -> Tuple[float, float, float, float, float]:
    """Compute scalar losses (rank, syn, ant, emb, l2)."""
    # Ranking hinge
    L_rank = 0.0
    if len(A):
        diff = x[B] - x[A]
        viol = np.maximum(0.0, margin - diff)
        w = Wp * (1.0 + is_seed * (opt.seed_weight - 1.0))  # opt captured later; will be passed correctly
        L_rank = float(np.sum(w * viol))

    # Syn smoothness
    L_syn = float(np.sum(S_w * (x[S_u] - x[S_v]) ** 2)) if len(S_u) else 0.0

    # Ant "anti-smoothness"
    L_ant = float(np.sum(T_w * (x[T_u] + x[T_v]) ** 2)) if len(T_u) else 0.0

    # Embedding prior
    L_emb = 0.0
    if has_embed and P is not None and t_anchor is not None:
        target = P @ t_anchor
        L_emb = float(np.sum((x - target) ** 2))

    # L2
    L_l2 = float(l2 * np.sum(x * x))

    return L_rank, L_syn, L_ant, L_emb, L_l2


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#
def fit_type_scale(
    scale: ScaleConfig,
    pairwise: Optional[List[Pair]] = None,
    syn_edges: Optional[List[Pair]] = None,
    ant_edges: Optional[List[Pair]] = None,
    anchor_sims: Optional[Dict[str, Dict[str, float]]] = None,
    opt: OptConfig = OptConfig(),
    calib: CalibConfig = CalibConfig(),
    graph: Optional["SynonymGraphType"] = None,
):
    """
    Learn scalar scores for a lexical type scale.

    Args:
        scale:     scale definition & anchors
    pairwise:  constraints (a<b) with weights
    syn_edges: synonym edges (u,v,w) encouraging closeness
    ant_edges: antonym edges (u,v,w) encouraging opposition
    graph:     optional SynonymGraph; edges are merged with syn_edges/ant_edges
        anchor_sims: optional {word: {anchor: sim in [0,1]}}
        opt:       optimization hyperparameters
        calib:     post-calibration strategy

    Returns:
        scores: dict[word] -> calibrated score in [y_min,y_max]
        raw:    dict[word] -> pre-calibration score
        info:   diagnostics dict
    """
    t0 = time.time()
    rng = np.random.default_rng(opt.rng)

    pairwise = list(pairwise or [])
    syn_edges = list(syn_edges or [])
    ant_edges = list(ant_edges or [])
    if graph is not None:
        if getattr(graph, "syn", None):
            syn_edges.extend((edge.u, edge.v, edge.w) for edge in graph.syn)
        if getattr(graph, "ant", None):
            ant_edges.extend((edge.u, edge.v, edge.w) for edge in graph.ant)

    # -------- Vocabulary, indices, init
    vocab = _vocab_from(scale.anchors_ordered, pairwise, syn_edges, ant_edges, anchor_sims)
    idx = _index_map(vocab)
    x = _init_scores(scale.anchors_ordered, vocab, scale.y_min, scale.y_max)

    # -------- Ranking arrays
    A = np.array([idx[a] for a, _, _ in pairwise], dtype=np.int32) if pairwise else np.zeros((0,), np.int32)
    B = np.array([idx[b] for _, b, _ in pairwise], dtype=np.int32) if pairwise else np.zeros((0,), np.int32)
    Wp = np.array([w for _, _, w in pairwise], dtype=np.float32) if pairwise else np.zeros((0,), np.float32)

    seed_pairs = {(scale.anchors_ordered[i], scale.anchors_ordered[i + 1])
                  for i in range(max(0, len(scale.anchors_ordered) - 1))}
    is_seed = np.array(
        [1.0 if (vocab[a], vocab[b]) in seed_pairs else 0.0 for a, b in zip(A, B)],
        dtype=np.float32,
    ) if pairwise else np.zeros((0,), np.float32)

    # -------- Syn/Ant arrays
    if syn_edges:
        S_u = np.array([idx[a] for a, _, _ in syn_edges], dtype=np.int32)
        S_v = np.array([idx[b] for _, b, _ in syn_edges], dtype=np.int32)
        S_w = np.array([w for _, _, w in syn_edges], dtype=np.float32)
    else:
        S_u = S_v = np.zeros((0,), np.int32)
        S_w = np.zeros((0,), np.float32)

    if ant_edges:
        T_u = np.array([idx[a] for a, _, _ in ant_edges], dtype=np.int32)
        T_v = np.array([idx[b] for _, b, _ in ant_edges], dtype=np.int32)
        T_w = np.array([w for _, _, w in ant_edges], dtype=np.float32)
    else:
        T_u = T_v = np.zeros((0,), np.int32)
        T_w = np.zeros((0,), np.float32)

    # -------- Embedding prior
    has_embed = bool(anchor_sims)
    P: Optional[np.ndarray] = None
    t_anchor: Optional[np.ndarray] = None
    if has_embed:
        anchors = [a for a in scale.anchors_ordered if a in idx]
        if anchors:
            A_idx = np.array([idx[a] for a in anchors], dtype=np.int32)  # noqa: F841  (kept for clarity)
            P = np.zeros((len(vocab), len(anchors)), dtype=np.float32)
            for w, sims in (anchor_sims or {}).items():
                if w not in idx:
                    continue
                row = idx[w]
                vec = np.array([float(max(0.0, sims.get(a, 0.0))) for a in anchors], dtype=np.float32)
                s = float(np.sum(vec))
                if s > 0.0:
                    P[row, :] = vec / s
            t_anchor = np.array(
                [scale.y_min + i * (scale.y_max - scale.y_min) / max(1, len(anchors) - 1)
                 for i, _ in enumerate(anchors)],
                dtype=np.float32,
            )
        else:
            has_embed = False

    # -------- Optimizer (Adam or GD)
    m = np.zeros_like(x)  # Adam m
    v = np.zeros_like(x)  # Adam v
    beta1, beta2, eps = opt.beta1, opt.beta2, opt.eps

    def step_update(grad: np.ndarray, t: int):
        nonlocal x, m, v
        if opt.optimizer.lower() == "adam":
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x = x - opt.lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            x = x - opt.lr * grad

    # -------- Training loop
    best_obj = np.inf
    last_improve_step = 0

    def compute_grads(x_now: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Return full gradient with proper λ-weighting + breakdown."""
        grad_rank = np.zeros_like(x_now)
        grad_syn = np.zeros_like(x_now)
        grad_ant = np.zeros_like(x_now)
        grad_emb = np.zeros_like(x_now)
        grad_l2 = 2.0 * opt.l2 * x_now

        # Ranking hinge
        L_rank = 0.0
        if len(A):
            diff = x_now[B] - x_now[A]
            viol = (opt.margin - diff)
            mask = (viol > 0).astype(np.float32)
            w = Wp * (1.0 + is_seed * (opt.seed_weight - 1.0))
            g = w * mask
            np.add.at(grad_rank, A, +g)   # d/dxA += w
            np.add.at(grad_rank, B, -g)   # d/dxB -= w
            L_rank = float(np.sum(w * np.maximum(0.0, viol)))

        # Syn
        L_syn = 0.0
        if len(S_u):
            d = (x_now[S_u] - x_now[S_v])
            g = 2.0 * S_w * d
            np.add.at(grad_syn, S_u, +g)
            np.add.at(grad_syn, S_v, -g)
            L_syn = float(np.sum(S_w * (d ** 2)))

        # Ant
        L_ant = 0.0
        if len(T_u):
            d = (x_now[T_u] + x_now[T_v])
            g = 2.0 * T_w * d
            np.add.at(grad_ant, T_u, +g)
            np.add.at(grad_ant, T_v, +g)
            L_ant = float(np.sum(T_w * (d ** 2)))

        # Embedding
        L_emb = 0.0
        if has_embed and P is not None and t_anchor is not None:
            target = P @ t_anchor
            diff = (x_now - target)
            grad_emb = 2.0 * diff
            L_emb = float(np.sum(diff * diff))

        # Combine with λ
        grad = grad_rank + opt.lambda_graph * (grad_syn + grad_ant) + opt.lambda_embed * grad_emb + grad_l2
        L_total = L_rank + opt.lambda_graph * (L_syn + L_ant) + opt.lambda_embed * L_emb + float(opt.l2 * np.sum(x_now * x_now))

        return grad, {
            "rank": L_rank,
            "syn": L_syn,
            "ant": L_ant,
            "emb": L_emb,
            "l2": float(opt.l2 * np.sum(x_now * x_now)),
            "total": L_total,
        }

    for step in range(1, opt.steps + 1):
        grad, parts = compute_grads(x)

        # Clip
        gnorm = float(np.linalg.norm(grad) + 1e-12)
        if gnorm > opt.clip:
            grad = grad * (opt.clip / gnorm)

        # Update
        step_update(grad, t=step)

        # Small jitter to escape flats
        if (step % 97) == 0:
            x += rng.normal(0, 1e-4, size=x.shape).astype(np.float32)

        # Early stop logic
        obj = parts["total"]
        if obj + opt.early_stop_tol < best_obj:
            best_obj = obj
            last_improve_step = step
        elif step - last_improve_step >= opt.early_stop_patience:
            logger.info(f"[{scale.name}] early stop at step {step} (no improvement {opt.early_stop_patience} steps).")
            break

        # Verbose
        if opt.verbose_every and (step % opt.verbose_every == 0 or step == 1):
            logger.info(
                f"[{scale.name}] step {step:4d} | total={parts['total']:.4f} "
                f"(rank={parts['rank']:.3f}, syn={parts['syn']:.3f}, ant={parts['ant']:.3f}, emb={parts['emb']:.3f}, l2={parts['l2']:.3f}) "
                f"| ||g||={gnorm:.3f}"
            )

    # -------------------- Post Calibration --------------------
    raw_scores = {w: float(v) for w, v in zip(vocab, x)}
    scores = raw_scores.copy()

    # Ensure seed edges are present for calibration ordering
    all_pairs = list(pairwise)
    existing = {(a, b) for a, b, _ in all_pairs}
    for i in range(len(scale.anchors_ordered) - 1):
        a, b = scale.anchors_ordered[i], scale.anchors_ordered[i + 1]
        if (a, b) not in existing:
            all_pairs.append((a, b, 1.0))

    if calib.method.lower() == "topo":
        topo, _ = _build_topo_from_pairs(all_pairs)
        if topo:
            pos = {w: i for i, w in enumerate(topo)}
            xs = np.array([pos.get(w, -1) for w in vocab], dtype=np.float32)
            keep = xs >= 0
            if np.sum(keep) >= 2 and len(np.unique(xs[keep])) >= 2:
                order_idx = np.argsort(xs[keep])
                x_axis = xs[keep][order_idx]
                y_sorted = x[keep][order_idx]
                ir = IsotonicRegression(y_min=scale.y_min, y_max=scale.y_max, increasing=True, out_of_bounds="clip")
                y_cal = ir.fit_transform(x_axis, y_sorted)
                x_fit = x.copy()
                x_fit[keep][order_idx] = y_cal
                scores = {w: float(v) for w, v in zip(vocab, x_fit)}
    elif calib.method.lower() == "proj":
        # Iterative isotonic projection over pairwise constraints: for each (a<b)
        # if s[a] > s[b], set s[a]=s[b]=(s[a]+s[b])/2. Repeat until no violations.
        s = x.copy()
        pair_idx = [(idx[a], idx[b], float(w)) for a, b, w in all_pairs if a in idx and b in idx]
        if pair_idx:
            for _ in range(calib.proj_max_iter):
                max_violation = 0.0
                # Sweep high to low weights first to respect strong constraints
                for ia, ib, w in sorted(pair_idx, key=lambda t: -t[2]):
                    if s[ia] > s[ib]:
                        avg = 0.5 * (s[ia] + s[ib])
                        s[ia] = avg
                        s[ib] = avg
                        max_violation = max(max_violation, float(s[ia] - s[ib]))
                if max_violation <= calib.proj_tol:
                    break
            # Clip to bounds
            s = np.clip(s, scale.y_min, scale.y_max)
            scores = {w: float(v) for w, v in zip(vocab, s)}
    else:
        # 'none' -> keep raw
        pass

    # -------------------- Diagnostics --------------------
    def count_violations(sc: MutableMapping[str, float]) -> int:
        cnt = 0
        for a, b, _ in pairwise:
            if sc.get(a, -np.inf) > sc.get(b, np.inf) + 1e-9:
                cnt += 1
        return cnt

    pre_v = count_violations(raw_scores)
    post_v = count_violations(scores)

    runtime = time.time() - t0
    info = {
        "scale": asdict(scale),
        "opt": {**asdict(opt), "optimizer": opt.optimizer},
        "calib": asdict(calib),
        "vocab_size": len(vocab),
        "pairs": len(pairwise),
        "syn_edges": len(syn_edges),
        "ant_edges": len(ant_edges),
        "violations_pre": pre_v,
        "violations_post": post_v,
        "runtime_sec": round(runtime, 4),
        "best_objective": round(float(best_obj), 6),
    }

    return scores, raw_scores, info


# -----------------------------------------------------------------------------#
# Pretty print / utils
# -----------------------------------------------------------------------------#
def explain(scores: Dict[str, float], topn: int = 25) -> str:
    """Return a human-readable ladder of scores."""
    items = sorted(scores.items(), key=lambda kv: kv[1])
    lines = [f"{w:20s} {v:+.3f}" for w, v in items[:topn]] + (["..."] if len(items) > topn else [])
    return "\n".join(lines)


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def _cli():
    ap = argparse.ArgumentParser(description="Type-level scale optimizer (Tier-4).")
    ap.add_argument("--name", type=str, default="temperature", help="Scale name")
    ap.add_argument("--anchors", type=str, nargs="+",
                    default=["freezing", "cold", "cool", "lukewarm", "warm", "hot", "boiling"])
    ap.add_argument("--pairs_json", type=str, default="", help='JSON file: [["a","b",w],...]')
    ap.add_argument("--syn_json", type=str, default="", help='JSON file: [["u","v",w],...]')
    ap.add_argument("--ant_json", type=str, default="", help='JSON file: [["u","v",w],...]')
    ap.add_argument("--anchor_sims_json", type=str, default="", help='JSON file: {word:{anchor:sim}}')
    ap.add_argument("--calib", type=str, default="topo", choices=["topo", "proj", "none"])
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "gd"])
    ap.add_argument("--verbose_every", type=int, default=800)
    ap.add_argument("--dump", type=str, default="", help="Write scores JSON to this path")
    args = ap.parse_args()

    # Load inputs
    def _load_list3(p: str) -> List[Pair]:
        if not p:
            return []
        with open(p, "r", encoding="utf-8") as f:
            L = json.load(f)
        return [(str(a), str(b), float(w)) for a, b, w in L]

    def _load_dict(p: str) -> Dict[str, Dict[str, float]]:
        if not p:
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return {k: {kk: float(vv) for kk, vv in v.items()} for k, v in json.load(f).items()}

    pairwise = _load_list3(args.pairs_json)
    syn = _load_list3(args.syn_json)
    ant = _load_list3(args.ant_json)
    anchor_sims = _load_dict(args.anchor_sims_json) if args.anchor_sims_json else None

    scale = ScaleConfig(name=args.name, anchors_ordered=args.anchors)
    opt = OptConfig(steps=args.steps, lr=args.lr, optimizer=args.optimizer, verbose_every=args.verbose_every)
    calib = CalibConfig(method=args.calib)

    scores, raw, info = fit_type_scale(scale, pairwise, syn, ant, anchor_sims, opt=opt, calib=calib)

    print("\n== Calibrated scores ==")
    print(explain(scores, topn=50))
    print("\ninfo:", json.dumps(info, indent=2))

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"scores": scores, "raw": raw, "info": info}, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.dump}")


# -----------------------------------------------------------------------------#
# Demo (unchanged spirit, tuned for Tier-4)
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    # If called with args, use CLI; else run a small demo
    import sys
    if len(sys.argv) > 1:
        _cli()
    else:
        scale = ScaleConfig(
            name="temperature",
            anchors_ordered=["freezing", "cold", "cool", "lukewarm", "warm", "hot", "boiling"],
            y_min=-1.0, y_max=1.0,
        )

        pairwise = [
            ("cool", "warm", 1.0),
            ("warm", "hot", 1.0),
            ("hot", "boiling", 1.0),
            ("cold", "cool", 1.0),
            ("lukewarm", "warm", 0.5),
            ("freezing", "cold", 0.8),
        ]

        syn = [("hot", "boiling", 1.0), ("cold", "freezing", 1.0), ("cool", "chilly", 0.8)]
        ant = [("hot", "cold", 1.0), ("boiling", "freezing", 1.0)]

        anchor_sims = {
            "toasty": {"warm": 0.7, "hot": 0.4},
            "scalding": {"hot": 0.6, "boiling": 0.9},
            "chilly": {"cool": 0.9, "cold": 0.6},
            "tepid": {"lukewarm": 0.8, "warm": 0.3},
        }

        opt = OptConfig(verbose_every=400, steps=2400, optimizer="adam", lr=0.05)
        calib = CalibConfig(method="topo")

        scores, raw, info = fit_type_scale(scale, pairwise, syn, ant, anchor_sims, opt=opt, calib=calib)

        print("\n== Calibrated scores ==")
        print(explain(scores, topn=999))
        print("\ninfo:", json.dumps(info, indent=2))
