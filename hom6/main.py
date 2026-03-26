from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple


def gf2_add_vec(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [(x ^ y) for x, y in zip(a, b)]


def gf2_mat_mul(A: Sequence[Sequence[int]], B: Sequence[Sequence[int]]) -> List[List[int]]:
    # A: m x k, B: k x n
    m = len(A)
    k = len(A[0]) if m else 0
    n = len(B[0]) if B else 0
    res = [[0] * n for _ in range(m)]
    for i in range(m):
        for t in range(k):
            if A[i][t] == 0:
                continue
            for j in range(n):
                res[i][j] ^= (B[t][j] & 1)
    return res


def vec_times_mat_mod2(u: Sequence[int], G: Sequence[Sequence[int]]) -> List[int]:
    # u: 1 x k, G: k x n
    k = len(G)
    n = len(G[0])
    out = [0] * n
    for i in range(k):
        if u[i] == 0:
            continue
        row = G[i]
        for j in range(n):
            out[j] ^= row[j]
    return out


def rank_mod2(A: Sequence[Sequence[int]]) -> int:
    M = [list(map(int, row)) for row in A]
    if not M:
        return 0
    m = len(M)
    n = len(M[0])
    r = 0
    col = 0
    while r < m and col < n:
        pivot = None
        for i in range(r, m):
            if M[i][col] == 1:
                pivot = i
                break
        if pivot is None:
            col += 1
            continue
        M[r], M[pivot] = M[pivot], M[r]
        for i in range(m):
            if i != r and M[i][col] == 1:
                M[i] = [(x ^ y) for x, y in zip(M[i], M[r])]
        r += 1
        col += 1
    return r


def rref_mod2(A: Sequence[Sequence[int]]) -> Tuple[List[List[int]], List[int]]:
    # Row-reduced echelon form over GF(2).
    M = [list(map(int, row)) for row in A]
    if not M:
        return [], []
    m = len(M)
    n = len(M[0])

    pivots: List[int] = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot = None
        for r in range(row, m):
            if M[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        M[row], M[pivot] = M[pivot], M[row]
        # Eliminate all other ones in this pivot column
        for r in range(m):
            if r != row and M[r][col] == 1:
                M[r] = [(x ^ y) for x, y in zip(M[r], M[row])]
        pivots.append(col)
        row += 1
    return M, pivots


def nullspace_basis_mod2(A: Sequence[Sequence[int]]) -> List[List[int]]:
    """
    Basis of the right nullspace: find all v (length n) such that A v^T = 0 over GF(2).
    Returns a list of basis vectors v.
    """
    if not A:
        raise ValueError("Matrix A must be non-empty")
    m = len(A)
    n = len(A[0])
    rref, pivots = rref_mod2(A)
    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]

    basis: List[List[int]] = []
    for free_col in free_cols:
        v = [0] * n
        v[free_col] = 1
        # For each pivot row i with pivot column p_i:
        # x_{p_i} = sum_{free j} rref[i][j] * x_j
        for i, pivot_col in enumerate(pivots):
            v[pivot_col] = rref[i][free_col]  # since only one free variable is 1
        basis.append(v)
    return basis


def vec_to_str(v: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in v)


def mat_to_str(M: Sequence[Sequence[int]]) -> str:
    lines = []
    for row in M:
        lines.append("[" + " ".join(str(int(x)) for x in row) + "]")
    return "\n".join(lines)


def enumerate_codewords(G: Sequence[Sequence[int]]) -> List[Tuple[List[int], List[int]]]:
    k = len(G)
    codewords: List[Tuple[List[int], List[int]]] = []
    for u in product([0, 1], repeat=k):
        u_list = list(map(int, u))
        c = vec_times_mat_mod2(u_list, G)
        codewords.append((u_list, c))
    return codewords


def compute_partial_syndromes_from_codewords(
    H: Sequence[Sequence[int]], codewords: Sequence[List[int]]
) -> Tuple[List[Set[Tuple[int, ...]]], List[Dict[Tuple[int, ...], Set[int]]]]:
    """
    Build syndrome states per level from reachable prefixes.
    state at level t is the syndrome after processing bits 0..t-1.
    """
    r = len(H)
    n = len(H[0])

    # Column vectors h_t for t in [0..n-1]
    H_cols = [[H[row][t] for row in range(r)] for t in range(n)]  # list length n, each length r

    zero_state = (0,) * r
    nodes_by_level: List[Set[Tuple[int, ...]]] = [set() for _ in range(n + 1)]
    codeword_sets_by_level: List[Dict[Tuple[int, ...], Set[int]]] = [
        {} for _ in range(n + 1)
    ]

    for idx, c in enumerate(codewords):
        state = list(zero_state)
        nodes_by_level[0].add(tuple(state))
        codeword_sets_by_level[0].setdefault(tuple(state), set()).add(idx)
        for t in range(n):
            if c[t] == 1:
                state = [(s ^ ht) for s, ht in zip(state, H_cols[t])]
            # else state unchanged
            nodes_by_level[t + 1].add(tuple(state))
            codeword_sets_by_level[t + 1].setdefault(tuple(state), set()).add(idx)

    return nodes_by_level, codeword_sets_by_level


@dataclass(frozen=True)
class Trellis:
    H_name: str
    H: Tuple[Tuple[int, ...], ...]
    n: int
    r: int
    nodes_by_level: List[List[Tuple[int, ...]]]  # sorted node labels
    edges: List[Dict[Tuple[int, ...], Dict[int, Tuple[int, ...]]]]  # level t: state -> {b: next}


def build_trellis_from_parity_check(
    H: Sequence[Sequence[int]], *, name: str, codewords: Sequence[List[int]]
) -> Trellis:
    r = len(H)
    n = len(H[0])
    H_cols = [[H[row][t] for row in range(r)] for t in range(n)]

    nodes_by_level_sets, _ = compute_partial_syndromes_from_codewords(H, codewords)
    nodes_by_level: List[List[Tuple[int, ...]]] = [
        sorted(list(s), key=lambda x: vec_to_str(x)) for s in nodes_by_level_sets
    ]

    edges: List[Dict[Tuple[int, ...], Dict[int, Tuple[int, ...]]]] = [
        {} for _ in range(n)
    ]
    for t in range(n):
        reachable_curr = nodes_by_level_sets[t]
        reachable_next = nodes_by_level_sets[t + 1]
        for s in reachable_curr:
            s_list = list(s)
            next_map: Dict[int, Tuple[int, ...]] = {}
            # b=0: syndrome doesn't change
            if s in reachable_curr:
                if s in reachable_next:
                    next_map[0] = s
            # b=1: add the syndrome contribution of column t
            ns = tuple([(si ^ hi) for si, hi in zip(s_list, H_cols[t])])
            if ns in reachable_next:
                next_map[1] = ns
            # Keep only existing branches
            if next_map:
                edges[t][s] = next_map

    H_tuple = tuple(tuple(int(x) for x in row) for row in H)
    return Trellis(
        H_name=name,
        H=H_tuple,
        n=n,
        r=r,
        nodes_by_level=nodes_by_level,
        edges=edges,
    )


def build_syndrome_trellis_from_codewords(
    H: Sequence[Sequence[int]],
    *,
    name: str,
    codewords: Sequence[List[int]],
) -> Trellis:
    """
    Direct construction of the syndrome trellis:
    - compute partial syndromes for each prefix by multiplying submatrix H_prefix;
    - build edges by reading which next syndrome each codeword induces at each position.
    """
    r = len(H)
    n = len(H[0])

    # For each codeword we store syndrome at each level.
    syndromes: List[List[Tuple[int, ...]]] = []
    nodes_by_level_sets: List[Set[Tuple[int, ...]]] = [set() for _ in range(n + 1)]

    for c in codewords:
        level_states: List[Tuple[int, ...]] = []
        # t=0: empty prefix => zero syndrome
        state0 = (0,) * r
        level_states.append(state0)
        nodes_by_level_sets[0].add(state0)

        for t in range(1, n + 1):
            prefix = c[:t]
            s_list: List[int] = []
            for row in range(r):
                acc = 0
                for j in range(t):
                    acc ^= (H[row][j] & 1) & (prefix[j] & 1)
                s_list.append(acc)
            st = tuple(s_list)
            level_states.append(st)
            nodes_by_level_sets[t].add(st)
        syndromes.append(level_states)

    # Build edges by reading transitions induced by codewords.
    edges: List[Dict[Tuple[int, ...], Dict[int, Tuple[int, ...]]]] = [{} for _ in range(n)]
    for cw_idx, c in enumerate(codewords):
        for t in range(n):
            s = syndromes[cw_idx][t]
            ns = syndromes[cw_idx][t + 1]
            b = c[t]
            edges[t].setdefault(s, {})
            if b in edges[t][s] and edges[t][s][b] != ns:
                raise RuntimeError(
                    "Inconsistent trellis construction: different codewords induce different next states "
                    f"(level t={t}, state={vec_to_str(s)}, b={b})"
                )
            edges[t][s][b] = ns

    nodes_by_level: List[List[Tuple[int, ...]]] = [
        sorted(list(s), key=lambda x: vec_to_str(x)) for s in nodes_by_level_sets
    ]

    H_tuple = tuple(tuple(int(x) for x in row) for row in H)
    return Trellis(
        H_name=name,
        H=H_tuple,
        n=n,
        r=r,
        nodes_by_level=nodes_by_level,
        edges=edges,
    )


def trellis_edges_to_list(T: Trellis) -> List[Tuple[int, Tuple[int, ...], int, Tuple[int, ...]]]:
    out: List[Tuple[int, Tuple[int, ...], int, Tuple[int, ...]]] = []
    for t in range(T.n):
        for s, m in T.edges[t].items():
            for b, ns in m.items():
                out.append((t, s, b, ns))
    return sorted(out, key=lambda x: (x[0], vec_to_str(x[1]), x[2], vec_to_str(x[3])))


def build_state_partition_by_codewords(
    H: Sequence[Sequence[int]], codewords: Sequence[List[int]]
) -> List[Dict[Tuple[int, ...], FrozenSet[int]]]:
    nodes_by_level_sets, codeword_sets_by_level = compute_partial_syndromes_from_codewords(H, codewords)
    partition: List[Dict[Tuple[int, ...], FrozenSet[int]]] = []
    for level_map in codeword_sets_by_level:
        partition.append({state: frozenset(idxs) for state, idxs in level_map.items()})
    return partition


def compare_trellises_up_to_node_relabeling(
    T1: Trellis, T2: Trellis, codewords: Sequence[List[int]]
) -> Tuple[bool, str]:
    if T1.n != T2.n:
        return False, "Different code lengths"

    # Partition states by the set of codewords passing through them.
    # That gives an unambiguous correspondence between nodes up to renumbering.
    P1 = build_state_partition_by_codewords(T1.H, codewords)
    P2 = build_state_partition_by_codewords(T2.H, codewords)

    for t in range(T1.n + 1):
        states1 = set(P1[t].keys())
        states2 = set(P2[t].keys())

        sets1 = sorted([tuple(sorted(P1[t][s])) for s in states1])
        sets2 = sorted([tuple(sorted(P2[t][s])) for s in states2])
        if sets1 != sets2:
            return False, f"State partitions differ at level t={t}"

    # Build mapping from T1 states to T2 states per level using identical codeword sets.
    mapping: List[Dict[Tuple[int, ...], Tuple[int, ...]]] = [dict() for _ in range(T1.n + 1)]
    for t in range(T1.n + 1):
        for s1, cwset1 in P1[t].items():
            matched = None
            for s2, cwset2 in P2[t].items():
                if cwset1 == cwset2:
                    matched = s2
                    break
            if matched is None:
                return False, f"No node match at level t={t}"
            mapping[t][s1] = matched

    # Check all edges.
    for t in range(T1.n):
        # Compare branch structure induced by codeword-partition mapping.
        for s1, out_map in T1.edges[t].items():
            s2 = mapping[t][s1]
            for b, ns1 in out_map.items():
                ns2_expected = mapping[t + 1][ns1]
                out_map2 = T2.edges[t].get(s2, {})
                if b not in out_map2:
                    return (
                        False,
                        f"Missing edge in T2: level t={t}, state1={vec_to_str(s1)}, b={b}",
                    )
                if out_map2[b] != ns2_expected:
                    return False, (
                        f"Edge mismatch at level t={t}: "
                        f"b={b}, T1 {vec_to_str(s1)}->{vec_to_str(ns1)} "
                        f"but T2 expects {vec_to_str(s2)}->{vec_to_str(ns2_expected)}"
                    )

    return True, "Требуемое совпадение выполнено (с точностью до переупорядочивания узлов на уровнях)."


def print_trellis(T: Trellis, codewords: Sequence[List[int]]) -> None:
    print(f"\n===== Решетка: {T.H_name} =====")
    print(f"H (размер {T.r}x{T.n}):")
    for row in T.H:
        print("  ", " ".join(str(x) for x in row))

    # Build partitions for printing.
    partitions = build_state_partition_by_codewords(T.H, codewords)

    for t in range(T.n + 1):
        print(f"\nУровень t={t}, число узлов: {len(partitions[t])}")
        # Print nodes in sorted order to make output stable.
        level_nodes_sorted = sorted(list(partitions[t].keys()), key=lambda x: vec_to_str(x))
        for node in level_nodes_sorted:
            cwset = sorted(list(partitions[t][node]))
            print(f"  node {vec_to_str(node)}: codewords={cwset}")

        if t < T.n:
            # Print outgoing edges.
            print("  Рёбра:")
            for node in level_nodes_sorted:
                out_map = T.edges[t].get(node, {})
                if not out_map:
                    print(f"    node {vec_to_str(node)}: (нет выходов)")
                else:
                    for b in sorted(out_map.keys()):
                        print(
                            f"    {vec_to_str(node)} --(b={b})-> {vec_to_str(out_map[b])}"
                        )


def main() -> None:
    import os
    import builtins

    # Given generator matrix G (k x n) over GF(2)
    G = [
        [1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0],
    ]

    out_path = os.path.join(os.path.dirname(__file__), "out.txt")
    log_f = open(out_path, "w", encoding="utf-8")

    stdout_print = builtins.print

    def log_print(*args, **kwargs) -> None:
        # Mirror stdout to a file.
        stdout_print(*args, **kwargs)
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(a) for a in args)
        log_f.write(text + end)
        log_f.flush()

    # From here on, use log_print instead of print.

    k = len(G)
    n = len(G[0])

    rG = rank_mod2(G)
    log_print("G (k x n) =")
    log_print(mat_to_str(G))
    log_print(f"rank(G) = {rG}, k={k}, n={n}")

    # Build H as a basis of right nullspace of G: G * H^T = 0
    H_basis = nullspace_basis_mod2(G)
    r = len(H_basis)
    H = H_basis  # rows of H

    log_print("\nПроверочная матрица H (как базис nullspace(G)):")
    log_print(mat_to_str(H))
    log_print(f"размер H: {r} x {n}")

    # Verify orthogonality: G * H^T = 0
    Ht = list(map(list, zip(*H)))  # n x r
    GHt = gf2_mat_mul(G, Ht)  # k x r
    log_print("\nПроверка G * H^T mod 2 (должно быть нулевой матрицей):")
    log_print(mat_to_str(GHt))

    # Enumerate all codewords
    u_and_c = enumerate_codewords(G)
    # Only unique codewords (for safety)
    seen: Set[Tuple[int, ...]] = set()
    codewords: List[List[int]] = []
    for u, c in u_and_c:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            codewords.append(c)

    log_print(f"\nВсего различных кодовых слов: {len(codewords)}")
    for i, c in enumerate(codewords):
        log_print(f"c[{i}] = {c}")

    # Trellis 1: "По проверочной матрице" (use H directly)
    T1 = build_trellis_from_parity_check(H, name="Т1: решетка по H", codewords=codewords)

    # Trellis 2: "Синдромная решетка"
    T2 = build_syndrome_trellis_from_codewords(
        H, name="Т2: синдромная решетка (из перечисления кодовых слов)", codewords=codewords
    )

    # For trellis printing, keep using normal print, but mirror it to the file by temporarily
    # overriding builtins.print.
    original_print = builtins.print
    try:
        builtins.print = log_print  # type: ignore[assignment]
        print_trellis(T1, codewords)
        print_trellis(T2, codewords)

        ok, reason = compare_trellises_up_to_node_relabeling(T1, T2, codewords)
        log_print("\n===== Сравнение решёток =====")
        log_print("Совпадение:", ok)
        log_print("Причина:", reason)
    finally:
        builtins.print = original_print  # type: ignore[assignment]

    log_f.close()


if __name__ == "__main__":
    main()

