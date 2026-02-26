from itertools import product


def matmul_mod2(u, G):
    """
    Умножение вектора u (1×k) на матрицу G (k×n) по модулю 2.

    u: список длины k
    G: список из k строк, каждая длины n
    """
    n = len(G[0])
    res = [0] * n
    for i, ui in enumerate(u):
        if ui == 0:
            continue
        row = G[i]
        for j in range(n):
            res[j] ^= row[j]
    return res


def weight(v):
    """Вес двоичного вектора (число единиц)."""
    return sum(v)


def hamming_distance(x, y):
    """Расстояние Хэмминга между двумя двоичными векторами одинаковой длины."""
    return sum((xi ^ yi) for xi, yi in zip(x, y))


def main():
    # Порождающая матрица расширенного кода Хэмминга (8,4)
    # Взята в виде, соответствующем изложению в LaTeX
    G8 = [
        [1, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 0],
    ]

    # Генерируем все кодовые слова
    codewords = []
    for u in product([0, 1], repeat=4):
        c = matmul_mod2(list(u), G8)
        codewords.append(c)

    # Убираем возможные дубликаты (на всякий случай)
    unique = []
    seen = set()
    for c in codewords:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique.append(c)

    print("Всего различных кодовых слов:", len(unique))
    print("Кодовые слова и их веса:")
    for idx, c in enumerate(unique):
        print(f"{idx:2d}: {c}  вес = {weight(c)}")

    # Матрица расстояний и поиск d_min
    n = len(unique)
    dist_matrix = [[0] * n for _ in range(n)]
    d_min = None
    for i in range(n):
        for j in range(n):
            d = hamming_distance(unique[i], unique[j])
            dist_matrix[i][j] = d
            if i != j:
                if d_min is None or d < d_min:
                    d_min = d

    print("\nМинимальное расстояние d_min =", d_min)

    # Можно вывести матрицу расстояний как в отчёте
    print("\nМатрица расстояний между кодовыми словами:")
    header = "     " + " ".join(f"{j:2d}" for j in range(n))
    print(header)
    for i in range(n):
        row = " ".join(f"{dist_matrix[i][j]:2d}" for j in range(n))
        print(f"{i:2d}: {row}")


if __name__ == "__main__":
    main()