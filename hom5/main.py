from itertools import product


def int_to_bits(x: int, length: int) -> list[int]:
    """Преобразовать целое число в двоичный вектор фиксированной длины (старший бит слева)."""
    return [(x >> i) & 1 for i in range(length - 1, -1, -1)]


def build_parity_check_matrix(r: int) -> list[list[int]]:
    """
    Формирует прямоугольную таблицу размером r x (2^r - 1),
    в столбцах которой записаны все ненулевые двоичные векторы длины r.

    Такая матрица используется как проверочная для кода Хэмминга,
    а как порождающая — для его дуального кода.
    """
    n = 2**r - 1
    H = [[0] * n for _ in range(r)]
    col = 0
    for value in range(1, 2**r):
        bits = int_to_bits(value, r)
        for row in range(r):
            H[row][col] = bits[row]
        col += 1
    return H


def multiply_vector_matrix(v: list[int], M: list[list[int]]) -> list[int]:
    """Перемножение (v) * M над полем из двух элементов."""
    rows = len(M)
    cols = len(M[0])
    assert len(v) == rows
    result = [0] * cols
    for j in range(cols):
        acc = 0
        for i in range(rows):
            acc ^= (v[i] & M[i][j])
        result[j] = acc
    return result


def weight(vec: list[int]) -> int:
    """Вес двоичного вектора (число единиц)."""
    return sum(vec)


def distance(a: list[int], b: list[int]) -> int:
    """Расстояние Хэмминга между двумя двоичными векторами одинаковой длины."""
    return sum(x ^ y for x, y in zip(a, b))


def analyse_code(r: int) -> dict:
    """
    Строит дуальный коду Хэмминга код длины n = 2^r - 1 и размерности k = r
    и проверяет, что все ненулевые кодовые слова имеют один и тот же вес,
    а попарные расстояния между различными словами совпадают.
    """
    H = build_parity_check_matrix(r)
    n = 2**r - 1
    k = r
    target_weight = 2 ** (r - 1)

    messages = list(product([0, 1], repeat=k))
    codewords = [multiply_vector_matrix(list(m), H) for m in messages]

    nonzero_weights = [weight(c) for c in codewords if any(c)]
    all_weights_equal = all(w == target_weight for w in nonzero_weights)

    distances = set()
    for i in range(len(codewords)):
        for j in range(i + 1, len(codewords)):
            d = distance(codewords[i], codewords[j])
            if d != 0:
                distances.add(d)

    single_distance = (len(distances) == 1)

    return {
        "n": n,
        "k": k,
        "target_weight": target_weight,
        "weights_set": sorted(set(nonzero_weights)),
        "all_weights_equal": all_weights_equal,
        "distances_set": sorted(distances),
        "single_distance": single_distance,
    }


def main() -> None:
    """Запуск проверки для нескольких значений r."""
    for r in (2, 3, 4, 5):
        info = analyse_code(r)
        n = info["n"]
        k = info["k"]
        print(f"r = {r}: ({n}, {k})-код")
        print(f"  ожидаемый вес ненулевых слов: {info['target_weight']}")
        print(f"  уникальные ненулевые веса: {info['weights_set']}")
        print(f"  все ненулевые слова одного веса: {'ДА' if info['all_weights_equal'] else 'НЕТ'}")
        print(f"  множество попарных расстояний: {info['distances_set']}")
        print(f"  расстояние между любыми разными словами одинаково: {'ДА' if info['single_distance'] else 'НЕТ'}")
        print()


if __name__ == "__main__":
    main()

