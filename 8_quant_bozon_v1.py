"""
8_quant_bozon_v1

Stvarni Qiskit/Aer kvantni pristup inspirisan Higgs/bozon mehanizmom.

Napomena:
  - ne koristi frekvenciju pojedinacnih brojeva
  - radi nad celim lex-indeksima / celim kombinacijama
  - 25 qubita = 5 blokova x 5 qubita, bez sirenja na 35q

Ideja:
  Higgs/bozon analog nije lokalna mezonska petlja, nego globalno polje.
  Peti 5q blok se tretira kao "Higgs field" koji preko CRY sprega daje
  razlicitu efektivnu "masu" prva cetiri bloka. To cuva celu kombinaciju
  kao jedno kvantno stanje, bez razbijanja na frekvenciju brojeva.

Output:
  8_quant_bozon_v1.txt
  8_quant_bozon_v1.png
"""

import csv
import math
import os
import random
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


T0 = time.time()
SEED = 39
CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7_4626_k44.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
TXT_OUT = os.path.join(HERE, "8_quant_bozon_v1.txt")
PNG_OUT = os.path.join(HERE, "8_quant_bozon_v1.png")

N_NUMBERS = 39
K_PICK = 7
TOTAL_COMB = math.comb(N_NUMBERS, K_PICK)
PLACEHOLDER = (1, 2, 3, 4, 5, 6, 7)

N_QUBITS = 25
BLOCKS = 5
Q_PER_BLOCK = 5
HIGGS_BLOCK = 4
LAYERS = 4

TRAIN_ITERS = 80
TRAIN_SHOTS = 4096
FINAL_SHOTS = 100000
TOP_K = 12
TARGET_SAMPLE_N = 768
GEN_SAMPLE_N = 768
MMD_SIGMA = 0.18
HIGGS_VEV = 0.55


def fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))


def load_loto_csv(path: str) -> tuple[list[tuple[int, ...]], int]:
    rows: list[tuple[int, ...]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            vals: list[int] = []
            for cell in row:
                try:
                    vals.append(int(str(cell).strip()))
                except ValueError:
                    continue
            if len(vals) < K_PICK:
                skipped += 1
                continue
            combo = tuple(sorted(vals[:K_PICK]))
            if len(set(combo)) == K_PICK and all(1 <= x <= N_NUMBERS for x in combo):
                rows.append(combo)
            else:
                skipped += 1
    if not rows:
        raise ValueError("CSV nije ucitan: nema validnih 7/39 kombinacija.")
    return rows, skipped


def lex_rank(combo: tuple[int, ...]) -> int:
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo, start=1):
        for x in range(prev + 1, value):
            rank0 += math.comb(N_NUMBERS - x, K_PICK - i)
        prev = value
    return rank0 + 1


def lex_derank(rank: int) -> tuple[int, ...]:
    r = int(rank) - 1
    combo: list[int] = []
    start = 1
    for i in range(K_PICK):
        remaining = K_PICK - i - 1
        for x in range(start, N_NUMBERS + 1):
            cnt = math.comb(N_NUMBERS - x, remaining)
            if r < cnt:
                combo.append(x)
                start = x + 1
                break
            r -= cnt
    return tuple(combo)


def int_to_bitstring(value: int, n_bits: int = N_QUBITS) -> str:
    return format(int(value), f"0{n_bits}b")


def lex_region(lex_val: int) -> str:
    pct = 100.0 * int(lex_val) / TOTAL_COMB
    decile = min(10, max(1, int(math.ceil(pct / 10.0))))
    return f"D{decile} ({pct:.2f}%)"


def recency_weights(n: int, tau: float = 950.0) -> np.ndarray:
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-ages / tau)
    weights /= weights.sum()
    return weights


def weighted_target_bits(lex_indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(N_QUBITS, dtype=np.float64)
    for idx, w in zip(lex_indices, weights):
        bits = int_to_bitstring(int(idx) - 1)
        out += w * np.fromiter((1.0 if b == "1" else 0.0 for b in bits), dtype=np.float64)
    return out


def weighted_target_sample(
    lex_indices: np.ndarray,
    weights: np.ndarray,
    sample_n: int = TARGET_SAMPLE_N,
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(lex_indices), size=sample_n, replace=True, p=weights)
    sample = lex_indices[pick].astype(np.float64) / TOTAL_COMB
    return sample.reshape(-1, 1)


def gaussian_mmd(x: np.ndarray, y: np.ndarray, sigma: float = MMD_SIGMA) -> float:
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xx = (x - x.T) ** 2
    yy = (y - y.T) ** 2
    xy = (x - y.T) ** 2
    denom = 2.0 * sigma * sigma
    kxx = np.exp(-xx / denom).mean()
    kyy = np.exp(-yy / denom).mean()
    kxy = np.exp(-xy / denom).mean()
    return float(kxx + kyy - 2.0 * kxy)


def counts_to_valid_lex_sample(counts: dict[str, int], sample_n: int = GEN_SAMPLE_N) -> np.ndarray:
    vals: list[float] = []
    for bitstr, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        lex_val = int(clean, 2) + 1
        if 1 <= lex_val <= TOTAL_COMB:
            vals.extend([lex_val / TOTAL_COMB] * min(int(count), sample_n - len(vals)))
        if len(vals) >= sample_n:
            break
    if not vals:
        vals = [0.5]
    while len(vals) < sample_n:
        vals.append(vals[-1])
    return np.array(vals[:sample_n], dtype=np.float64).reshape(-1, 1)


def counts_to_bit_probs(counts: dict[str, int]) -> np.ndarray:
    total = max(1, sum(counts.values()))
    probs = np.zeros(N_QUBITS, dtype=np.float64)
    for bitstr, count in counts.items():
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        probs += count * np.fromiter((1.0 if b == "1" else 0.0 for b in clean), dtype=np.float64)
    return probs / total


def params_per_layer() -> int:
    single = 2 * N_QUBITS
    intra_block_cry = BLOCKS * (Q_PER_BLOCK - 1)
    higgs_ring = 1
    higgs_mass_couplings = (BLOCKS - 1) * Q_PER_BLOCK
    matter_backreaction = BLOCKS - 1
    return single + intra_block_cry + higgs_ring + higgs_mass_couplings + matter_backreaction


def build_bozon_qcbm(theta: np.ndarray, seed_lex: int) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    seed_bits = int_to_bitstring(int(seed_lex) - 1)
    for q, bit in enumerate(reversed(seed_bits)):
        if bit == "1":
            qc.x(q)

    p = 0
    higgs_start = HIGGS_BLOCK * Q_PER_BLOCK
    higgs_qubits = list(range(higgs_start, higgs_start + Q_PER_BLOCK))
    for layer in range(LAYERS):
        sign = 1.0 if layer % 2 == 0 else -1.0

        for q in range(N_QUBITS):
            field_shift = sign * HIGGS_VEV if q in higgs_qubits else 0.0
            qc.ry(float(theta[p] + field_shift), q)
            p += 1
            qc.rz(float(theta[p]), q)
            p += 1

        for block in range(BLOCKS):
            start = block * Q_PER_BLOCK
            for j in range(Q_PER_BLOCK - 1):
                qc.cry(float(theta[p]), start + j, start + j + 1)
                p += 1

        qc.cry(float(theta[p]), higgs_qubits[-1], higgs_qubits[0])
        p += 1

        # Globalno Higgs polje: peti blok kontrolise efektivne mase prva cetiri bloka.
        for block in range(BLOCKS - 1):
            start = block * Q_PER_BLOCK
            for j, hq in enumerate(higgs_qubits):
                qc.cry(float(theta[p]), hq, start + j)
                p += 1

        # Backreaction: materija vraca slab signal u centralni Higgs mod.
        h_center = higgs_start + 2
        for block in range(BLOCKS - 1):
            matter_center = block * Q_PER_BLOCK + 2
            qc.cry(float(theta[p]), matter_center, h_center)
            p += 1

        for block in range(BLOCKS - 1):
            qc.cz(block * Q_PER_BLOCK + 2, h_center)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def run_counts(
    theta: np.ndarray,
    simulator: AerSimulator,
    shots: int,
    seed_lex: int,
    seed_offset: int = 0,
) -> dict[str, int]:
    qc = build_bozon_qcbm(theta, seed_lex)
    tqc = transpile(qc, simulator, optimization_level=1, seed_transpiler=SEED + seed_offset)
    result = simulator.run(tqc, shots=shots, seed_simulator=SEED + seed_offset).result()
    return result.get_counts()


def cost_from_counts(
    counts: dict[str, int],
    target_sample: np.ndarray,
    target_bits: np.ndarray,
) -> float:
    generated_sample = counts_to_valid_lex_sample(counts)
    mmd = gaussian_mmd(generated_sample, target_sample)
    bit_mse = float(np.mean((counts_to_bit_probs(counts) - target_bits) ** 2))
    return float(mmd + 0.15 * bit_mse)


def init_theta_from_target(target_bits: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    ppl = params_per_layer()
    theta = np.zeros(LAYERS * ppl, dtype=np.float64)
    base_ry = 2.0 * np.arcsin(np.sqrt(np.clip(target_bits, 1e-6, 1.0 - 1e-6)))

    p = 0
    for layer in range(LAYERS):
        sign = 1.0 if layer % 2 == 0 else -1.0
        layer_scale = 1.0 / math.sqrt(layer + 1.0)
        for q in range(N_QUBITS):
            symmetry_break = sign * HIGGS_VEV if q // Q_PER_BLOCK == HIGGS_BLOCK else 0.0
            theta[p] = base_ry[q] * layer_scale + symmetry_break + rng.normal(0.0, 0.035)
            p += 1
            theta[p] = rng.normal(0.0, 0.10)
            p += 1
        for _ in range(BLOCKS * (Q_PER_BLOCK - 1)):
            theta[p] = rng.normal(0.0, 0.20)
            p += 1
        theta[p] = rng.normal(0.0, 0.25)
        p += 1
        for _ in range((BLOCKS - 1) * Q_PER_BLOCK):
            theta[p] = rng.normal(0.0, 0.32)
            p += 1
        for _ in range(BLOCKS - 1):
            theta[p] = rng.normal(0.0, 0.22)
            p += 1
    return np.mod(theta, 2.0 * np.pi)


def spsa_train(
    theta0: np.ndarray,
    target_sample: np.ndarray,
    target_bits: np.ndarray,
    simulator: AerSimulator,
    seed_lex: int,
) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    rng = np.random.default_rng(SEED)
    theta = theta0.copy()
    losses: list[float] = []
    mmd_losses: list[float] = []
    bit_losses: list[float] = []

    for it in range(1, TRAIN_ITERS + 1):
        a = 0.13 / (it ** 0.33)
        c = 0.10 / (it ** 0.12)
        delta = rng.choice([-1.0, 1.0], size=theta.shape)

        counts_plus = run_counts(theta + c * delta, simulator, TRAIN_SHOTS, seed_lex, 2 * it)
        counts_minus = run_counts(theta - c * delta, simulator, TRAIN_SHOTS, seed_lex, 2 * it + 1)
        loss_plus = cost_from_counts(counts_plus, target_sample, target_bits)
        loss_minus = cost_from_counts(counts_minus, target_sample, target_bits)

        ghat = (loss_plus - loss_minus) / (2.0 * c) * delta
        theta = np.mod(theta - a * ghat, 2.0 * np.pi)

        counts_eval = counts_plus if loss_plus <= loss_minus else counts_minus
        generated = counts_to_valid_lex_sample(counts_eval)
        mmd_now = gaussian_mmd(generated, target_sample)
        bit_now = float(np.mean((counts_to_bit_probs(counts_eval) - target_bits) ** 2))
        loss_now = float(mmd_now + 0.15 * bit_now)

        losses.append(loss_now)
        mmd_losses.append(float(mmd_now))
        bit_losses.append(bit_now)
        print(
            f"  SPSA iter {it:02d}/{TRAIN_ITERS}  "
            f"loss={loss_now:.8f}  mmd={mmd_now:.8f}  bit_mse={bit_now:.8f}"
        )

    return theta, losses, mmd_losses, bit_losses


def valid_sample_rows(
    counts: dict[str, int],
    historical_set: set[int],
) -> tuple[list[dict[str, object]], int, int, int]:
    rows: list[dict[str, object]] = []
    seen_combos: set[tuple[int, ...]] = set()
    skipped_out = 0
    skipped_placeholder = 0
    skipped_seen = 0

    for bitstr, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        lex_val = int(clean, 2) + 1
        if not (1 <= lex_val <= TOTAL_COMB):
            skipped_out += int(count)
            continue
        combo = lex_derank(lex_val)
        if combo == PLACEHOLDER:
            skipped_placeholder += int(count)
            continue
        if lex_val in historical_set:
            skipped_seen += int(count)
            continue
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        rows.append(
            {
                "count": int(count),
                "prob": float(count) / FINAL_SHOTS,
                "lex": int(lex_val),
                "region": lex_region(int(lex_val)),
                "combo": combo,
            }
        )
        if len(rows) >= TOP_K:
            break

    return rows, skipped_out, skipped_placeholder, skipped_seen


def make_png(
    losses: list[float],
    mmd_losses: list[float],
    rows: list[dict[str, object]],
    target_bits: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=1.4, label="loss")
    ax1.plot(range(1, len(mmd_losses) + 1), mmd_losses, linewidth=1.2, label="MMD")
    ax1.set_title("Bozon QCBM SPSA loss")
    ax1.set_xlabel("iter")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(N_QUBITS), target_bits, color="#7c3aed")
    ax2.axvspan(HIGGS_BLOCK * Q_PER_BLOCK - 0.5, N_QUBITS - 0.5, color="#fde68a", alpha=0.25)
    ax2.set_title("Target bit amplitude + Higgs blok")
    ax2.set_xlabel("bit pozicija")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_rows = [
        [i + 1, row["count"], f"{row['prob']:.6f}", row["lex"], row["region"], str(row["combo"])]
        for i, row in enumerate(rows)
    ]
    table = ax3.table(
        cellText=table_rows,
        colLabels=["rang", "count", "prob", "lex", "region", "kombinacija"],
        cellLoc="center",
        loc="center",
        colWidths=[0.06, 0.09, 0.10, 0.15, 0.12, 0.39],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#312e81")
            cell.set_text_props(color="white", weight="bold")
        elif r == 1:
            cell.set_facecolor("#ede9fe")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")

    fig.suptitle("8_quant_bozon_v1 - Qiskit QCBM Higgs/global-field 25q", fontweight="bold")
    fig.tight_layout()
    plt.show()
    fig.savefig(PNG_OUT, dpi=200, bbox_inches="tight")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print()
    print("=" * 72)
    print("8_quant_bozon_v1 - Qiskit QCBM Higgs/global-field nad lex-indeksima")
    print("=" * 72)
    print()

    combos, skipped_rows = load_loto_csv(CSV_PATH)
    lex_indices = np.array([lex_rank(c) for c in combos], dtype=np.int64)
    historical_set = set(int(x) for x in lex_indices)
    weights = recency_weights(len(lex_indices))
    target_bits = weighted_target_bits(lex_indices, weights)
    target_sample = weighted_target_sample(lex_indices, weights)
    seed_lex = int(lex_indices[-1])

    print(f"CSV:                 {CSV_PATH}")
    print(f"Validnih izvlacenja: {len(combos)}")
    print(f"Preskocenih redova:  {skipped_rows}")
    print(f"C(39,7):             {TOTAL_COMB:,}")
    print(f"Zadnji lex seed:     {seed_lex:,}")
    print(f"Qubita:              {N_QUBITS} = {BLOCKS} blokova x {Q_PER_BLOCK} qubita")
    print(f"Higgs blok:          blok {HIGGS_BLOCK + 1} / qubits {HIGGS_BLOCK * Q_PER_BLOCK}-{N_QUBITS - 1}")
    print(f"Layers:              {LAYERS}")
    print(f"Parametara:          {LAYERS * params_per_layer()}")
    print(f"Simulator:           AerSimulator qasm, shots train={TRAIN_SHOTS}, final={FINAL_SHOTS}")
    print()

    simulator = AerSimulator(method="automatic")
    theta0 = init_theta_from_target(target_bits)

    t_train = time.time()
    theta, losses, mmd_losses, bit_losses = spsa_train(
        theta0,
        target_sample,
        target_bits,
        simulator,
        seed_lex,
    )
    train_seconds = time.time() - t_train

    print()
    print("Finalno semplovanje istreniranog bozon kola...")
    final_counts = run_counts(theta, simulator, FINAL_SHOTS, seed_lex, 10_000)
    rows, skipped_out, skipped_placeholder, skipped_seen = valid_sample_rows(final_counts, historical_set)

    if not rows:
        raise RuntimeError("Nema validnih novih sampled lex kandidata posle filtera.")

    main_row = rows[0]
    total_seconds = time.time() - T0

    lines: list[str] = []
    lines.append("8_quant_bozon_v1 - Qiskit QCBM Higgs/global-field 25q")
    lines.append("=" * 72)
    lines.append("")
    lines.append("KORAK 1: Weierstrass lex-kriva nad svim validnim do sad izvucenim kombinacijama")
    lines.append("")
    lines.append(f"  CSV izvucenih:        {CSV_PATH}")
    lines.append(f"  Validnih izvlacenja:   {len(combos)}")
    lines.append(f"  Preskocenih redova:    {skipped_rows}")
    lines.append(f"  C(39,7):              {TOTAL_COMB:,}")
    lines.append(f"  Zadnji lex seed:       {seed_lex:,}")
    lines.append("  f(t) = lex-indeks cele kombinacije u skupu svih 39C7")
    lines.append("")
    lines.append("KORAK 2: Stvarni kvantni model BOZON")
    lines.append("")
    lines.append("  Model:                QCBM / parametrizovano kvantno kolo")
    lines.append("  Loss:                 MMD(lex distribucija) + 0.15*MSE(bit-marginale)")
    lines.append("  Recency:              exponential weights nad celom krivom")
    lines.append(f"  Qubita:               {N_QUBITS} = {BLOCKS} blokova x {Q_PER_BLOCK}")
    lines.append(f"  Layers:               {LAYERS}")
    lines.append(f"  Parametara:           {len(theta)}")
    lines.append("  Conditional seed:     zadnji lex-indeks enkodovan X-gateovima")
    lines.append("  Higgs field:          peti 5q blok kao globalno polje")
    lines.append("  Symmetry breaking:    +/- Higgs VEV u inicijalizaciji i slojevima")
    lines.append("  Coupling:             Higgs blok -> prva cetiri bloka preko CRY")
    lines.append("  NEMA frekvencije:     izlaz su cele kombinacije i lex-regioni")
    lines.append(f"  SPSA iteracija:        {TRAIN_ITERS}")
    lines.append(f"  train shots:           {TRAIN_SHOTS}")
    lines.append(f"  final shots:           {FINAL_SHOTS}")
    lines.append(f"  initial loss:          {losses[0]:.8f}")
    lines.append(f"  final loss:            {losses[-1]:.8f}")
    lines.append(f"  final MMD:             {mmd_losses[-1]:.8f}")
    lines.append(f"  final bit MSE:         {bit_losses[-1]:.8f}")
    lines.append("")
    lines.append("Filter finalnih kandidata:")
    lines.append(f"  out-of-range shots:    {skipped_out}")
    lines.append(f"  placeholder shots:     {skipped_placeholder}")
    lines.append(f"  vec izvuceni shots:    {skipped_seen}")
    lines.append("")
    lines.append("PREDIKCIJA 1: NEXT / 8_quant_bozon_v1")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Glavna kvantna BOZON prognoza:")
    lines.append(f"  sampled count:         {main_row['count']}")
    lines.append(f"  sampled prob:          {main_row['prob']:.8f}")
    lines.append(f"  pred. lex:             {main_row['lex']:,}")
    lines.append(f"  lex-region:            {main_row['region']}")
    lines.append(f"  pred. kombinacija:     {main_row['combo']}")
    lines.append("  vec izvucena ranije:   NE (filtrirano)")
    lines.append("")
    lines.append("Top kvantni BOZON kandidati (cele kombinacije, ne frekvencija brojeva):")
    lines.append(f"  {'rang':<5}{'count':>8}{'prob':>12}{'lex':>14}  {'region':<12} {'kombinacija':<30}")
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"  {i:<5}{row['count']:>8}{row['prob']:>12.8f}{row['lex']:>14,}  "
            f"{str(row['region']):<12} {str(row['combo']):<30}"
        )
    lines.append("")
    lines.append(f"Vreme treninga:       {fmt_time(train_seconds)} ({train_seconds:.1f} s)")
    lines.append(f"Ukupno vreme:         {fmt_time(total_seconds)} ({total_seconds:.1f} s)")
    lines.append(f"PNG:                  {PNG_OUT}")
    lines.append("")

    text = "\n".join(lines)
    print()
    print(text)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"TXT saved -> {TXT_OUT}")

    make_png(losses, mmd_losses, rows, target_bits)
    print(f"PNG saved -> {PNG_OUT}")
    print()


if __name__ == "__main__":
    main()



"""
========================================================================
8_quant_bozon_v1 - Qiskit QCBM Higgs/global-field nad lex-indeksima
========================================================================

CSV:                 /Users/4c/Desktop/GHQ/data/loto7_4626_k44.csv
Validnih izvlacenja: 4626
Preskocenih redova:  0
C(39,7):             15,380,937
Zadnji lex seed:     2,770,100
Qubita:              25 = 5 blokova x 5 qubita
Higgs blok:          blok 5 / qubits 20-24
Layers:              4
Parametara:          380
Simulator:           AerSimulator qasm, shots train=4096, final=100000

  SPSA iter 01/80  loss=0.02932031  mmd=0.01348172  bit_mse=0.10559060
  SPSA iter 02/80  loss=0.03204247  mmd=0.01549408  bit_mse=0.11032258
  SPSA iter 03/80  loss=0.21227297  mmd=0.19583365  bit_mse=0.10959541
  SPSA iter 04/80  loss=0.37541101  mmd=0.35888411  bit_mse=0.11017933
  SPSA iter 05/80  loss=0.24199417  mmd=0.22651256  bit_mse=0.10321076
  SPSA iter 06/80  loss=0.14651457  mmd=0.13442538  bit_mse=0.08059460
  SPSA iter 07/80  loss=0.03028248  mmd=0.01897853  bit_mse=0.07535970
  SPSA iter 08/80  loss=0.17295822  mmd=0.16303207  bit_mse=0.06617427
  SPSA iter 09/80  loss=0.06708628  mmd=0.05521479  bit_mse=0.07914327
  SPSA iter 10/80  loss=0.05805026  mmd=0.04569156  bit_mse=0.08239134
  SPSA iter 11/80  loss=0.09009665  mmd=0.07693215  bit_mse=0.08776331
  SPSA iter 12/80  loss=0.04124662  mmd=0.02719118  bit_mse=0.09370295
  SPSA iter 13/80  loss=0.03937170  mmd=0.02434791  bit_mse=0.10015860
  SPSA iter 14/80  loss=0.03800339  mmd=0.02576847  bit_mse=0.08156615
  SPSA iter 15/80  loss=0.03002403  mmd=0.01573898  bit_mse=0.09523372
  SPSA iter 16/80  loss=0.03471028  mmd=0.02071227  bit_mse=0.09332010
  SPSA iter 17/80  loss=0.15579643  mmd=0.14143924  bit_mse=0.09571460
  SPSA iter 18/80  loss=0.12786238  mmd=0.11409266  bit_mse=0.09179809
  SPSA iter 19/80  loss=0.35388411  mmd=0.34315944  bit_mse=0.07149774
  SPSA iter 20/80  loss=0.26766534  mmd=0.25646861  bit_mse=0.07464486
  SPSA iter 21/80  loss=0.04673917  mmd=0.03653428  bit_mse=0.06803261
  SPSA iter 22/80  loss=0.08723008  mmd=0.07584497  bit_mse=0.07590074
  SPSA iter 23/80  loss=0.11318694  mmd=0.10225942  bit_mse=0.07285012
  SPSA iter 24/80  loss=0.02811484  mmd=0.01806413  bit_mse=0.06700475
  SPSA iter 25/80  loss=0.08975962  mmd=0.07848303  bit_mse=0.07517723
  SPSA iter 26/80  loss=0.21452121  mmd=0.20154434  bit_mse=0.08651244
  SPSA iter 27/80  loss=0.26159789  mmd=0.24802788  bit_mse=0.09046673
  SPSA iter 28/80  loss=0.12154511  mmd=0.10980146  bit_mse=0.07829096
  SPSA iter 29/80  loss=0.02514284  mmd=0.01240397  bit_mse=0.08492582
  SPSA iter 30/80  loss=0.06853099  mmd=0.05332909  bit_mse=0.10134600
  SPSA iter 31/80  loss=0.11099977  mmd=0.09828974  bit_mse=0.08473352
  SPSA iter 32/80  loss=0.11219314  mmd=0.09978924  bit_mse=0.08269267
  SPSA iter 33/80  loss=0.03725183  mmd=0.02440073  bit_mse=0.08567402
  SPSA iter 34/80  loss=0.07015381  mmd=0.05649622  bit_mse=0.09105064
  SPSA iter 35/80  loss=0.09491042  mmd=0.08167960  bit_mse=0.08820546
  SPSA iter 36/80  loss=0.08480713  mmd=0.07191666  bit_mse=0.08593645
  SPSA iter 37/80  loss=0.06520394  mmd=0.05466449  bit_mse=0.07026300
  SPSA iter 38/80  loss=0.04302837  mmd=0.03408125  bit_mse=0.05964747
  SPSA iter 39/80  loss=0.07505054  mmd=0.06613869  bit_mse=0.05941231
  SPSA iter 40/80  loss=0.07990316  mmd=0.06995396  bit_mse=0.06632795
  SPSA iter 41/80  loss=0.07384330  mmd=0.06352561  bit_mse=0.06878463
  SPSA iter 42/80  loss=0.05980329  mmd=0.04903501  bit_mse=0.07178857
  SPSA iter 43/80  loss=0.05129527  mmd=0.04178633  bit_mse=0.06339295
  SPSA iter 44/80  loss=0.06001686  mmd=0.04917156  bit_mse=0.07230197
  SPSA iter 45/80  loss=0.07586859  mmd=0.06641305  bit_mse=0.06303689
  SPSA iter 46/80  loss=0.04329320  mmd=0.03373468  bit_mse=0.06372344
  SPSA iter 47/80  loss=0.04905375  mmd=0.04002873  bit_mse=0.06016681
  SPSA iter 48/80  loss=0.04069180  mmd=0.03138043  bit_mse=0.06207577
  SPSA iter 49/80  loss=0.03922908  mmd=0.02984927  bit_mse=0.06253204
  SPSA iter 50/80  loss=0.08349415  mmd=0.07350470  bit_mse=0.06659630
  SPSA iter 51/80  loss=0.06751433  mmd=0.05675966  bit_mse=0.07169783
  SPSA iter 52/80  loss=0.03881138  mmd=0.02852372  bit_mse=0.06858436
  SPSA iter 53/80  loss=0.03404366  mmd=0.02164923  bit_mse=0.08262955
  SPSA iter 54/80  loss=0.03088272  mmd=0.01854610  bit_mse=0.08224414
  SPSA iter 55/80  loss=0.03543839  mmd=0.02494273  bit_mse=0.06997103
  SPSA iter 56/80  loss=0.03925040  mmd=0.02894776  bit_mse=0.06868429
  SPSA iter 57/80  loss=0.04056941  mmd=0.02668326  bit_mse=0.09257435
  SPSA iter 58/80  loss=0.04117212  mmd=0.02723246  bit_mse=0.09293108
  SPSA iter 59/80  loss=0.04712235  mmd=0.03222486  bit_mse=0.09931659
  SPSA iter 60/80  loss=0.02186355  mmd=0.00908371  bit_mse=0.08519887
  SPSA iter 61/80  loss=0.03393670  mmd=0.02138018  bit_mse=0.08371014
  SPSA iter 62/80  loss=0.01648655  mmd=0.00280143  bit_mse=0.09123419
  SPSA iter 63/80  loss=0.02158073  mmd=0.00821763  bit_mse=0.08908732
  SPSA iter 64/80  loss=0.07890972  mmd=0.06460486  bit_mse=0.09536576
  SPSA iter 65/80  loss=0.02016041  mmd=0.00649194  bit_mse=0.09112316
  SPSA iter 66/80  loss=0.01980345  mmd=0.00520020  bit_mse=0.09735499
  SPSA iter 67/80  loss=0.02201394  mmd=0.00844035  bit_mse=0.09049060
  SPSA iter 68/80  loss=0.03056921  mmd=0.01694660  bit_mse=0.09081737
  SPSA iter 69/80  loss=0.02270683  mmd=0.00871370  bit_mse=0.09328752
  SPSA iter 70/80  loss=0.02307462  mmd=0.00978224  bit_mse=0.08861588
  SPSA iter 71/80  loss=0.02309487  mmd=0.01057602  bit_mse=0.08345895
  SPSA iter 72/80  loss=0.01850542  mmd=0.00364161  bit_mse=0.09909210
  SPSA iter 73/80  loss=0.02741196  mmd=0.01366572  bit_mse=0.09164160
  SPSA iter 74/80  loss=0.02375897  mmd=0.01000088  bit_mse=0.09172058
  SPSA iter 75/80  loss=0.02522828  mmd=0.01198971  bit_mse=0.08825715
  SPSA iter 76/80  loss=0.01906467  mmd=0.00659040  bit_mse=0.08316179
  SPSA iter 77/80  loss=0.03406558  mmd=0.01998067  bit_mse=0.09389944
  SPSA iter 78/80  loss=0.02689187  mmd=0.01286725  bit_mse=0.09349749
  SPSA iter 79/80  loss=0.01790362  mmd=0.00458379  bit_mse=0.08879888
  SPSA iter 80/80  loss=0.02135480  mmd=0.00763071  bit_mse=0.09149391

Finalno semplovanje istreniranog bozon kola...

8_quant_bozon_v1 - Qiskit QCBM Higgs/global-field 25q
========================================================================

KORAK 1: Weierstrass lex-kriva nad svim validnim do sad izvucenim kombinacijama

  CSV izvucenih:        /Users/4c/Desktop/GHQ/data/loto7_4626_k44.csv
  Validnih izvlacenja:   4626
  Preskocenih redova:    0
  C(39,7):              15,380,937
  Zadnji lex seed:       2,770,100
  f(t) = lex-indeks cele kombinacije u skupu svih 39C7

KORAK 2: Stvarni kvantni model BOZON

  Model:                QCBM / parametrizovano kvantno kolo
  Loss:                 MMD(lex distribucija) + 0.15*MSE(bit-marginale)
  Recency:              exponential weights nad celom krivom
  Qubita:               25 = 5 blokova x 5
  Layers:               4
  Parametara:           380
  Conditional seed:     zadnji lex-indeks enkodovan X-gateovima
  Higgs field:          peti 5q blok kao globalno polje
  Symmetry breaking:    +/- Higgs VEV u inicijalizaciji i slojevima
  Coupling:             Higgs blok -> prva cetiri bloka preko CRY
  NEMA frekvencije:     izlaz su cele kombinacije i lex-regioni
  SPSA iteracija:        80
  train shots:           4096
  final shots:           100000
  initial loss:          0.02932031
  final loss:            0.02135480
  final MMD:             0.00763071
  final bit MSE:         0.09149391

Filter finalnih kandidata:
  out-of-range shots:    1806
  placeholder shots:     0
  vec izvuceni shots:    0

PREDIKCIJA 1: NEXT / 8_quant_bozon_v1
========================================================================

Glavna kvantna BOZON prognoza:
  sampled count:         46
  sampled prob:          0.00046000
  pred. lex:             1,054,932
  lex-region:            D1 (6.86%)
  pred. kombinacija:     (1, 4, 12, 23, 32, 36, 38)
  vec izvucena ranije:   NE (filtrirano)

Top kvantni BOZON kandidati (cele kombinacije, ne frekvencija brojeva):
  rang    count        prob           lex  region       kombinacija                   
  1          46  0.00046000     1,054,932  D1 (6.86%)   (1, 4, 12, 23, 32, 36, 38)    
  2          39  0.00039000     3,152,082  D3 (20.49%)  (2, 4, 5, 8, 24, 29, 37)      
  3          33  0.00033000    11,802,834  D8 (76.74%)  (7, 14, 16, 24, 29, 34, 36)   
  4          32  0.00032000     9,705,684  D7 (63.10%)  (5, 13, 14, 15, 17, 24, 33)   
  5          30  0.00030000    11,540,690  D8 (75.03%)  (7, 11, 12, 13, 16, 19, 29)   
  6          29  0.00029000     9,443,540  D7 (61.40%)  (5, 10, 12, 23, 28, 29, 38)   
  7          26  0.00026000    13,637,716  D9 (88.67%)  (10, 14, 23, 26, 27, 30, 37)  
  8          24  0.00024000     5,249,108  D4 (34.13%)  (3, 4, 9, 12, 21, 24, 27)     
  9          24  0.00024000    13,703,252  D9 (89.09%)  (10, 16, 19, 28, 29, 36, 38)  
  10         23  0.00023000    14,030,932  D10 (91.22%) (11, 15, 17, 19, 28, 32, 35)  
  11         23  0.00023000    13,965,396  D10 (90.80%) (11, 13, 27, 28, 29, 31, 38)  
  12         23  0.00023000     3,160,274  D3 (20.55%)  (2, 4, 5, 10, 33, 36, 38)     

Vreme treninga:       0:21:57 (1317.3 s)
Ukupno vreme:         0:22:03 (1323.1 s)
PNG:                  /Users/4c/Desktop/GHQ/KarlWeierstrass/8_quant_bozon_v1.png

TXT saved -> /Users/4c/Desktop/GHQ/KarlWeierstrass/8_quant_bozon_v1.txt
PNG saved -> /Users/4c/Desktop/GHQ/KarlWeierstrass/8_quant_bozon_v1.png
"""





"""
Analiza BOZON v1
Zadnji seed lex je 2,770,100.

Trening loss:

initial: 0.02932031
final: 0.02135480
Pad ~27.2%. 
Konvergirao jeste, ali slabije od mezona v2 (koji je pao ~60%). 
Razlog: bozon kolo ima 380 parametara (vs 312 kod mezona) 
i mnogo gušći entanglement (Higgs coupling ide na 20 qubita), 
pa SPSA u istih 80 iteracija teže pomera tako veliki prostor. 
Trening je i mnogo duži: 22 minuta vs ~8 min za mezon v2.

Ključna razlika — multimodalnost je mnogo jača:

Mezon v2 je grupisao kandidate u 2-3 uska lex-klastera. 
BOZON v1 ih razmazuje preko celog prostora — pogledaj region kolonu:

region	kandidati
D1 (6.86%)
rang 1
D3 (~20%)
rang 2, 12
D4 (34%)
rang 8
D7 (61-63%)
rang 4, 6
D8 (75-77%)
rang 3, 5
D9 (88-89%)
rang 7, 9
D10 (90-91%)
rang 10, 11
To je upravo ono što symmetry breaking treba da uradi — 
Higgs polje "bira stranu" i pravi više vakuuma, 
pa distribucija ima više modova razbacanih po celom lex prostoru. 
Fizička analogija je ovde stvarno proradila u ponašanju kola.

Glavna prognoza:

(1, 4, 12, 23, 32, 36, 38) — lex 1,054,932, donji deo prostora (D1).

Zanimljivo: glavna prognoza je u D1 (kombinacije koje počinju sa 1), 
dok je kod mezona glavna bila u D10 (počinje sa 17). 
Dva različita kvantna ansatza pokazuju na suprotne krajeve lex-krive.

Poređenje mezon v2 vs bozon v1 (na nivou kombinacija, ne frekvencije):

Mezon v2 glavna: (17, 18, 20, 25, 28, 30, 38) — D10, vrh prostora
Bozon v1 glavna: (1, 4, 12, 23, 32, 36, 38) — D1, dno prostora
Preklapanja lex-regiona: oba modela imaju kandidate u D7-D10 (gornji deo). 
Mezon je bio koncentrisan tamo, bozon ima i tamo (rang 3,4,5,7,9,10,11) ali i raširenu donju polovinu.

Zajednička tačka koju vredi primetiti: 
broj 38 se pojavljuje u glavnoj prognozi oba modela (...36, 38 kod bozona, ...30, 38 kod mezona). 
Ne kao frekvencija — nego kao deo strukturno najjače kombinacije u oba nezavisna kola. 
To je jedini "konsenzus" signal.

Filteri:

out-of-range: 1806 / 100000 = 1.8% (još bolje od mezona 2.86%)
placeholder: 0
već izvučeni: 0
Higgs coupling drži uzorke unutar validnog prostora bolje nego mezonska petlja.

Zaključak BOZON v1
Konvergencija slabija (27% vs 60% mezon), ali model radi.
Mnogo bogatija multimodalnost — pokriva ceo lex prostor, ne samo jednu zonu. 
To je direktna posledica Higgs symmetry-breaking strukture i baš ono čemu se analogija nadala.
Daje suprotnu glavnu prognozu od mezona (D1 vs D10) — što je korisno: dva nezavisna kvantna pogleda na istu krivu.
Cena: 22 min trening.
Glavna BOZON NEXT prognoza:

(1, 4, 12, 23, 32, 36, 38)

Ako tražiš presek dva kvantna modela, jedini jak zajednički element je gornja zona (D7-D10) + broj 38 u obe glavne kombinacije.
"""







"""
Higsob bozon (Higgs/global-field) Bozija cestica

kao i mezoni, i ovo je fizička analogija nad istom lex-krivom. 
Razlika je u tome kakvu strukturu kola/treninga analogija sugeriše.

Šta bi „Higgs/bozon" realno mogao značiti za moj model:

Fizička ideja → kvantni mehanizam:

Higgs daje masu kroz spontano narušenje simetrije (Mexican-hat potencijal). 
Sve čestice se "kače" na Higgs polje različitom jačinom (coupling ∝ masa).
Kvantni analog: 
globalno polje-qubit (ili mali registar) koji je spregnut sa svim blokovima i daje svakom bloku različitu "masu" = različitu jačinu rotacije. 
To je drugačije od mezona, gde je sprega bila lokalna petlja blok-do-blok.

Konkretno za 8_quant_bozon_v1.py:
Isti lex pipeline (4626, 25 qubita = 5x5, Aer, bez 35q)
Higgs registar (npr. zadnji blok kao "polje") koji preko CRY daje masu ostalim blokovima — 
svaki broj/pozicija dobija svoj coupling
Symmetry breaking inicijalizacija: 
parametri ne kreću iz nule/simetrije nego iz dva minimuma (Mexican-hat), 
pa kolo "bira stranu" — to prirodno daje multimodalnu distribuciju
Isti MMD loss + SPSA kao v2
Isti TXT/PNG izlaz + NEXT predikcija
Rezultat će opet biti uporediv sa v2 — neće "razbiti" loto. 
Vrednost je u tome da im treća nezavisna kvantna prognozu drugačije strukturisanu 
(globalna sprega umesto lokalne petlje), pa može da uđe kao još jedan model 


Frekvencija brojeva je baš ono od čega trba bežati 
jer uništava strukturu: razbije kombinaciju (17,18,20,25,28,30,38) 
na 7 nezavisnih brojeva i izgubi se sve o tome koji broj ide sa kojim.

Kvantni pristup je vredan baš zato što radi suprotno — 
uzorkuje celu kombinaciju kao jedno stanje (lex-indeks = jedan bitstring), 
pa čuva korelacije između brojeva. 
Ako bozon-rezultat posle sameljemo u frekvenciju, bacili bismo upravo tu prednost.

Pa umesto frekvencije, bozon model dodaje vrednost na nivou celih kombinacija:

daje svoju top-K listu kombinacija (ne brojeva)
onda poredim kvantne modele (mezon v2 vs bozon v1) na nivou preklapanja kombinacija / lex-regiona — 
da li dva nezavisno strukturisana kola pokazuju na iste lex-oblasti
ako se dva različita kvantna ansatza slažu oko istog lex-klastera, 
to je jači signal nego bilo koja frekvencija brojeva
Drugim rečima: 
agregacija da, ali po kombinacijama/lex-regionima, ne po pojedinačnim brojevima.

8_quant_bozon_v1.py 
(Higgs globalna sprega + symmetry breaking, izlaz su kombinacije + lex-regioni, 
bez ikakvog frekvencijskog brojanja brojeva).
Izlaz će biti cele kombinacije i lex-regioni.

Podaci imaju jedan nekompletan red na kraju (1,9,13), pa loader ostavljam robustan: uzima samo validne redove sa 7 brojeva. Sad pravim bozon model kao novi fajl.

Qiskit + Aer simulator
25 qubita = 5 blokova x 5
BOZON/Higgs analog: peti 5q blok kao globalno polje
CRY coupling: Higgs blok → prva četiri bloka
symmetry breaking preko +/- Higgs VEV
MMD loss nad lex-distribucijom
exponential recency
conditional seed iz zadnjeg lex-indeksa
bez frekvencije brojeva: izlaz su cele kombinacije + lex-regioni
filter: bez placeholder-a i bez već izvučenih kombinacija
izlazi:
8_quant_bozon_v1.txt
8_quant_bozon_v1.png
"""
