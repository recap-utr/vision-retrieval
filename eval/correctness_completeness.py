from typing import Mapping
import statistics

# from https://github.com/wi2trier/cbrkit/blob/main/cbrkit/eval/_common.py


def correctness_completeness(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    k: int | None = None,
) -> tuple[float, float]:
    keys = set(qrels.keys()).intersection(set(run.keys()))

    scores = [_correctness_completeness_single(qrels[key], run[key], k) for key in keys]
    correctness_scores = [score[0] for score in scores]
    completeness_scores = [score[1] for score in scores]

    try:
        return statistics.mean(correctness_scores), statistics.mean(completeness_scores)
    except statistics.StatisticsError:
        return float("nan"), float("nan")


def _correctness_completeness_single(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
) -> tuple[float, float]:
    sorted_run = sorted(run.items(), key=lambda x: x[1], reverse=True)
    run_k = {x[0]: x[1] for x in sorted_run[:k]}

    concordant_pairs = 0
    discordant_pairs = 0
    total_pairs = 0

    case_keys = list(qrel.keys())

    for i in range(len(case_keys)):
        for j in range(i + 1, len(case_keys)):
            idx1, idx2 = case_keys[i], case_keys[j]
            qrel1, qrel2 = qrel[idx1], qrel[idx2]

            if qrel1 != qrel2:
                total_pairs += 1

                if idx1 in run_k and idx2 in run_k:
                    run1, run2 = run_k[idx1], run_k[idx2]

                    if (qrel1 < qrel2 and run1 < run2) or (
                        qrel1 > qrel2 and run1 > run2
                    ):
                        concordant_pairs += 1
                    else:
                        discordant_pairs += 1

    correctness = (
        (concordant_pairs - discordant_pairs) / (concordant_pairs + discordant_pairs)
        if (concordant_pairs + discordant_pairs) > 0
        else 0.0
    )

    completeness = (
        (concordant_pairs + discordant_pairs) / total_pairs if total_pairs > 0 else 0.0
    )

    return correctness, completeness
