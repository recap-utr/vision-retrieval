from __future__ import absolute_import, annotations

from typing import Dict, List
from torch.nn import functional as F
from model import ImageEmbeddingGraph
from ranx import Run, Qrels, evaluate
import statistics
from time import time
from correctness_completeness import _correctness_completeness_single


class Evaluation:
    """Class for calculating and storing evaluation measures

    Candiates are fetched automatically from a file.
    The order of the candiates is not relevant for the calculations.
    """

    user_candidates: List[str]
    system_candidates: List[str]
    ground_truth_rankings: Dict[str, Dict[str, int]]
    queries: List[ImageEmbeddingGraph]
    case_base: Dict[str, ImageEmbeddingGraph]
    debug: bool
    times: bool

    def __init__(
        self,
        case_base: Dict[str, ImageEmbeddingGraph],
        ground_truth: Dict[str, Dict[str, int]],
        mac_results: Dict[str, List[str]],
        queries: List[ImageEmbeddingGraph],
        debug: bool = False,
        times: bool = False,
    ) -> None:
        self.k = 0
        self.start = time()
        self.case_base = case_base
        self.mac_results = mac_results
        self.queries = queries
        self.debug = debug
        self.times = times
        self.qrels = {}
        predicted_relevances = {}
        self.ground_truth_rankings = ground_truth
        for query in queries:
            max_value = int(max(ground_truth[query.name].values()))
            self.qrels[query.name] = {
                k: (max_value - int(v) + 1) for k, v in ground_truth[query.name].items()
            }
            predicted_relevances[query.name] = self._get_model_predictions(query)
        self.duration = time() - self.start
        self.run = Run(predicted_relevances)
        self._get_ndcg()

    def _get_model_predictions(self, query: ImageEmbeddingGraph) -> Dict[str, float]:
        query_embedding = query.embedding
        search_space = {
            k: v for k, v in self.case_base.items() if k in self.mac_results[query.name]
        }
        start = time()
        similiarities = {
            case.name: F.cosine_similarity(
                query_embedding, case.embedding, dim=-1
            ).item()
            for case in search_space.values()
        }
        # sort similiarities desc
        similiarities = {
            k: v
            for k, v in sorted(
                similiarities.items(), key=lambda item: item[1], reverse=True
            )
        }
        if self.times:
            print(
                f"Processed {len(similiarities)} candidates (similarities) in {time()-start} seconds"
            )
        return similiarities

    def _get_ndcg(self) -> None:
        evaluate(
            Qrels(self.qrels),
            self.run,
            ["ndcg_burges", "ndcg", "map", "f1", "recall", "precision"],
            return_mean=False,
        )

    

    def as_dict(self):
        results = self.run.mean_scores
        correctness, completeness = [], []
        for query in self.queries:
            corr, comp = _correctness_completeness_single(
                self.qrels[query.name], self.run[query.name], self.k)
            correctness.append(corr)
            completeness.append(comp)
        results["correctness"] = statistics.mean(correctness)
        results["completeness"] = statistics.mean(completeness)
        results["duration"] = self.duration

        return results
