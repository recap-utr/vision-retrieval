from __future__ import absolute_import, annotations

from typing import Dict, List, Callable
from torch.nn import functional as F
from .model import ImageEmbeddingGraph
import torch
from ranx import Run, Qrels, evaluate
import statistics
from time import time


class Evaluation(object):
    """Class for calculating and storing evaluation measures

    Candiates are fetched automatically from a file.
    The order of the candiates is not relevant for the calculations.
    """

    user_candidates: List[str]
    system_candidates: List[str]
    ground_truth_rankings: Dict[str, Dict[str, int]]
    embedding_func: Callable
    queries: List[ImageEmbeddingGraph]
    case_base: Dict[str, ImageEmbeddingGraph]

    def __init__(
        self, case_base: Dict[str, ImageEmbeddingGraph], ground_truth: Dict[str, Dict[str, int]], mac_results: Dict[str, List[str]], queries: List[ImageEmbeddingGraph], embedding_func: Callable[..., torch.Tensor], debug: bool = False
    ) -> None:
        self.k = 0
        self.start = time()
        self.case_base = case_base
        self.mac_results = mac_results
        self.queries = queries
        self.qrels = {}
        predicted_relevances = {}
        self.ground_truth_rankings = ground_truth
        for query in queries:
            max_value = int(max(ground_truth[query.name].values()))
            self.qrels[query.name] = {k: (max_value - int(v) + 1) for k, v in ground_truth[query.name].items()}
            predicted_relevances[query.name] = self._get_model_predictions(query)
        self.embedding_func = embedding_func
        self.duration = time() - self.start
        self.run = Run(predicted_relevances)
        self.debug = debug
        self._get_ndcg()

    def _get_model_predictions(self, query: ImageEmbeddingGraph)-> Dict[str, float]:
        query_embedding = query.embedding
        search_space = {k: v for k, v in self.case_base.items() if k in self.mac_results[query.name]}
        similiarities = {case.name: F.cosine_similarity(query_embedding, case.embedding, dim=-1).item() for case in search_space.values()}
        # sort similiarities desc
        similiarities = {k: v for k, v in sorted(similiarities.items(), key=lambda item: item[1], reverse=True)}
        return similiarities
    
    def _get_ndcg(self) -> None:
        evaluate(Qrels(self.qrels), self.run, ["ndcg_burges", "ndcg", "map", "f1", "recall", "precision"], return_mean=False)

    def _correctness_completeness(self, query) -> tuple[float, float]:
        key = query.name
        if self.debug:
            print(key)
        qrel = self.qrels[key]

        # The following produces a ranking in retrieval order (most similar result is shown first): 
        # most similar doc --> 1
        # second most similar doc --> 2
        # ...
        sorted_run = sorted(self.run[key].items(), key=lambda x: x[1], reverse=True)
        if self.debug:
            print(sorted_run)
        run_ranking = {x[0]: i + 1 for i, x in enumerate(sorted_run)}

        orders = 0
        concordances = 0
        disconcordances = 0

        correctness = 1
        completeness = 1

        for user_key_1, user_rank_1 in qrel.items():
            for user_key_2, user_rank_2 in qrel.items():
                if user_key_1 != user_key_2 and user_rank_1 > user_rank_2:
                    orders += 1

                    system_rank_1 = run_ranking.get(user_key_1)
                    system_rank_2 = run_ranking.get(user_key_2)

                    # if rel(doc1) > rel(doc2) then the following should hold for concordance: 
                    # similiarity(doc1) > similiarity(doc2) and rank(doc1) < rank(doc2)
                    if system_rank_1 is not None and system_rank_2 is not None:
                        if system_rank_1 < system_rank_2:
                            concordances += 1
                        elif system_rank_1 > system_rank_2:
                            disconcordances += 1

        if concordances + disconcordances > 0:
            correctness = (concordances - disconcordances) / (
                concordances + disconcordances
            )
        if orders > 0:
            completeness = (concordances + disconcordances) / orders
        if self.debug:
            print("orders", orders, "concordances", concordances, "disconcordances", disconcordances, "correctness", correctness, "completeness", completeness)

        return correctness, completeness
    
    def as_dict(self):
        results = self.run.mean_scores
        correctness, completeness = [], []
        for query in self.queries:
            corr, comp = self._correctness_completeness(query)
            correctness.append(corr)
            completeness.append(comp)
        results["correctness"] = statistics.mean(correctness)
        results["completeness"] = statistics.mean(completeness)
        results["duration"] = self.duration

        return results
    
    
    
    


    