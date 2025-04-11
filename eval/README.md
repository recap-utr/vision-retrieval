# Evaluation
In this directory, the code for our evaluation of our various vision models and the fine-tuned GPT-4o model can be found.
- `new_evaluation.py` contains the implementation of our evaluation for our vision models, but gets configured and launched from `evaluate_torch_models.py`.
- `eval_cli.py` provides a CLI to evaluate fine-tuned GPT-4o models (eval-oai command) and pre-trained and fine-tuned torch models (eval-torch) on argumentation graphs.
- `model.py` and `correctness_completeness.py` contain helpers for our evaluation logic.
- `results.py` is a utility script to summarize results obtained through running `evaluate_torch_models.py`.