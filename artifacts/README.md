# Artifacts

Generated model artifacts and scored transactions are intentionally excluded
from Git because they are reproducible outputs.

Generate them using:

python -m abusering.evaluation.run_experiments \
  --data data_store \
  --artifacts artifacts \
  --reports reports \
  --cost-config configs/cost_config.json \
  --optuna-trials 20 \
  --seed 42