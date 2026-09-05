# Data Store

Generated datasets and feature stores are intentionally excluded from Git
because they are reproducible artifacts.

Generate them using:

python -m abusering.data.generate --profile demo --seed 42 --out data_store

python -m abusering.features.build_cli --data data_store --out data_store

python -m abusering.splits.ring_split --data data_store --out data_store --seed 42