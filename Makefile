.DEFAULT_GOAL := help
PYTHON        := python

# Sentinel files track whether each stage has run so Make can skip up-to-date stages.
BRONZE_SENTINEL  := data/bronze/.done
SILVER_SENTINEL  := data/silver/.done
GOLD_SENTINEL    := data/gold/.done
FEATURE_SENTINEL := data/gold/ml_vessel_delay_features.parquet
MODEL_SENTINEL   := models/random_forest.joblib

.PHONY: help install generate-data bronze silver gold features train quality test run-all clean

help:
	@echo ""
	@echo "Port Operations Analytics Lakehouse"
	@echo "------------------------------------"
	@echo "  make install        Install Python dependencies"
	@echo "  make generate-data  Generate synthetic source CSVs"
	@echo "  make bronze         Ingest CSVs into Bronze Parquet layer"
	@echo "  make silver         Clean, deduplicate, and flag Bronze → Silver"
	@echo "  make gold           Aggregate Silver → Gold KPI tables"
	@echo "  make features       Build PIT-correct ML feature table"
	@echo "  make train          Train models and write metrics/importance"
	@echo "  make quality        Run 14-check data quality suite"
	@echo "  make test           Run pytest test suite (106 tests)"
	@echo "  make run-all        Full pipeline from scratch"
	@echo "  make clean          Remove all generated data and outputs"
	@echo ""

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------

install:
	pip install -r requirements.txt

# ----------------------------------------------------------------------------
# Pipeline stages
# ----------------------------------------------------------------------------

generate-data: data/raw/.done

data/raw/.done:
	$(PYTHON) src/generate_data.py
	@touch $@

bronze: $(BRONZE_SENTINEL)

$(BRONZE_SENTINEL): data/raw/.done
	$(PYTHON) src/bronze_ingestion.py
	@touch $@

silver: $(SILVER_SENTINEL)

$(SILVER_SENTINEL): $(BRONZE_SENTINEL)
	$(PYTHON) src/silver_transformations.py
	@touch $@

gold: $(GOLD_SENTINEL)

$(GOLD_SENTINEL): $(SILVER_SENTINEL)
	$(PYTHON) src/gold_kpis.py
	@touch $@

features: $(FEATURE_SENTINEL)

$(FEATURE_SENTINEL): $(GOLD_SENTINEL)
	$(PYTHON) src/feature_engineering.py

train: $(MODEL_SENTINEL)

$(MODEL_SENTINEL): $(FEATURE_SENTINEL)
	$(PYTHON) src/train_model.py

quality: $(SILVER_SENTINEL) $(GOLD_SENTINEL) $(FEATURE_SENTINEL)
	$(PYTHON) src/data_quality_checks.py

test:
	pytest tests/ -v

# ----------------------------------------------------------------------------
# Composite targets
# ----------------------------------------------------------------------------

run-all: install generate-data bronze silver gold features train quality test
	@echo ""
	@echo "Pipeline complete."
	@echo "  outputs/model_metrics.json      — model evaluation"
	@echo "  outputs/feature_importance.csv  — feature importance"
	@echo "  outputs/data_quality_report.json — DQ check results"
	@echo ""

# ----------------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------------

clean:
	rm -rf data/raw data/bronze data/silver data/gold data/quarantine
	rm -rf models outputs
	@echo "Cleaned all generated data and outputs."
