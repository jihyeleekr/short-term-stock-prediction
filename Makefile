PYTHON = python
PIP = pip

.PHONY: install test preprocess stock-plots baseline cross-logistic random-forest gradient-boosting compare reproduce clean

install:
	$(PIP) install -r requirements.txt

test:
	PYTHONPATH=.:src pytest -q

preprocess:
	$(PYTHON) src/preprocessing.py

stock-plots:
	$(PYTHON) src/plot_stock_visualizations.py

baseline:
	$(PYTHON) src/train_classification.py

cross-logistic:
	$(PYTHON) src/train_cross_asset_model.py

random-forest:
	$(PYTHON) src/train_cross_asset_random_forest.py

gradient-boosting:
	$(PYTHON) src/train_cross_asset_gradient_boosting.py

compare:
	$(PYTHON) src/compare_models.py
	$(PYTHON) src/plot_train_val_test_comparison.py

interactive:
	$(PYTHON) src/plot_interactive_visualizations.py

reproduce: install preprocess stock-plots baseline cross-logistic random-forest gradient-boosting compare interactive
clean:
	rm -rf outputs/figures/*
	rm -rf outputs/metrics/*
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +