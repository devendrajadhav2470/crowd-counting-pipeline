.PHONY: install install-dev train evaluate demo test lint format clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install with all dev dependencies
	pip install -e ".[all]"
	pre-commit install

train: ## Train CSRNet (use CONFIG=configs/shb.yaml to override)
	python scripts/train.py --config $(or $(CONFIG),configs/shb.yaml)

evaluate: ## Evaluate model on test set
	python scripts/evaluate.py --config $(or $(CONFIG),configs/shb.yaml) --checkpoint $(or $(CHECKPOINT),weights/best_model.pth)

infer: ## Run inference on a single image (use IMAGE=path/to/image.jpg)
	python scripts/infer.py --image $(IMAGE) --checkpoint $(or $(CHECKPOINT),weights/best_model.pth)

demo: ## Launch Gradio demo
	python -m crowd_counting.serving.gradio_app

export-onnx: ## Export model to ONNX format
	python scripts/export_onnx.py --checkpoint $(or $(CHECKPOINT),weights/best_model.pth)

test: ## Run unit tests
	pytest tests/ -v

lint: ## Run linter
	ruff check src/ scripts/ tests/

format: ## Format code with black and ruff
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image
	docker build -t crowd-counting .

docker-run: ## Run Docker container with Gradio demo
	docker run -p 7860:7860 crowd-counting

