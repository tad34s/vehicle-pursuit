# Makefile

.PHONY: check install-hooks

# Run pre-commit checks (ruff, isort, codespell)
check: ## Run all checks via pre-commit
	@echo "Running pre-commit checks..."
	uv run pre-commit run  --all-files

# Install pre-commit hooks into git
install-hooks: ## Install pre-commit hooks
	pre-commit install

train:
	uv run src/train.py

test:
	uv run src/test.py models/leader.onnx -f models/follower.onnx 

dataset:
	uv run src/test.py models/leader.onnx -f models/follower.onnx -c -d; uv run src/depth_net/utils/create_masks.py
