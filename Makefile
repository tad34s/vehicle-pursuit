# Makefile

.PHONY: check install-hooks

# Run pre-commit checks (ruff, isort, codespell)
check: ## Run all checks via pre-commit
	@echo "Running pre-commit checks..."
	pre-commit run  --all-files

# Install pre-commit hooks into git
install-hooks: ## Install pre-commit hooks
	pre-commit install

train:
	uv run src/train.py
