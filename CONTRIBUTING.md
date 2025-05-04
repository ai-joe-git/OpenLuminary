# Contributing to OpenLuminary

Thank you for your interest in contributing to OpenLuminary! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported in our [Issues](https://github.com/ai-joe-git/OpenLuminary/issues)
- Use the bug report template to create a new issue
- Include detailed steps to reproduce the bug
- Include any relevant logs or screenshots

### Suggesting Features

- Check if the feature has already been suggested in our [Issues](https://github.com/ai-joe-git/OpenLuminary/issues)
- Use the feature request template to create a new issue
- Describe the feature in detail and why it would be valuable

### Code Contributions

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Write tests for your changes
4. Ensure all tests pass
5. Submit a pull request

### Documentation

- Help improve our documentation
- Fix typos or clarify explanations
- Add examples or tutorials

## Development Setup

1. Clone the repository:
git clone https://github.com/ai-joe-git/OpenLuminary.git
cd OpenLuminary

text

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

text

3. Install development dependencies:
pip install -e ".[dev]"

text

4. Install pre-commit hooks:
pre-commit install

text

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Add type hints to function signatures
- Keep functions focused on a single responsibility
- Write unit tests for new functionality

## Testing

Run tests with pytest:

pytest

text

Run tests with coverage:

pytest --cov=src

text

## Pull Request Process

1. Update the README.md or documentation with details of changes if appropriate
2. Update the CHANGELOG.md with a description of your changes
3. The PR should work for Python 3.9 and above
4. PRs require review from at least one maintainer
5. Make sure all CI checks pass

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

## License

By contributing to OpenLuminary, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
