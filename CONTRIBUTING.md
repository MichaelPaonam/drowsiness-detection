# Contributing to Drowsiness Detection

Thank you for contributing.

## Before You Start

* Check existing issues and pull requests before creating a new one.
* Open an issue first for large features or architectural changes.
* Keep pull requests focused on a single change.

---

## Development Setup

### Clone the Repository

```bash
git clone https://github.com/MichaelPaonam/drowsiness-detection.git
cd drowsiness-detection
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Coding Guidelines

### Python

* Follow PEP 8.
* Use meaningful variable and function names.
* Add docstrings for public functions.
* Avoid hardcoded paths and secrets.

### Machine Learning / Computer Vision

* Document model changes clearly.
* Include dataset references when applicable.
* Provide evaluation metrics for model improvements.
* Keep training and inference code separated when possible.

---

## Pull Request Process

1. Create a feature branch.

```bash
git checkout -b feature/short-description
```

2. Commit using clear messages.

```text
feat: add eye aspect ratio calibration
fix: correct webcam initialization issue
docs: update setup instructions
```

3. Push your branch.

```bash
git push origin feature/short-description
```

4. Open a Pull Request.

### PR Checklist

* [ ] Code builds successfully
* [ ] Tests pass
* [ ] Documentation updated if needed
* [ ] No unnecessary files included
* [ ] Related issue linked

---

## Reporting Bugs

Include:

* Operating system
* Python version
* Steps to reproduce
* Expected behavior
* Actual behavior
* Logs or screenshots if available

---

## Feature Requests

Provide:

* Problem statement
* Proposed solution
* Alternative approaches considered
* Expected impact

---

## Code of Conduct

Be respectful and constructive during discussions and reviews.

Thank you for helping improve this project.
