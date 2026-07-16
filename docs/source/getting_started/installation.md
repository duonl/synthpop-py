# Installation

This page explains how to install synthpop-py.

For most users, the recommended installation method is through PyPI. If you want to contribute to synthpop-py or use the latest development version, see the source installation instructions.

## Requirements

Synthpop-py requires:

- Python 3.13 or later
- `pip`

We recommend installing synthpop-py in a virtual environment to keep dependencies isolated from other Python projects.

## Create a virtual environment

```bash
# Create a virtual environment
python -m venv YOUR_VENV_NAME

# Activate it:
# Linux/macOS
source YOUR_VENV_NAME/bin/activate

# Windows (Command Prompt)
YOUR_VENV_NAME\Scripts\activate

# Windows (PowerShell)
YOUR_VENV_NAME\Scripts\Activate.ps1
```
After activation, install synthpop-py as described below.

---

## Install synthpop-py from PyPI

The recommended installation method is:
```bash
pip install synthpop-py
```
Alternatively:
```bash
python -m pip install synthpop-py
```

---

## Verify the installation

Check that synthpop-py can be imported successfully:
```bash
python -c "import synthpop-py"
```
Or start Python:
```python
import synthpop-py
```
If no errors occurs, synthpop-py has been installed successfully.

---

## Install from source

Installing from source if useful if you want to:
- contribute to synthpop-py;
- test unreleased features;
- modify the package locally.

Clone the repository and install the package:

```bash
git clone https://github.com/duonl/synthpop-py.git
cd synthpop-py
python -m pip install .
```
For development setup instructions, including documentation building and development dependencies, see the [developer documentation](../developer/way_of_working/developing.md).

---

## Updating synthpop-py

Upgrade to the latest release:

```bash
python -m pip install --upgrade synthpop-py
```

If you installed from source:

```bash
git pull
python -m pip install .
```

---

## Troubleshooting

If you encounter installation problems:
- Confirm that you are using Python 3.13 or newer.
- Ensure your virtual environment is activated.
- Check that `pip` is associated with the Python version you are using:
```bash
python -m pip --version
```
