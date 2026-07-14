# Installation

This guide describes how to install Synthpop using either the latest stable release from PyPI or from the source repository.

For the recommended setup, install Synthpop inside a virtual environment to keep dependencies isolated and avoid conflicts with other Python projects.


## Requirements

Synthpop requires the following:

- Python 3.13 or later
- `pip`

We strongly recommend installing Synthpop inside a virtual environment to avoid dependency conflicts.

```bash
# Create a virtual environment
python -m venv YOUR_VENV_NAME

# Activate it on Linux/macOS
source YOUR_VENV_NAME/bin/activate

# Activate it on Windows (Command Prompt)
YOUR_VENV_NAME\Scripts\activate

# Activate it on Windows (PowerShell)
YOUR_VENV_NAME\Scripts\Activate.ps1
```

Once the virtual environment is active, proceed with the Synthpop installation as described below.

---

## Install from PyPI

Install the latest stable release using `pip`:

```bash
pip install synthpop
```

Or:

```bash
python -m pip install synthpop
```

---

## Install from Source

Clone the repository and install the package locally:

```bash
git clone https://github.com/duonl/synthpop-py.git
cd synthpop-py
python -m pip install .
```

## For development

```bash
git clone https://github.com/duonl/synthpop-py.git
cd synthpop-py
```

The project uses **pipx** and **Poetry** for dependency management. Follow the steps below after cloning the repository.

1. Open a terminal in the root of the repository.
2. Install pipx: ```python -m pip install --user pipx```
3. Navigate to Scripts folder (the exact path is shown in the warning from the previous command): ```cd <USER folder>\AppData\Roaming\Python\Python<VERSIE>\Scripts```. 
4. Add pipx to your $PATH: ```run .\pipx.exe ensurepath```
5. Close the terminal and open a new one (anywhere).
6. Verify the installation: ```pipx --version```
7. Install poetry: ```pipx install poetry```
8. Close the terminal and open a new one (anywhere).
9. Verify the installation: ```poetry --version```

You can close the terminal.

### Install project dependencies

1. Open a terminal in the root of the repository.
2. Install all dependencies: ```poetry install --with=docs```
3. Navigate to the **docs** directory: ```cd docs``` 
4. Build the documentation: ```poetry run sphinx-build source build```

---

## Verify the Installation

Verify that Synthpop is installed correctly:

```bash
python -c "import synthpop"
```

Or start a Python session:

```python
import synthpop
```

If no errors are raised, the installation was successful.

---

## Updating

Upgrade to the latest release with:

```bash
python -m pip install --upgrade synthpop
```

If you installed from source:

```bash
git pull
python -m pip install .
```

---

## Troubleshooting

If you encounter installation issues:

- Ensure you are using Python 3.13 or newer.
- If using a virtual environment, ensure it is activated.
