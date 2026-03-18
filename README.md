# Synthpop

This package is currently being developed. It mostly contains placeholders and documentation.

## What is synthpop?
Synthpop is a package to generate synthetic data for use in data analysis. The package exists in both R and python.
The aim is to generate synthetic data that is as close to the observed data as possible without disclosing the actual data.
This allows for analysis of sensitive data while adhering to the GDPR. 

The [R package](https://www.synthpop.org.uk/) has been used some time for this purpose.

## The goals of this project
The goal of this project is to make it easier to generate synthetic data for data analysis. 
For now, we aim to make it easier for two groups of people.
One group are people who are just starting out with synthetic data.
The other group are people that want to provide synthetic data to third parties or want to include synthetic data in their standard toolbox.

## General roadmap
The first aim is to make a package that is comparable to synthpop. 
This should attract enough active users to decide what the next aim should be.

## Contributing

If you wish to contribute to synthpop, start by cloning the repository.

### Prerequisites

The project uses **pipx** and **Poetry** for dependency management. Follow the steps below after cloning the repository.

1. Open a terminal in the root of the repository.
2. Install pipx: ```py -m pip install --user pipx```
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

You can close the terminal.

### Updating the documentation

After making changes to the documentation, rebuild the HTML output by running the following commands:

1. Open a terminal in the root of the repository:
2. Navigate to the **docs** directory: ```cd docs``` 
3. Delete current documentation files: ``` poetry run sphinx-build -M clean source build```
4. Build the new documentation: ``` poetry run sphinx-build source build```
