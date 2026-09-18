# Environment setup:

## Virtual environment:

```
python -m venv .venv
source .venv/bin/activate
pip install ipykernel
which python
deactivate
pip install -r requirements.txt
pip freeze > requirements.txt

```

## Jupyter kernel management:

```
pip install ipykernel
pip install --upgrade pip
python -m ipykernel install --user --name rdm-kernel --display-name="rdm kernel"

jupyter kernelspec list
jupyter kernelspec uninstall rdm-kernel

