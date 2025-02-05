# Installation

Clone the repository, create a virtual environment preferably with python3.12, and run:
```bash
pip install -r requirements.txt
```
(In case of problems, try running `pip install -r .pip_freeze.txt` instead, to install the exact tested versions.)

# Running MUDMAN

```bash
python src/MUDMAN.py
```

It contains a simple example of unlearning Llama-3.2-1B on the Pile-Bio dataset and then relearning back.

