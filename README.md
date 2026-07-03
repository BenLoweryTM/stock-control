# Set-up
Make sure repo is cloned using `git clone git@github.com:BenLoweryTM/stock-control.git` 

Ensure the following `pyproject.toml` file exists, this is for STORM we need to precompile some linux wheels. On other machines a simple pull from github of the three `tool.uv.sources` would sufice.
```python
[project]
name = "stock-control"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12,<3.14"
dependencies = [
    "inventorygyms",
    "one-period-shortage",
    "optimalpolicy",
    "pyarrow>=24.0.0",
]

[tool.uv.sources]
inventorygyms = { path = "wheels/inventorygyms-0.3.0-py3-none-any.whl" }
one-period-shortage = { path = "wheels/one_period_shortage-0.1.3-cp312-cp312-linux_x86_64.whl" }
optimalpolicy = { path = "wheels/optimalpolicy-0.2.4-cp39-abi3-linux_x86_64.whl" }
```
