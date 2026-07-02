# Set-up
Make sure repo is cloned using `git clone git@github.com:BenLoweryTM/stock-control.git` 

I think you just need to do the following (in storm)
1. Install UV (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
2. Install the wheels `uv add ./wheels/inventorygyms-0.3.0-py3-none-any.whl` and `uv add ./wheels/one_period_shortage-0.1.3-cp312-cp312-linux_x86_64.whl`
3. Check it all works by running `uv run main.py`