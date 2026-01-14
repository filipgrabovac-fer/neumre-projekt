Installation

macOS

curl -LsSf https://astral.sh/uv/install.sh | sh


windows

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.9.25/install.ps1 | iex"


Commands to run after you pulled the GitHub repository:

uv venv
source .venv/bin/activate
uv sync

- if VSCode asks which python interpreter to use, select the current virtual environment (it should have a star next to it)
- if it doesn't, press ctrl + shift + P (shift + command + P) and type (Python: Select Interpreter) and then select the environment