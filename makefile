python = venv/bin/python
pip = venv/bin/pip

setup:
	python3 -m venv venv
	$(python) -m pip install --upgrade pip 
	$(pip) install -r requirements.txt

run:
	$(python) main.py

clean:
	@if exist steps\__pycache__ (rmdir /s /q steps\__pycache__)
	@if exist __pycache__ (rmdir /s /q __pycache__)


remove:
	@if exist venv (rmdir /s /q venv)