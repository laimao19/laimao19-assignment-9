# Define your virtual environment and Flask app
VENV = venv
FLASK_APP = app.py

# Platform-specific settings
ifeq ($(OS),Windows_NT)
    PIP = $(VENV)\Scripts\pip
    PYTHON = $(VENV)\Scripts\python
    FLASK = $(VENV)\Scripts\flask
    SET_ENV = set
else
    PIP = $(VENV)/bin/pip
    PYTHON = $(VENV)/bin/python
    FLASK = $(VENV)/bin/flask
    SET_ENV = export
endif

# Install dependencies
install:
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Run the Flask application
run:
	$(SET_ENV) FLASK_APP=$(FLASK_APP) && $(SET_ENV) FLASK_ENV=development && $(FLASK) run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install
