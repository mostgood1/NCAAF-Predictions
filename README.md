# NCAA Football Predictability Modeling

This project provides tools for ingesting NCAA football data, building predictive models, and displaying results via a simple Flask web interface.

## Features
- Data ingestion from CSV, web scraping, or API
- Modeling for winner prediction, spread coverage, and team totals (quarter/half/game)
- Flask web interface to display predictions and model factors

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python main.py
   ```

## Usage
- Place your data files in the `src/data` directory.
- Configure settings in `config/settings.py`.
- Access the web interface at `http://localhost:5000` after starting the app.

## Directory Structure
```
NCAFCompare/
├── main.py
├── requirements.txt
├── README.md
├── config/
│   └── settings.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── ingestion.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   └── predict.py
│   └── web/
│       ├── __init__.py
│       └── app.py
```
