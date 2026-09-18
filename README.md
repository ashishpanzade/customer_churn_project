# Customer Churn Prediction

Customer churn prediction solution using the IBM Telco Customer Churn dataset, a Decision Tree model, and a FastAPI REST API.

## Prerequisites

* Python 3.10+
* Windows
* Command Prompt or PowerShell

## Setup

Open a terminal in the project root:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the API

Start the FastAPI application:

```powershell
uvicorn app:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Test the API

Open `http://localhost:8000/docs` and use the Swagger UI to try `/predict` and `/health`
directly in the browser. Use `sample_request.json` as the body for `/predict`.

Example `/predict` response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.689
}
```

Expected `/health` response:

```json
{
  "status": "ok"
}
```

## Notebook

To open the existing analysis notebook:

```powershell
jupyter notebook notebook\my_churn_analysis.ipynb
```

The notebook contains the data analysis, feature engineering, model training, and evaluation.

## Stop

Press:

```text
CTRL + C
```

To deactivate the virtual environment:

```powershell
deactivate
```
