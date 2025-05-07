# Overview

Book Rating Prediction API is a FastAPI-based service that predicts a numerical rating (e.g., 1 to 5) based on a given user review of a book. It uses a fine-tuned ModernBERT model exported to ONNX for efficient inference. The API provides endpoints for health checks and real-time rating predictions from raw text inputs.

## Project Structure
<pre> ```Review-Rating-Prediction-API/
├── app/
│   ├── __init__.py
│   ├── BERT_regressor.py       # Loads the ONNX model and handles predictions
│   ├── config.py			    # Contains configuration
│   ├── exception_handler.py   #exception handling logic for FastAPI
│   ├── main.py # FastAPI entry point with endpoints
│   ├── model.py # Defines the machine learning model wrapper
│   ├── schema.py # Pydantic data schemas for request and response validation
│   └── example/
│       └── client.py # Example client for testing the API
├── notebooks/
│   ├── convert_onnx.ipynb # Converts the trained PyTorch model to ONNX format
│   ├── expand_dataset.ipynb # Data augmentation
│   ├── ML.ipynb # Classic ML model experiments
│   └── ModernBERT.ipynb # Fine-tuning and evaluation of the ModernBERT 
├── model/
│   ├── bert_regressor.onnx # Exported ONNX model used for inference
│   └── tokenizer/ # Tokenizer files corresponding to the ModernBERT 
├── requirements.txt # Python dependencies for the project
├── .gitignore
└── README.md``` </pre>

## Installation

### - Setup Python Virtual Environment
- `python -m venv env`

### - Activate The Environment
- Windows
  `.\.venv\Scripts\Activate.ps1`
 - macOS/Linux
 `source .venv/bin/activate`

### - Install Requirements
`pip install -r requirements.txt`

### - Configure the model
Set the `GLOBAL_CONFIG` variable as follows `config.py`:
<pre><code><details> 
GLOBAL_CONFIG = { 
"MODEL_PATH": "../model/bert_regressor.onnx", 
"BERT_MODEL": "answerdotai/ModernBERT-base", 
"MAX_LEN": 2048, 
"DEVICE": "cpu" 
}  </details> </code></pre>

### - Run the API
`uvicorn main:app --reload`


## How to use API?
### - Swagger UI

> FastAPI automatically provides a Swagger-based user interface. You can
> test the API using this interface

 1. Start the API server:
 `uvicorn main:app --reload`
 2. Go to the following address from your browser:
 `http://localhost:8000/docs`
 3. From the Swagger interface: 
	 -You can see if the model is ready by testing the `/health` endpoint.
	 -Select the `/predict` endpoint, enter a text in the reviewText field and press 	  ‘Execute’ to get the prediction.

### - CMD

>   You can send requests to API endpoints using the curl command from the command line.
- `/predict` Endpoint
	- <pre>curl -X POST http://localhost:8000/predict \ -H "Content-Type: application/json" \ -d "{\"reviewText\": \"This book is amazing!\"}" </pre>
- `/health` Endpoint
	- <pre>curl http://localhost:8000/health</pre>

### - Powershell
> You can send a request to the API via PowerShell.
- <pre>(Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method POST -Headers @{ "Content-Type" = "application/json" } -Body '{"reviewText":"This book is amazing!').Content | ConvertFrom-Json | Select-Object -ExpandProperty rating</pre>

### - By Code
> You can call the API programmatically using Python and the `requests` library:
1. Start the API server:
 `uvicorn main:app --reload`
2. Run `python client.py`

## API Outputs and Endpoints
### Health Check
- **URL:** `/health`
- **Method:** GET
- **Description:** It is used to check whether the model is loaded.
- **Example Request:**
`curl http://localhost:8000/health`
- **Success Response:**
Status Code: 200 OK
`{
"rating": 0.0
}`
- **Error:**
<pre><code>{ 
"error": true, 
"message": "ONNX model session not initialized", 
"traceback": "..." 
}  </code></pre>

### Prediction
- **URL:** `/predict`
- **Method:**: POST
- **Description:** Estimates a score based on the given review text.
- - **Example Request:** 
<pre> curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"reviewText\": \"This product is amazing! The quality is top notch.\"}"</pre>

- **Success Response:** 
Status Code: 200 OK
<pre><code>{
"rating": 5.0
} </code></pre>

- **Error:**
400 Bad Request: If the review text is missing or incorrect.
<pre><code>{
"error": true,
"message": "Invalid request data. Please check the 'reviewText' field."
}</code></pre>


## Error Management

 - **Validation Errors**
> FastAPI automatically throws this error when the API client sends invalid data (for example, a missing or incorrect type of field).

Example Cases:
- If the reviewText field is missing in the POST /predict request
- If the wrong data type was sent (e.g. reviewText: int)
---
- **Internal Server Error**

>   Failure to load the model while the application is running, NoneType errors, exception in the predict function of the model, unexpected Python errors such as file access.

Example Cases:
- If there is no model file
- model_wrapper.session does not exist (NoneType error)
- if there is an error inside the predict() function (e.g. the tokeniser is not loaded properly)
---
- **FileNotFoundError**
>   Thrown if the file being accessed on the file system does not exist.

Example Cases: 
- File not found when trying to load model file (Model(model_path=...)).
- File path access error received from the user
---
- **TimeoutError**

>   This error occurs if the asynchronous operation is not completed within a certain time.

Example Cases:
- An estimate or data processing that takes too long.
- Connection timeout when receiving data from external services.
---
- **RuntimeError**

>  The program encounters an unexpected situation at runtime.

Example Cases:
- The model could not be initialised (model_wrapper.session is None).
- The ONNX session could not be created or there is missing configuration.
