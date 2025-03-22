# Medical Assistant Bot

A medical question-answering system capable of effectively answering user queries related to medical diseases.

## Project Overview

This project implements a medical assistant chatbot using Gemini for dataset evaluation, Langchain for orchestration, and PEFT (Parameter-Efficient Fine-Tuning) for adapting a Deepseek model to medical question answering tasks.

## Key Features

- Data preprocessing using Gemini to filter high-quality QA pairs
- Fine-tuning of the Deepseek-1.3 model using LoRA (Low-Rank Adaptation)
- Comprehensive model evaluation using LangSmith
- Containerized deployment on Google Cloud Run
- API endpoint for medical question answering

## Assumptions

1. The original dataset's quality is low and contains a mix of correct and incorrect QA pairs that need to be filtered
2. The medical domain requires specialized knowledge, making a general LLM insufficient without fine-tuning
3. Model size constraints require parameter-efficient training methods
4. Evaluation metrics should prioritize factual accuracy and helpfulness


## Quick Start

### Prerequisites

- Python 3.9+
- Docker (for containerized deployment)

### Installation

1. Create a Python environment:
   ```
   python3 -m venv medical-env
   . medical-env/bin/activate
   ```

2. Clone the repository:
   ```
   git clone https://github.com/your-username/medical-assistant-bot.git
   cd medical-assistant-bot
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   # Edit .env file with your API keys
   ```

### Running Locally

1. Run the data preprocessing and training:
   ```
   python -m src.main train --dataset /path_to_dataset.csv
   ```

2. Run the evaluation:
   ```
   python -m src.evaluate_model
   ```

3. Run local inference:
   ```
   python -m src.evaluate_model
   ```

4. For the API server:
   ```
   uvicorn src.main:app --reload
   ```

### Deployment

To deploy on Google Cloud Run:

1. Build the Docker image:
   ```
   docker build -t gcr.io/[PROJECT_ID]/medical-assistant-bot:latest .
   ```

2. Push to Google Container Registry:
   ```
   docker push gcr.io/[PROJECT_ID]/medical-assistant-bot:latest
   ```

3. Deploy to Cloud Run:
   ```
   gcloud run deploy medical-assistant-bot \
     --image gcr.io/[PROJECT_ID]/medical-assistant-bot:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## Project Structure

```
medical-assistant-bot/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_processor.py     # Data preprocessing class
│   │   └── data_loader.py        # Data loading utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py          # Model evaluation class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py         # Abstract base model class
│   │   └── deepseek_model.py     # Deepseek model implementation
│   └── main.py                   # Main application entry point
└── .env.example                  # Example environment variables
```

## To Do

1. Enhance data preprocessing with data correction and filtering
2. Incorporate medical ontologies for better entity recognition
3. Add specialized modules for different medical subdomains
4. Increase model size or explore more parameter-efficient techniques
5. Implement a feedback loop system for continuous improvement

## Screenshots/Diagrams

![Model training](images/image1.png)