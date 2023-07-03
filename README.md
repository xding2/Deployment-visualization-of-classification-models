# Deployment-visualization-of-classification-models

## Hate Speech Detection Web App
This is a demo application built using Flask, OpenAI's GPT-3 and Huggingface's transformers library. It utilizes three different models for hate speech detection, providing an insightful comparison of their outputs. The first two models are BERT and RoBERTa-based models that are trained to classify hate speech. And the token-level interpretation of classification model results can be performed by _transformers_interpret_ [https://github.com/cdpierse/transformers-interpret]. The third model is OpenAI's GPT-3, used to interpret the result.

### Prerequisites
To use this web application, you'll need:

- Python (3.7 or above)
- Flask
- Transformers library from Hugging Face
- OpenAI API key
  
Additionally, make sure to have GPU support if you're planning to use this application on a large scale as model loading and inference could be resource-intensive.

### How to Use
1. Clone the Repository

First, clone this repository to your local system using git.

2. Setup Environment

It's recommended to setup a Python virtual environment before proceeding. You can use the venv module for that.

```python
python3 -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```


