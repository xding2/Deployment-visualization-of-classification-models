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

3. Install Dependencies

After activating the virtual environment, you need to install necessary dependencies. The main dependencies are Flask and transformers.

```python
pip install flask transformers openai transformers_interpret
```

4. Set OpenAI API Key

Replace # change to your openai key here "sk-##########################" in the code with your actual OpenAI API key. Please do not share your OpenAI API key publicly.

5. Start the Application

You can start the application by running the python script.

```python
python3 run demo.py
```

6. Use the Web App

The demo application has an input field where you can enter text to classify. After you submit the text, it gets classified by the three models, and the results are displayed on the page. The result includes:

- Classification labels from the BERT, RoBERTa, and GPT-3 models.
- Confidence scores from the models.
- Important words considered by the GPT-3 model.

## Reference

ToxVis: Enabling Interpretability of Implicit vs. Explicit Toxicity Detection Models with Interactive Visualization: [https://arxiv.org/pdf/2303.09402.pdf]

```
@article{gunturi_toxvis_2023,
	title = {{ToxVis}: Enabling Interpretability of Implicit vs. Explicit Toxicity Detection Models with Interactive Visualization},
	author = {Gunturi, Uma and Ding, Xiaohan and Rho, Eugenia H.},
	journal = {arxiv preprint arXiv::2303.09402},
  year={2023},
}
```


