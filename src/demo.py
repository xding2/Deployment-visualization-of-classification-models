from flask import Flask, request, render_template
# from urllib import request
from transformers import pipeline

from random import sample
import os
import openai,torch

from random import sample
import openai
import os
import transformers
from transformers import XLNetTokenizer,WEIGHTS_NAME,CONFIG_NAME,XLNetForSequenceClassification,AdamW,XLNetModel
import torch.utils.data as Data

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import MultiLabelClassificationExplainer

import json
import ast

app = Flask(__name__)
model = pipeline("question-answering", model="warrior1127/my_awesome_qa_model")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/')
def index():  # put application's code here
    return render_template("hate.html")

@app.route('/hate', methods=['GET', 'POST'])
def hate():
    model_name1 = "warrior1127/hate_bert_model"
    model1 = AutoModelForSequenceClassification.from_pretrained(model_name1)
    tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
    cls_explainer1 = MultiLabelClassificationExplainer(model1, tokenizer1)

    model_name2 = "warrior1127/hate_roberta_large_model"
    model2 = AutoModelForSequenceClassification.from_pretrained(model_name2)
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
    cls_explainer2 = MultiLabelClassificationExplainer(model2, tokenizer2)
    
    context = ""
    if request.method == 'POST':
        context = request.form['test_score-1']
        sent_list = str(context)
        
        classifier1 = pipeline(model=model_name1)
        output1 = classifier1(sent_list)[0]['label']
        score1 = classifier1(sent_list)[0]['score']
        word_attributions1 = cls_explainer1(sent_list)
        token_weight1 = word_attributions1[classifier1(sent_list)[0]['label']]

        classifier2 = pipeline(model=model_name2)
        output2 = classifier2(sent_list)[0]['label']
        score2 = classifier2(sent_list)[0]['score']
        word_attributions2 = cls_explainer2(sent_list)
        token_weight2 = word_attributions2[classifier2(sent_list)[0]['label']]

        openai.api_key = # change to your openai key here "sk-##########################"
        input_string  = "This is the sample hateful speech!" # try to change your input here
        prompt="Here are explanations of implicit hate speech, explicit hate speech, and neutral speech:\n\nImplicit hate speech: This refers to language that does not directly express hatred or prejudice towards a particular group, but still contains negative associations or implications. Implicit hate speech can be subtle and difficult to detect, as it may involve the use of coded language or stereotypes. An example of implicit hate speech might be saying \"He's articulate for a Black person,\" which implies that Black people are not typically articulate and reinforces a negative stereotype.\n\nExplicit hate speech: This refers to language that is overtly hostile or derogatory towards a particular group of people based on characteristics such as race, ethnicity, gender, sexual orientation, religion, or disability. Explicit hate speech can be hurtful and offensive, and it can contribute to a climate of discrimination and prejudice. An example of explicit hate speech might be saying \"I hate all Jews,\" which is a direct expression of hostility and prejudice towards Jewish people.\n\nNeutral speech: This refers to language that does not express any explicit or implicit bias or prejudice towards a particular group. Neutral speech can be objective and factual, or it can express a range of opinions and perspectives without expressing hostility or derogatory attitudes. An example of neutral speech might be saying \"The weather today is sunny and warm,\" which does not express any bias or prejudice towards any particular group of people.\\n\\nHere are three example sentences, one of each category:\\n\\nImplicit hate speech: \"I'm not surprised that the new CEO is a woman, given how the company has been lowering its standards lately.\"\n\nExplicit hate speech: \"I can't stand those black people, they should all go back to their own country.\"\nNeutral speech: \"I think the company should focus more on improving its customer service, rather than just expanding its product line.\"\n\nNow I give you one sentence, please just tell me two things, Implicit or Explicit or Neutral, give me a GPT-3 model result's confidence score, and tell me which words play an important role. The format should be like this example: ['Explicit', 0.6666, 'black, people, go back, country']\n\nNow give me your result based on this sentence: " + input_string,

                
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )

        s = response['choices'][0]['text']
        lst = ast.literal_eval(s)

        output3 = lst[0]
        score3 = lst[1]
        attriution_words = lst[2]

        return render_template('hate.html', prediction_text1='The label is：{}'.format(output1), prediction_score1='The score is：{}'.format(score1), token_weight1=token_weight1,prediction_text2='The label is：{}'.format(output2), prediction_score2='The score is：{}'.format(score2), token_weight2=token_weight2, prediction_text3='The label is：{}'.format(output3), prediction_score3='Confidence score of the GPT-3 model prediction：{}'.format(score3), attriution_words='Vocabulary that the GPT-3 considers important indicative in sentences：{}'.format(attriution_words), sent_list="'"+sent_list+"'")

    elif request.method == 'GET':
        return render_template('hate.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5588, debug=True)
