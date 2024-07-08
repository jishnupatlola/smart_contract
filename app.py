from flask import Flask, request, render_template
import pandas as pd
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = TFRobertaForSequenceClassification.from_pretrained('./model')
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

# Define the label mapping
label_mapping = {0: "Logical error", 1: "buffer overflow",2:"No issues"}

def predict(contract):
    inputs = tokenizer(contract, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    predicted_label = tf.argmax(predictions, axis=-1).numpy()[0]
    return label_mapping[predicted_label]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)
        df['prediction'] = df['contract'].apply(lambda x: predict(x))
        return df.to_html()

if __name__ == '__main__':
    app.run(debug=True)
    
