Smart Contract Classification with CodeBERT
This project is a web application that uses a fine-tuned CodeBERT model to classify smart contracts. The model can identify whether a smart contract contains logical errors, buffer overflows, or no issues.


Table of Contents
-Demo
-Features
-Installation
-Usage
-File Upload Format
-Model Training
-Contributing
-License
-Acknowledgments


Demo output:
 ![tune4](https://github.com/jishnupatlola/vulnerability-detection-using-codebert/assets/97329738/99a6857f-e99e-4209-983b-27bd9f7e584d)
![final res](https://github.com/jishnupatlola/vulnerability-detection-using-codebert/assets/97329738/6bb750c9-b800-48f9-abf3-8f64345f77d1)


Features:
-Upload smart contracts in CSV format
-Predicts the type of issues (logical errors, buffer overflows, or no issues) in smart contracts using a fine-tuned CodeBERT model
-Displays predictions in an easy-to-read HTML table

Installation:
Prerequisites
Python 3.7 or later
TensorFlow
Transformers library from Hugging Face
Flask
Pandas

Setup:
Clone the repository:
git clone https://github.com/yourusername/smart-contract-classification.git
cd smart-contract-classification


Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
pip install -r requirements.txt

Download the fine-tuned model and place it in the model directory.

Ensure the model is fine-tuned and saved in the appropriate format the application expects.

Usage:
-Run the Flask application:
  python app.py
-Open your web browser and navigate to http://127.0.0.1:5000.
-Upload a CSV file with the smart contracts to get predictions.
  File Upload Format
  The uploaded CSV file should have the following format:
  
    contract	                  label
    pragma solidity ^0.8.0;...	1
    pragma solidity ^0.8.0;...	0
    ...	...
contract: The Solidity code of the smart contract.
label: The true label (optional), which can be used for evaluation purposes.


-Model Training
To train your model, follow these steps:

Prepare your dataset.
Fine-tune the CodeBERT model using your dataset.
Save the fine-tuned model to the model directory.
Example training script:
# Import necessary libraries
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, TFTrainer, TFTrainingArguments
import tensorflow as tf

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = TFRobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=3)

# Prepare the dataset (example)
train_dataset = ...

# Define training arguments
training_args = TFTrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Create Trainer instance
trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./model')

Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Hugging Face Transformers
TensorFlow
Flask


Feel free to customize this template further to fit the specifics of your project. If you have any additional sections or details that you want to include, you can easily expand this template.
  

 
