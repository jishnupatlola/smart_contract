import pandas as pd
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.model_selection import train_test_split

# Load the data
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

# Preprocess the data
def preprocess_data(tokenizer, df, max_length):
    encodings = tokenizer(df['contract'].tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='tf')
    labels = df['label'].tolist()
    return encodings, labels

# Fine-tune the model
def fine_tune_model():
    # Load the dataset
    train_df, test_df = load_data()

    # Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    # Preprocess the data
    train_encodings, train_labels = preprocess_data(tokenizer, train_df, max_length=512)
    test_encodings, test_labels = preprocess_data(tokenizer, test_df, max_length=512)

    # Convert data to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

    # Batch and shuffle the datasets
    train_dataset = train_dataset.shuffle(len(train_df)).batch(2)
    test_dataset = test_dataset.batch(2)

    # Initialize the model
    model = TFRobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=3)

    # Define optimizer, loss function, and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Fine-tune the model
    model.fit(train_dataset, validation_data=test_dataset, epochs=3)

    # Save the model
    model.save_pretrained('./model')

if __name__ == "__main__":
    fine_tune_model()
