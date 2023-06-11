import os
import pandas as pd
import transformers as tf 
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
def analyze_sentiment(sentence):
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs[0]).item()
    print(predicted_label)
    return predicted_label


current_dir = os.getcwd()

# Search for a file containing "zain" and ending with ".csv" in the current directory and subdirectories
for dirpath, dirnames, filenames in os.walk(current_dir):
    for filename in filenames:
        if "zain" in filename and filename.endswith('.csv'):
            # Change the current working directory to the directory that contains the file
            os.chdir(dirpath)
            print("Changed current working directory to:", dirpath)
            break
# Output:
# [[{'label': 'anger', 0},
#   {'label': 'disgust', 1},
#   {'label': 'fear', '2},
#   {'label': 'joy', '3},
#   {'label': 'neutral', 4},
#   {'label': 'sadness', 5},
#   {'label': 'surprise', 6}]]




# Print the file names in the new current working directory
zain = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and "zain" in f and f.endswith('.csv')]
for i in zain:
    df = pd.DataFrame()
    df = pd.read_csv(r"{}".format(i))
    tokenizer = tf.AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base", truncation = True)
    model = tf.AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    df['sentiment'] = df['tweetContent'].apply(lambda x: analyze_sentiment(x))
    df.to_csv("ResultsEmotion.csv")



