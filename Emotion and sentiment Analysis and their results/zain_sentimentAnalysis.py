import os
import pandas as pd
import transformers as tf 
import torch

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

# Print the file names in the new current working directory
zain = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and "zain" in f and f.endswith('.csv')]
for i in zain:
    df = pd.DataFrame() 
    df = pd.read_csv(r"{}".format(i))
    tokenizer = tf.AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
    model = tf.AutoModelForSequenceClassification.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
    df['sentiment'] = df['tweetContent'].apply(lambda x: analyze_sentiment(x))
    df.to_csv("ResultsBert.csv")


