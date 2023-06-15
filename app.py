import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go

# Page title and description
st.title("Depression Checking App")
st.markdown("This app allows you to check your depression level based on a set of questions. "
            "Please note that the results provided by this app are not a substitute for professional medical advice. "
            "If you suspect you may be depressed, it is important to consult a psychiatrist for a proper diagnosis and treatment.")

# Display images of depressed people
st.image(["d3.jpeg", "d4.jpeg", "d6.jpeg"], width=200)

# Introduction and note
st.markdown("## About Depression")
st.markdown("Depression is a mood disorder that causes a persistent feeling of sadness, "
            "loss of interest, and a lack of motivation. It can affect your thoughts, feelings, behavior, and overall well-being.")

st.markdown("## Note")
st.markdown("This app is designed to help you evaluate your depression level based on a set of questions. "
            "However, it is crucial to seek professional help from a psychiatrist to get an accurate diagnosis and appropriate treatment.")

# File selection
file = st.file_uploader("Upload a CSV file to check depression level for multiple responses", type=["csv"])

# Define the questions
questions = [
    "Over the past two weeks, have you felt down, depressed, or hopeless?",
    "Do you often experience a lack of interest or pleasure in activities that you used to enjoy?",
    "Have you noticed any significant changes in your appetite or weight recently?",
    "Do you struggle with insomnia (difficulty falling asleep or staying asleep) or hypersomnia (excessive sleepiness)?",
    "Have you felt unusually fatigued or lacked energy, even when not engaging in physically or mentally demanding tasks?",
    "Do you often feel a sense of worthlessness or excessive guilt?",
    "Have you experienced difficulties in concentrating, making decisions, or experiencing a general lack of focus?",
    "Do you have recurrent thoughts of death or suicidal ideation?",
    "Have you noticed changes in your psychomotor activity (e.g., slowed movements, restlessness) as observed by others?",
    "Have you experienced a persistent feeling of sadness, emptiness, or a sense of 'being trapped'?"
]

# Ask questions and get answers
answers = []
for i in range(len(questions)):
    st.subheader(f"Question {i+1}")
    answer = st.text_area(questions[i])
    answers.append(answer)

# Button for checking depression
if st.button("Check Depression"):
    # Classification using your model
    model_path = "C:\\Users\\Administrator\\PycharmProjects\\FYP\\FYPFrontEnd\\lstm_depressed.h5"  # Replace with the path to your saved model
    model = load_model(model_path)

    # Tokenizer
    num_words = 10000  # Replace with your desired number of words
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(questions)

    # Set the maximum sequence length
    max_sequence_length = 30  # Replace with your desired maximum sequence length

    depressed_count = 0
    non_depressed_count = 0
    depressed_questions = []
    non_depressed_questions = []

    for i in range(len(answers)):
        answer = answers[i]
        # Tokenize the answer
        answer_sequence = tokenizer.texts_to_sequences([answer])
        answer_sequence = pad_sequences(answer_sequence, maxlen=max_sequence_length)

        # Perform classification using your LSTM model on the tokenized answer
        # Replace the placeholder code below with your actual classification logic
        predicted_label = model.predict(answer_sequence)

        if predicted_label[0][0] * 100 > 50:
            depressed_count += 1
            depressed_questions.append(questions[i])
        else:
            non_depressed_count += 1
            non_depressed_questions.append(questions[i])

    # Calculate percentage of depressed questions
    depressed_percentage = (depressed_count / len(answers)) * 100
    non_depressed_percentage = (non_depressed_count / len(answers)) * 100

    # Display classification results
    st.markdown("## Classification Results")
    st.write(f"Percentage of depressed questions: {depressed_percentage:.2f}%")
    st.write(f"Percentage of non-depressed questions: {non_depressed_percentage:.2f}%")

    # Suggest exercises and entertainment materials for depressed users
    if depressed_percentage >= 60:
        st.markdown("## Suggestions for Depressed Users")
        st.write("Based on your responses, it seems that you may be experiencing symptoms of depression. "
                 "Here are some suggestions that may help improve your mood:")
        st.markdown("- Engage in regular physical exercise, such as walking, jogging, or yoga.")
        st.markdown("- Practice relaxation techniques, such as deep breathing exercises or meditation.")
        st.markdown("- Spend time engaging in activities you enjoy, such as hobbies or creative outlets.")
        st.markdown("- Reach out to friends or family for support and social interaction.")
        st.markdown("- Consider seeking professional help from a psychiatrist or therapist for proper diagnosis and treatment.")

    # Display word cloud
    st.markdown("## Word Cloud")
    st.write("Depressed Questions Word Cloud:")
    depressed_text = ' '.join(depressed_questions)
    wordcloud = WordCloud(width=800, height=400).generate(depressed_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Depressed Questions Word Cloud')
    st.pyplot(plt)

    st.write("Non-Depressed Questions Word Cloud:")
    non_depressed_text = ' '.join(non_depressed_questions)
    wordcloud = WordCloud(width=800, height=400).generate(non_depressed_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Non-Depressed Questions Word Cloud')
    st.pyplot(plt)

# Display footer
st.markdown("Email: zainulabideen1796@gmail.com")
st.markdown("Developed by Engr Zain ul Abideen")
