import streamlit as st
from ensemble_with_single_text import predict_single_text

# Set the page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon=":smiley:",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title of the app
st.title('Sentiment Analysis')

# Add a description
st.markdown("""
    <style>
    .description {
        font-size: 18px;
        color: #4F4F4F;
    }
    </style>
    <div class="description">
        <i class="fas fa-info-circle"></i> Enter a piece of text to analyze its sentiment. The model will predict whether the sentiment is positive or negative along with the confidence score.
    </div>
    <br>
""", unsafe_allow_html=True)

# Input field for user to enter text
st.markdown('<i class="fas fa-keyboard"></i> Enter text here:', unsafe_allow_html=True)
user_input = st.text_area("", height=150)

# Predict sentiment when the user clicks the button
if st.button('Predict Sentiment'):
    if user_input:
        # Predict sentiment using the imported function
        roberta_path = 'roberta-base1'  # Specify the path to your model
        num_labels = 2  # Specify the number of labels
        predicted_class, confidence = predict_single_text(user_input, roberta_path, num_labels)
        
        # Display the prediction
        st.markdown(f"""
            <style>
            .result {{
                font-size: 20px;
                color: #2E8B57;
            }}
            </style>
            <div class="result">
                <i class="fas fa-smile"></i> <b>Sentiment:</b> {predicted_class}<br>
                <i class="fas fa-chart-line"></i> <b>Confidence:</b> {confidence:.2f}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.write("Please enter some text to analyze.")
