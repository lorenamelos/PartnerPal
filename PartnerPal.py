# Loading Models with Pytorch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load the emotion detection model
emotion_model_name = "ahmettasdemir/distilbert-base-uncased-finetuned-emotion"
emotion_model = DistilBertForSequenceClassification.from_pretrained(emotion_model_name)
emotion_tokenizer = DistilBertTokenizer.from_pretrained(emotion_model_name)

# Load the sentiment analysis model
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Replace with your fine-tuned sentiment model if available
sentiment_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(sentiment_model_name)


# Predicting Emotions and Sentiments

import torch

def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors='pt')
    outputs = emotion_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_emotion = torch.argmax(probs, dim=1).item()
    return emotion_model.config.id2label[predicted_emotion]

def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors='pt')
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_sentiment = torch.argmax(probs, dim=1).item()
    return sentiment_model.config.id2label[predicted_sentiment]




#________________________________________________________________________________________________


import streamlit as st
import replicate

# Initialize session state to store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

def stream_llama_response(few_shot_examples, user_input, top_k=50, top_p=0.9, max_tokens=512, min_tokens=0,
                          temperature=0.5, presence_penalty=1.15, frequency_penalty=0.2):
    model_name = "meta/meta-llama-3-70b-instruct"

    # Update the prompt template to include few-shot examples, conversation history, and the {prompt} placeholder
    prompt_template = f"""
    system

    You are a supportive and caring partner. Respond to your partner's messages with empathy, love, and understanding.

    {few_shot_examples}

    user

    {{prompt}}

    partner
    """

    response = ""

    # Stream the output from the model
    for event in replicate.stream(
        model_name,
        input={
            "top_k": top_k,
            "top_p": top_p,
            "prompt": user_input,  # This will replace {prompt} in the template
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "temperature": temperature,
            "prompt_template": prompt_template,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        },
    ):
        response += str(event)
        yield response  # Yield the current response to update the UI incrementally

def generate_response_with_emotional_intelligence(user_input):
    # Step 1: Use your predefined few-shot examples
    few_shot_examples = """
    user: Hey babe, how was your day?
    partner: It was pretty good! I managed to finish an important project at work, which feels like a huge weight off my shoulders. How about you? What did you get up to today?

    user: I didn’t like the way you talked to me earlier.
    partner: I’m sorry if I hurt your feelings. Let’s talk about it and make things right.

    user: I’m feeling really stressed about work.
    partner: I’m here for you. Let’s take a break and relax together. Maybe we should do something different. Do you wanna go for a weekend trip to relax in the woods?

    user: Can you get me a chai latte on the way home please?
    partner: Sure! Do you want it with any special add-ons or just the regular chai latte?

    user: You never listen to what I say.
    partner: I'm sorry you feel that way. I really do try to listen, but maybe I'm not understanding something. Can we talk about this more and figure out what's going wrong?

    user: I feel like you don't have space for me. When I share something important for me with you, you usually don't say much. I am not sure you can support me when I need.
    partner: I am sorry. I'm feeling a bit overwhelmed by work, but I really want to be there for you. How can we overcome this? I want to support you and be there for you.

    user: Hey bae
    partner: Hey babe, how was your day?
    user: It was terrible. I had an awful day at work...
    partner: Aw, I'm so sorry to hear that. Can you tell me more about what happened? I'm all ears and here to listen.
    user: My boss is such a jerk. The situation was with one of my colleagues, but it made me feel like it was with me. he is such a misogynist. so sad...
    partner: I'm so sorry to hear that. What can I do to make you feel better?

    """

    # Step 2: Generate a streaming response with LLaMA via Replicate using few-shot learning
    response_generator = stream_llama_response(few_shot_examples, user_input)

    return response_generator

#___________________________________________________________________________


# Streamlit interface

## Page Configuration
st.set_page_config(
    page_title="PartnerPal",
    page_icon=":white_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)
# # Sidebar for page navigation
# page = st.sidebar.selectbox("Navigate", ["Chatbot", "About PartnerPal"])

# Page: Chatbot

st.title(":rainbow[PartnerPal: Your Emotionally Intelligent Chatbot]")
st.subheader(":blue[Empathy and Understanding at Your Fingertips]")
st.markdown("""
        <div style="text-align: justify">
        Welcome to PartnerPal, your AI-powered companion designed to enhance emotional intelligence in every conversation.
        Whether you need someone to talk to, seek advice, or just want a friendly chat, PartnerPal is here to listen,
        understand, and respond with empathy. Our chatbot not only engages in meaningful dialogue but also detects the underlying
        emotions and sentiments in each interaction, providing responses that resonate with your feelings. Start a conversation
        and experience the difference that emotional intelligence can make.
        </div>
        """, unsafe_allow_html=True)





# Display chat messages using st.chat_message
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("enter your message here"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user's message immediately
    with st.chat_message(""):
        st.markdown(prompt)

    # Predict emotion and sentiment
    emotion = predict_emotion(prompt)
    sentiment = predict_sentiment(prompt)

    # Display predicted emotion and sentiment
    with st.chat_message("system"):
        st.markdown(f"**Emotion:** {emotion}, **Sentiment:** {sentiment}")

    # Placeholder for the chatbot's streaming response
    response_placeholder = st.chat_message("assistant")
    response_text = response_placeholder.empty()

    # Generate response
    response_generator = generate_response_with_emotional_intelligence(prompt)

    # Stream the response and update the placeholder
    full_response = ""
    for response in response_generator:
        full_response = response
        response_text.markdown(full_response)

    # Store the full assistant's response in the session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
