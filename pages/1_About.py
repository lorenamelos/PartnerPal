# Page: About PartnerPal

import streamlit as st

st.set_page_config(page_title="About", page_icon=":white_heart:")
st.title('About PartnerPal')
st.header('Technical Aspects')
st.subheader('Integration of LLMs and Prompts')
st.markdown("""
   <div style="text-align: justify">
   Welcome to PartnerPal, your AI companion designed to make conversations more emotionally intelligent and meaningful.

    ## What is PartnerPal?

    PartnerPal is an AI-powered chatbot that not only engages in meaningful dialogue but also understands the emotions behind your words.
    Whether you’re feeling happy, sad, or anything in between, PartnerPal is here to listen, respond with empathy, and help you navigate your feelings.

    ## How It Works

    1. **Emotion and Sentiment Detection**: PartnerPal uses advanced AI models to detect the emotion and sentiment behind your messages.
    It then tailors its responses to ensure they resonate with how you're feeling.

    2. **Conversation Memory**: The chatbot remembers the conversation, ensuring that its responses are consistent and contextually relevant throughout the interaction.

    3. **Real-Time Responses**: PartnerPal responds in real-time, creating a smooth and natural conversation flow.

    ## Why Use PartnerPal?

    - **Empathy at its Core**: Every response is designed to be understanding and supportive, making PartnerPal not just a chatbot, but a companion you can rely on.

    - **Interactive and Engaging**: With real-time responses and conversation memory, PartnerPal feels more like talking to a friend than interacting with a machine.

    - **Emotionally Aware**: By understanding your emotions, PartnerPal provides responses that truly connect, making your interactions more meaningful.

    ## Get Started

    Simply start typing in the chat below, and let PartnerPal take care of the rest. Whether you’re looking for advice, a friendly chat, or just someone to listen, PartnerPal is here for you.

    ---

    _Disclaimer: PartnerPal is an AI-based chatbot. While it strives to provide helpful and empathetic responses, it is not a substitute for professional advice or human interaction._
    </div>
    """, unsafe_allow_html=True)
