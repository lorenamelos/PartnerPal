# PartnerPal: An Emotionally Intelligent Chatbot 💞
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://partnerpal.streamlit.app/)

## Overview

PartnerPal is an AI-powered chatbot designed to simulate emotionally intelligent conversations between partners. It leverages state-of-the-art NLP models to detect underlying emotions and sentiments in user input and responds accordingly with empathy and understanding. This project demonstrates the integration of emotion and sentiment analysis models with a large language model to create a highly interactive and context-aware chatbot.

## Features

- **Emotion and Sentiment Detection**: The chatbot uses fine-tuned `DistilBERT` models to predict the user's emotions and sentiments. These predictions guide the chatbot's responses, ensuring they are contextually appropriate and emotionally resonant.
  
- **Few-Shot Learning**: The chatbot leverages few-shot examples to maintain a consistent conversational style and tone. These examples are incorporated into the prompt to guide the model's behavior.

- **Streaming Response Generation**: The chatbot generates responses in a streaming fashion, creating a more interactive and real-time experience for the user.

- **Replicate API Integration**: PartnerPal uses the Replicate API to run LLaMA 3, a large language model that powers the chatbot's responses. This allows for efficient and scalable deployment of the model without the need for extensive local resources. Learn more about using LLaMA 3 with Replicate [here](https://replicate.com/blog/run-llama-3-1-with-an-api).

## Models Used

1. **Emotion Detection Model**: [`ahmettasdemir/distilbert-base-uncased-finetuned-emotion`](https://huggingface.co/ahmettasdemir/distilbert-base-uncased-finetuned-emotion)
2. **Sentiment Analysis Model**: [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
3. **Language Model**: `meta/meta-llama-3-70b-instruct` via [Replicate API](https://replicate.com/blog/run-llama-3-1-with-an-api)

## How It Works

1. **User Input**: The user initiates a conversation through the Streamlit interface.
2. **Emotion and Sentiment Analysis**: The input is processed to detect the underlying emotion and sentiment using models from Hugging Face.
3. **Response Generation**: The chatbot generates a response using the LLaMA 3 language model, running through the Replicate API. The response is guided by few-shot examples and takes into account the entire conversation history.
4. **Interaction**: The conversation continues with the chatbot maintaining context and adapting its responses based on the detected emotional cues.

## Requirements

- `torch`
- `transformers`
- `streamlit`
- `replicate`

## Setup

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/PartnerPal.git
    cd partnerpal
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run PartnerPal.py
    ```

## Future Work

- **Enhanced Contextual Understanding**: Further fine-tuning the models to improve the chatbot's ability to understand complex emotions and nuanced sentiments.
- **Multi-turn Memory**: Expanding the chatbot's memory to persist across different sessions.
- **Add Better Conversation Memory**: PartnerPal remembers previous interactions within the session, allowing it to maintain context and provide more meaningful responses as the conversation progresses.

## Acknowledgments

- **Replicate API**: Thanks to the Replicate team for providing a scalable API to run LLaMA 3. Learn more about using LLaMA 3 [here](https://replicate.com/blog/run-llama-3-1-with-an-api).
- **Hugging Face**: Special thanks to Hugging Face for providing pre-trained models for emotion and sentiment analysis. Check out the emotion detection model [here](https://huggingface.co/ahmettasdemir/distilbert-base-uncased-finetuned-emotion) and the sentiment analysis model [here](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Model Licenses

1. **DistilBERT Sentiment Analysis Model**: Licensed under the Apache 2.0 License. See details [here](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/discussions).
2. **Replicate API**: Licensed under the Apache 2.0 License. See the [Replicate Python SDK License](https://github.com/replicate/replicate-python/blob/main/LICENSE).

