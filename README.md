# sentiment_analyser
Simple Sentiment analyzer using NLP and Machine Learning, used Docker for containerization and created an api endpoint and build simple Html form for getting text input and show result using flask. 

# First Run the Jupyter Notebook
Then the model and vectorizer will save in your current working directory.

# Docker build command:
docker build -t sentiment-analysis-app

# Docker run command
docker run -p 5000:5000 sentiment-analysis-app
