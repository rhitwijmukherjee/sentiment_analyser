FROM python:3.9.13

#Set working directory to app
WORKDIR C:/Users/User/NLP_project_1/app

#Copy over files to app directory
COPY ./requirements.txt ./

# Install the required packages
RUN pip install -r requirements.txt

# Download the 'vader_lexicon' resource
RUN python -c "import nltk; nltk.download('vader_lexicon')"

#Copy rest of files over to working directory
COPY ./ ./

# Expose the port that the Flask app will run on
EXPOSE 5000

# Set the command to run the Flask app
CMD ["python", "app.py"]