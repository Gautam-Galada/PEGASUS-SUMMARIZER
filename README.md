This repository has an ipynb file which works with abstractive sumamrizer and it is very basic implementation


# Flask Summarizer
This is a Flask application that utilizes the Pegasus model for text summarization. It provides a web interface where users can input text and get summarized versions of the input.

# Prerequisites
Make sure you have the following dependencies installed:

- Flask
- regex
- transformers
- torch
- flask-ngrok
- splitter
You can install the dependencies by running the following command:

```
pip install flask regex transformers torch flask-ngrok splitter
```

# Usage

Clone the repository or download the source code files.
Run the Flask application by executing the following command in your terminal:

```
python app.py

```

The Flask application will start running, and you will see the URL where you can access the web interface (usually http://127.0.0.1:5000/).

Open the provided URL in your web browser.

In the web interface, enter the text you want to summarize in the input fields.

Click the "Submit" button.

The summarized versions of the input text will be displayed on the web page.

Description
The Flask application uses the Pegasus model for text summarization. It follows the following steps:

It imports the necessary libraries and sets up the Flask application.

It defines a function preparing_transcript that takes a text as input and performs the following tasks:

Processes the input text and saves it to a file.
Splits the text into smaller chunks based on the length.
Loads the quantized Pegasus model.
Generates summaries for each chunk of text using the quantized model.
Combines the summaries into a single summary.
It defines two routes: the root route ('/') and the form submission route ('/' with POST method).

The root route renders an HTML form where users can enter their text.

The form submission route receives the form data, processes the input text using the preparing_transcript function, and renders the results in the form.
