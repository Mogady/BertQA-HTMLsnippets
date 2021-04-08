# Introduction
This is a Question-Answering Bert model for HTML content.
The model simply answers the questions using the text of the HTML context, then
post-process the answer and return the html section that contains the predicted 
answer.
### Request :
```json
        {"html_url": "the url for the article contains the article",
         "question": "user question",
         "article": "HTML article to extract answer from"
        }
```
### Output:
```json
        {"html_url": "the url for the whole article contains the article",
         "question": "user question",
         "article": "HTML article to extract answer from",
         "html_snippet": "HTML chunk/section that holds the answer of the question",
         "text_snippet": "text chunk/section that holds the answer of the question",
         "images": "list of images exists in the article"
  
        }
```

### How does it work:
Here I use distilbert which is pre-trained on the QA task and only
works with text, not HTML, however, I want a model that can returns the HTML version
of the answer, to do that I have to search for the answer in the HTML content and find
the container element which has the answer in it.

I parsed the HTML as a tree and started looking in each branch
for the model predicted answer.

### How to run:
This is deployed using [cortex-project](https://github.com/cortexproject/cortex).
#### Install the Cortex CLI.
    $ bash -c "$(curl -sS https://raw.githubusercontent.com/cortexlabs/cortex/0.18/get-cli.sh)"
#### inside the project folder run
    cortex deploy
    cortex get reader

#### to monitor the server run
    cortex log reader
