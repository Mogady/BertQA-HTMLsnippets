import lxml
from io import StringIO
from bs4 import BeautifulSoup

from images_layout import get_snippets

parser = lxml.etree.HTMLParser()


def extract_text(body):
    """extract clean text form HTML body"""
    soup = BeautifulSoup(body, 'html.parser')
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text


def remove_odd_nodes(tree):
    """ remove empty nodes that has children like <body> """
    for branch in tree:
        if len(branch.getchildren()) == 1:
            if branch.text is None:
                branch.getparent().replace(branch, branch[0])
                remove_odd_nodes(tree)
    return tree


def _recurse_over_nodes(tree, parent_key, data):
    """ reconstruct the tree to have only 1 level of elements rather than having too many branches"""
    for branch in tree:
        key = branch.tag
        if branch.getchildren():
            key = '%s_%s' % (parent_key, key)
            data.append((key, branch))
            data = _recurse_over_nodes(branch, key, data)
        else:
            key = '%s_%s' % (parent_key, key)
            data.append((key, branch))
    return data


def post_process(html_context, answer, url, tokenizer):
    """
    post process the model prediction to get the full HTML chunk the holds the answer
    :param html_context:
    :param answer:
    :param url:
    :param tokenizer:
    :return:
    """
    # encode the full text to get the same decoding as the model answer
    full_text = tokenizer.decode(tokenizer.encode(extract_text(html_context),  max_length=512, truncation=True),
                                 skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # construct the html as a tree and convert it to 1 level tree with many elements acts as chunks
    tree = lxml.etree.parse(StringIO(html_context), parser=parser)
    tree = remove_odd_nodes(tree.getroot())
    paths = _recurse_over_nodes(tree, 'root', [])
    paths_lengths = [len(x[0].split('_')) for x in paths]

    cleaned = []
    for x in paths:
        text = x[1].text
        tail = x[1].tail
        if (len(x[0].split('_')) == (min(paths_lengths))) and (
                (text is not None) and (len(text) > 1) or (tail is not None) and (len(tail) > 1)):
            cleaned.append(x[1])
        elif len(x[0].split('_')) == (min(paths_lengths) + 1) and x[1].getparent() not in cleaned:
            cleaned.append(x[1])

    # merge the cleaned tree to html again
    html_elements = [lxml.etree.tostring(d).strip().decode('utf-8') for d in cleaned]
    # decode the text of the html with the same tokenizer
    texts = [
        tokenizer.decode(tokenizer.encode(extract_text(x),  max_length=512, truncation=True), skip_special_tokens=True,
                         clean_up_tokenization_spaces=True)
        for x in
        html_elements]

    # search for the chunk that holds the extracted answer from the full content
    chunks = []
    for i, text in enumerate(texts):
        start = full_text.find(text)
        end = start + len(text)
        chunks.append((i, start, end))

    start_chunk_answer = list(filter(lambda x: answer[1] in range(x[1], x[2] + 1), chunks))[0][0]
    end_chunk_answer = list(filter(lambda x: answer[2] in range(x[1], x[2] + 1), chunks))[0][0]
    chunk_html = [html_elements[x[0]] for x in chunks[start_chunk_answer:end_chunk_answer + 1]]

    # clean the snippet
    html_snippet = '\n'.join(chunk_html)
    html_snippet, text_snippet, images = get_snippets(html_snippet, url)

    return html_snippet, text_snippet, images
