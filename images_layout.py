import re
from bs4 import BeautifulSoup
import html2text
from tldextract import extract

transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-—", u"'''\"\"---")])
# pre-defined regex
h = html2text.HTML2Text()
RE_ORDERED_LIST_MATCHER = re.compile(r"\d+\.\s")
RE_UNORDERED_LIST_MATCHER = re.compile(r"[-\*\+]\s")
RE_SPACE = re.compile(r"\s\+")
RE_LINK = re.compile(r"((\[.*?\]) ?(\(.*?\)))|((\[.*?\]):(.*?))")
RE_STRONG = re.compile(r"\*\*((?!\*\*).+?)\*\*|\*\*((?!\*\*).+)")
RE_MD_CHARS_MATCHER_ALL = re.compile(r"([`\*_{}\[\]\(\)#!])")
RE_IMAGES = re.compile(r"<img.*?/>")


def clean_html(html):
    """
    clean html code from non needed elements
    :param html:
    :return:
    """
    soup = BeautifulSoup(html, 'html.parser')
    footers = soup.findAll(["footer"])  # remove all javascript and stylesheet code
    if len(footers) > 0:
        footers[-1].extract()
    return str(soup)


def create_text_snippet(text, limit=150):
    """
    create text snippet using html2text with specific limit
    :param text:
    :param limit:
    :return:
    """
    # remove extra newlines from html2text output
    text = re.sub(r'\\n+', '', text)
    text = re.sub(r'\n+', '\n', text)
    # loop through all the lines till reaching the limit
    lines = text.split('\n')
    lines = [line for line in lines if re.search(r'\w+', line)]
    target_length = 0
    target_text = []
    for line in lines:
        needed_length = limit - target_length
        target_length += len(line.split())
        if target_length <= limit:
            target_text.append(line)
        else:
            # if it was one long paragraph crop it to the needed length to reach the limit
            target_text.append(' '.join(line.split()[:needed_length]))
            break

    text_snippet = '\n'.join(target_text)
    text_snippet = re.sub(RE_MD_CHARS_MATCHER_ALL, '', text_snippet).strip()
    return text_snippet


def create_html_snippet(text, limit=150):
    """
    create html snippet from html2text output
    :param text:
    :param limit:
    :return:
    """
    # remove extra new lines
    text = re.sub(r'\\n+', '', text)
    text = re.sub(r'\n+', '\n', text)
    # replace highlighted elements with <b> tag
    html_snippet = re.sub(RE_STRONG, '<b>\\1\\2</b>', text)

    HText = []
    target_length = 0
    lines = html_snippet.split('\n')
    for i, line in enumerate(lines):
        # only include lines that have valid text
        if re.search(r'\w+', line):
            needed_length = limit - target_length
            # check the length of the text
            target_length += len(re.sub(r'<.*?>', '', line).split())
            if target_length <= limit:
                # convert any # element to <h4> tag
                if line.strip().startswith('#'):
                    line = re.sub(r'^\s*#+', '<h4>', line)
                    line += '</h4>'
                else:
                    # if it is an ordered list replace with an un order list and wrap with list tags
                    if re.match(RE_ORDERED_LIST_MATCHER, line.strip()):
                        count = min(2, int((len(line) - len(line.lstrip(' '))) / 2))
                        line = re.sub(RE_ORDERED_LIST_MATCHER, '<li>', line.strip()) + '</li>'
                        start_list = ['<ul>'] * count
                        end_list = ['</ul>'] * count
                        line = ''.join(start_list + [line.strip()] + end_list)
                    # if it is an unordered list replace with an un order list and wrap with list tags
                    elif re.match(RE_UNORDERED_LIST_MATCHER, line.strip()):
                        count = min(2, int((len(line) - len(line.lstrip(' '))) / 2))
                        line = re.sub(RE_UNORDERED_LIST_MATCHER, '<li>', line.strip()) + r'</li>'
                        start_list = ['<ul>'] * count
                        end_list = ['</ul>'] * count
                        line = ''.join(start_list + [line.strip()] + end_list)
                    else:
                        # wrap anything else with <p> tag
                        line = '<p>' + line.strip() + '</p>'
                HText.append(line)
            else:
                # if it is a long paragraph crop to the needed length
                line = ' '.join(line.split()[:needed_length])
                HText.append('<p>' + line.strip() + '</p>')
                break
        else:
            continue

    html_snippet = '\n'.join(HText)
    html_snippet = re.sub(RE_MD_CHARS_MATCHER_ALL, '', html_snippet).strip()
    return html_snippet


def get_images(text, url):
    """
    get the images array from html
    :param text:
    :param url:
    :return:
    """
    text = re.sub(r'\\n+', '', text)
    text = re.sub(r'\n+', '\n', text)
    images = []
    matches = re.findall(RE_IMAGES, text)
    # correct images link to include the main domain if not included
    for match in matches:
        soup = BeautifulSoup(match, 'html.parser')
        src = soup.img.get('src')
        if not extract(src).domain:
            main_domain = url.split("://")[1].split("/")[0]
            correct_link = "https://" + main_domain + src
            src = correct_link
        images.append(src)
    return images


def get_snippets(html, url):
    """
    main function for extracting html,text snippets
    :param html:
    :param url:
    :return:
    """
    try:
        html = clean_html(html)
        h.body_width = 1000
        # ignore any non text element to generate text snippet
        h.ignore_links = True
        h.ignore_images = True
        h.ignore_tables = False
        h.bypass_tables = False
        text = h.handle(html)
        text_snippet = create_text_snippet(text)
        # include images to be extracted
        h.bypass_tables = True
        h.ignore_images = False
        h.images_as_html = True
        # remove images again to generate html snippet without it
        text = h.handle(html)
        images = get_images(text, url)
        h.ignore_images = True
        text = h.handle(html)
        html_snippet = create_html_snippet(text)

        return html_snippet, text_snippet, images

    except Exception:
        return '', '', []
