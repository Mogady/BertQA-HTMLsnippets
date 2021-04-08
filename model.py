import numpy as np


def decode_scores(start, end, max_answer_len, topk=1):
    """
    source : https://huggingface.co/transformers/_modules/transformers/pipelines.html#QuestionAnsweringPipeline
    Take the output of any QuestionAnswering head and will generate probabilities for each span to be
    the actual answer.
    In addition, it filters out some unwanted/impossible cases like answer len being greater than
    max_answer_len or answer end position being before the starting position.
    The method supports output the k-best answer through the topk argument.

    Args:
        start: numpy array, holding individual start probabilities for each token
        end: numpy array, holding individual end probabilities for each token
        topk: int, indicates how many possible answer span(s) to extract from the model's output
        max_answer_len: int, maximum size of the answer to extract from the model's output
    """
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
    return start, end, candidates[0, start, end]


def compute_scores(start, end, undesired_tokens_mask, max_length):
    """
    source: https://huggingface.co/transformers/_modules/transformers/pipelines.html#QuestionAnsweringPipeline
    calculate the scores for start and end spans taking attention and other token masks into consideration
    :param start:
    :param end:
    :param undesired_tokens_mask:
    :param max_length:
    :return:
    """
    start_, end_ = start.detach().numpy()[0], end.detach().numpy()[0]
    start_ = np.where(undesired_tokens_mask, -10000.0, start_)
    end_ = np.where(undesired_tokens_mask, -10000.0, end_)
    start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
    end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))
    #     start_[0] = end_[0] = 0.0
    starts, ends, scores = decode_scores(start_, end_, max_length)
    return starts[0], ends[0], scores[0]


def predict(question, context, tokenizer, model, max_length=512):
    """
    predict start and answer spans
    :param question:
    :param context:
    :param tokenizer:
    :param model:
    :param max_length:
    :return:
    """
    # get question encoded
    truncated_query = tokenizer.encode(
        question, add_special_tokens=False, truncation=True, max_length=max_length
    )
    # encode the concatenation of the question with the context
    inputs = tokenizer.encode_plus(question, context, truncation="only_second", add_special_tokens=True,
                                   return_tensors="pt", max_length=max_length)

    # get the tokenizer ids
    input_ids = inputs["input_ids"].tolist()[0]
    # mask question tokens to not be considered in score calculation
    # [1:len(question)+1:len(answer)]
    p_mask = np.ones_like(input_ids)
    # [[cls]question[sep] answer]
    # [11111111000000000]
    p_mask[len(truncated_query) + 2:] = 0
    special_token_indices = np.asarray(
        tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    ).nonzero()
    # [111111110000000001]
    p_mask[special_token_indices] = 1

    # [00000011111110]
    undesired_tokens = np.abs(np.array(p_mask) - 1)
    undesired_tokens_mask = undesired_tokens == 0.0
    start, end = model(**inputs)
    start_, end_, score = compute_scores(start, end, undesired_tokens_mask, max_length)

    # decode the model answer using the same tokenizer
    answer = tokenizer.decode(input_ids[start_:end_ + 1], skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)
    # decode the original context using the same tokenizer to have the same decoding result
    full_text = tokenizer.decode(input_ids[len(truncated_query) + 2:], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
    # find the start position and the end position of the predicted answer in the original context
    start = max(0, full_text.find(answer))
    end = max(0, start + len(answer))
    if len(answer.split()) <= 3:
        return '', 0, 0, 0
    return answer, start, end, score

