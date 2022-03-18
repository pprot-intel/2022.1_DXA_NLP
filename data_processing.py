import html_reader as reader
import tokens_bert as tokens
import numpy as np
from urllib import parse


# function to load text from given urls
def load_context(sources):
    input_urls = []
    paragraphs = []
    for source in sources:
        result = parse.urlparse(source)
        if all([result.scheme, result.netloc]):
            input_urls.append(source)
        else:
            paragraphs.append(source)

    paragraphs.extend(reader.get_paragraphs(input_urls))
    # produce one big context string
    return "\n".join(paragraphs)

# based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L163


def postprocess(output_start, output_end, question_tokens, context_tokens_start_end, input_size):

    def get_score(logits):
        out = np.exp(logits)
        return out / out.sum(axis=-1)

    # get start-end scores for context
    score_start = get_score(output_start)
    score_end = get_score(output_end)

    # index of first context token in tensor
    context_start_idx = len(question_tokens) + 2
    # index of last+1 context token in tensor
    context_end_idx = input_size - 1

    # find product of all start-end combinations to find the best one
    max_score, max_start, max_end = find_best_answer_window(start_score=score_start,
                                                            end_score=score_end,
                                                            context_start_idx=context_start_idx,
                                                            context_end_idx=context_end_idx)

    # convert to context text start-end index
    max_start = context_tokens_start_end[max_start][0]
    max_end = context_tokens_start_end[max_end][1]

    return max_score, max_start, max_end


# based on https://github.com/openvinotoolkit/open_model_zoo/blob/bf03f505a650bafe8da03d2747a8b55c5cb2ef16/demos/common/python/openvino/model_zoo/model_api/models/bert.py#L188
def find_best_answer_window(start_score, end_score, context_start_idx, context_end_idx):
    context_len = context_end_idx - context_start_idx

    score_mat = np.matmul(
        start_score[context_start_idx:context_end_idx].reshape(
            (context_len, 1)),
        end_score[context_start_idx:context_end_idx].reshape((1, context_len)),
    )

    # reset candidates with end before start
    score_mat = np.triu(score_mat)
    # reset long candidates (>16 words)
    score_mat = np.tril(score_mat, 16)
    # find the best start-end pair
    max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
    max_score = score_mat[max_s, max_e]

    return max_score, max_s, max_e

# generator of a sequence of inputs


def prepare_input(question_tokens, context_tokens, input_keys):
    # path to vocabulary file
    vocab_file_path = "data/vocab.txt"

    # create dictionary with words and their indices
    vocab = tokens.load_vocab_file(vocab_file_path)

    # define special tokens
    cls_token = vocab["[CLS]"]
    sep_token = vocab["[SEP]"]

    input_ids = [cls_token] + question_tokens + \
        [sep_token] + context_tokens + [sep_token]
    # 1 for any index
    attention_mask = [1] * len(input_ids)
    # 0 for question tokens, 1 for context part
    token_type_ids = [0] * (len(question_tokens) + 2) + \
        [1] * (len(context_tokens) + 1)

    # create input to feed the model
    input_dict = {
        "input_ids": np.array([input_ids], dtype=np.int32),
        "attention_mask": np.array([attention_mask], dtype=np.int32),
        "token_type_ids": np.array([token_type_ids], dtype=np.int32),
    }

    # some models require additional position_ids
    if "position_ids" in [i_key.any_name for i_key in input_keys]:
        position_ids = np.arange(len(input_ids))
        input_dict["position_ids"] = np.array([position_ids], dtype=np.int32)

    return input_dict
