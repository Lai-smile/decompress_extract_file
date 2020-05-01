# Created by hudiibm at 2019/1/7
"""
Feature: #Enter feature name here
# Enter feature description here
Scenario: #Enter scenario name here
# Enter steps here
Test File Location: # Enter]
"""
import re
from difflib import SequenceMatcher


COLON = [':', '：']


def split_by_colon(val):
    # 将包含冒号的token内容拆分成两个token
    return re.split(r'[：:]', val)


def contain_multi_right_bracket(str):
    right_bracket = [')', '）']
    if any_item_be_contain(right_bracket, str):
        if str.count(right_bracket[0]) > 1 or str.count(right_bracket[1]) > 1:
            return True
    return False


def have_chinese_(input_string):
    pattern = u'[\u4e00-\u9fff]+'
    found_chinese = re.findall(pattern, input_string)
    return True if found_chinese else False


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def repair_num_str(s):
    if s.endswith('.0'):
        return s[0:s.find('.0')]
    else:
        return s


def remove_text_in_parenthesis(s):
    regex = re.compile(".*?\((.*?)\)")

    s = s.replace('（', '(')
    s = s.replace('）', ')')
    result = re.findall(regex, s)

    start = s.find('(')
    end = s.find(')')
    if start != -1 and end != -1:
        result = s[:start] + s[end + 1:]

    return result or s


def unit_seperitors(text):
    text = str(text)
    text = text.replace('；', '|')
    text = text.replace(',', '|')
    text = text.replace('、', '|')
    text = text.replace('，', '|')
    text = text.replace('\r', '|')
    return text


def clean_nameinsys(text):
    text = str(text)
    text = text.replace(':', '')
    text = text.replace('/', '')
    return text


def clean_colon_only(text):
    text = str(text)
    text = text.replace(':', '')
    text = text.replace('：', '')
    return text.strip()


def clean_tail_colon(text):
    if any_item_end_with(COLON, text):
        str_list = list(text)
        return ''.join(str_list[0:-1])
    else:
        return text


def remove_space(val):
    if have_chinese_(val):  # 英文单不能去除单词间的空格
        val = val.replace(' ', '')
    else:
        val = val.strip()
    return val


def convert_none_to_empty(s):
    str = '' if (s is None or s == 'None') else s
    return str


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def jaccard_list(a, b):
    # a,b must be a list
    return len(list(set(a).intersection(set(b)))) / len(list(set(a).union(set(b))))


def similar_to_any_elements(input_string, elements):
    return any(similarity(input_string, l) > 0.6 for l in elements)


def any_item_be_contain(str_list, target_string):
    if isinstance(target_string, str):
        for i in str_list:
            if i in target_string:
                return True
            else:
                continue
    return False


def replace_all(str_list, target_string):
    for i in str_list:
        target_string = re.sub(i, '', target_string)
    return target_string


def get_match_str(str_list, target_string):
    if isinstance(target_string, str):
        for i in str_list:
            if i in target_string:
                return i
    return None


def any_item_start_with(str_list, target_string):
    for s in str_list:
        if target_string.startswith(s):
            return True
        else:
            continue
    return False


def any_item_end_with(str_list, target_string):
    for s in str_list:
        if target_string.endswith(s):
            return True
        else:
            continue
    return False


def any_item_be_equal(str_list, target_string):
    if isinstance(target_string, str):
        for i in str_list:
            if i == target_string:
                return True
            else:
                continue
    return False


def all_item_be_contain(str_list, target_string):
    for s in str_list:
        if s not in target_string:
            return False
        else:
            continue
    return True


def get_number(num):
    nums = re.findall(r"\d+\.?\d*", num)
    return nums[0]


def get_alphabet(txt):
    return list(filter(None, txt.split('$')))[0]