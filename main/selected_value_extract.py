from txt_utilities import any_item_be_contain, replace_all, any_item_start_with

full_check_flag_list = ["■", "√", "□", "þ", "☑"]
left_bracket = ['(', '（']
right_bracket = [')', '）']


def extract_selected_value(value):

    def repair_value_list(v_l):
        joined_value = ''
        remove_values = []
        for i, v in enumerate(v_l):
            if joined_value:
                remove_values.append(v)
            if any_item_be_contain(left_bracket, v) and not any_item_be_contain(right_bracket, v):
                for j, _v in enumerate(v_l[i+1:]):
                    if any_item_be_contain(right_bracket, _v):
                        joined_value = ' '.join(v_l[i:i+j+2])
                        break
                v_l[i] = joined_value
        for r_v in remove_values:
            v_l.remove(r_v)
        return v_l

    value_list = list(filter(None, value.replace('\n', ' ').split(' ')))
    colon_list = [":"]
    if not any_item_be_contain(full_check_flag_list, value_list[0]):
        # 焊盘与孔径等大（即内外直径相同）的孔，按非金属化孔处理      □ 同意          √  不同意
        if any_item_be_contain(colon_list, value_list[0]):
            result = value_list[0]
        else:
            result = value_list[0] + ":"
        select_val = get_selected_value(repair_value_list(value_list[1:]))
        if select_val:
            result += select_val
        return result
    else:
        result = get_selected_value(repair_value_list(value_list))
    return result


def get_selected_value(value_list):
    selected_flag_list = ["■", "√", "þ", "☑"]
    next_item_flags = ["□", "■", "√"]
    selected_value = ""
    for i, val in enumerate(value_list):
        if any_item_be_contain(selected_flag_list, val):
            if len(val) == 1:
                if len(value_list) == i + 1:
                    continue
                # ■ 添加  □ 不添加 or ■ 电性能测试报告  ■ 阻抗测试报告
                if len(value_list) == i + 2 or any_item_be_contain(next_item_flags, value_list[i + 2]):
                    selected_value = selected_value + "," + value_list[i + 1]
                else:
                    # ■ 海格 2级
                    selected_value = selected_value + "," + value_list[i + 1] + ' ' + value_list[i + 2]
            else:
                # ■添加  □不添加 or  ■电性能测试报告  ■阻抗测试报告
                if len(value_list) == i + 1 or "□" in value_list[i + 1] or any_item_be_contain(selected_flag_list,
                                                                                               value_list[i + 1]):
                    if any_item_start_with(selected_flag_list, val):
                        selected_value = selected_value + "," + val[1:]
                    else:
                        # 1，有铅喷锡□   2，无铅喷锡√
                        selected_value = selected_value + "," + val
                else:
                    # ■海格 2级
                    selected_value = selected_value + "," + val[1:] + ' ' + value_list[i + 1]
    if selected_value:
        # replace process for "√□添加" case
        return replace_all(full_check_flag_list, selected_value[1:])
    else:
        return selected_value
