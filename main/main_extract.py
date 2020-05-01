import os
from file_utilities import unpack_zip
from get_table_from_pdf import get_pdf_table
import pandas as pd
import numpy as np
from get_content_from_word import doc2docx
import docx
from get_content_from_docx import xml_to_dict


def decompress():
    base_url = r"F:\b074_10"
    original_order_list = []  # 原始订单名列表
    original_file_list = []  # 原始文件名列表
    article_name_list = []  # 文档名列表
    content_title_list = []  # 提取项（工程要求或技术要求）
    content_list = []  # 内容列表

    dirs = os.listdir(base_url)
    for dir in dirs:
        original_order_name = dir
        content = []
        try:
            ewr = unpack_zip(dir, base_url)
            if not ewr:
                pass
            else:
                order_file = ewr.split('/')[0]
                if order_file:  # 解压原始订单名压缩包的文件
                    folder_list = os.listdir(order_file)
                    for file in folder_list:
                        if not file:
                            pass
                        else:
                            if file.split(' ')[-1] == dir.split('.')[0] or file.split(' ')[-1] == \
                                    dir.split('.')[0].split('-')[-1]:
                                file_path = os.path.join(order_file, file)
                                original_subfile_list = os.listdir(os.path.join(order_file, file))
                                for inner_file in original_subfile_list:
                                    doc_dirname_path = os.path.join(file_path, inner_file)
                                    if inner_file.endswith('doc'):
                                        try:
                                            docxfile = doc2docx(doc_dirname_path)
                                            doc = docx.Document(docxfile)
                                            work_content_list = []
                                            for tbl in doc.tables:
                                                for row in tbl.rows:
                                                    tc_list = row._tr.tc_lst
                                                    for tc_item in tc_list:
                                                        this_tc = tc_item
                                                        this_plist = list(this_tc.iter_block_items())
                                                        for part_index, part in enumerate(this_plist):
                                                            row_dict = xml_to_dict(part.xml)
                                                            row_content = row_dict['p'].replace('\n', ' ').strip()
                                                            work_content_list.append(row_content)
                                            for content_index in range(len(work_content_list)):
                                                if work_content_list[content_index] == '工程要求':
                                                    extract_content = work_content_list[content_index + 1:]
                                                    for work_content in extract_content:
                                                        content_list.append(work_content)
                                                        original_order_list.append(original_order_name)
                                                        original_file_name = file + '.zip'  # 原始订单名
                                                        doc_name = inner_file
                                                        original_file_list.append(original_file_name)
                                                        article_name_list.append(doc_name)
                                                        content_title_list.append('工程要求')
                                                        bo74_df = pd.DataFrame(
                                                            np.arange(len(content_list) * 4).reshape(len(content_list),
                                                                                                     4))
                                                        bo74_df['原始订单名'] = original_order_list
                                                        bo74_df['原始文件名'] = original_file_list
                                                        bo74_df['doc文件名'] = article_name_list
                                                        bo74_df['提取项'] = content_title_list
                                                        bo74_df['提取内容'] = content_list
                                                        bo74_df.to_excel('datas_of_B074.xlsx')
                                        except Exception as e:
                                            print(e)

                                    if os.path.isdir(doc_dirname_path) == True:  # 如果与doc文件同目录的另外一个文件是目录
                                        innerest_file_list = os.listdir(doc_dirname_path)  # 最里层文件
                                        for innerest_file in innerest_file_list:
                                            if innerest_file.endswith('pdf'):
                                                pdf_path = os.path.join(doc_dirname_path, innerest_file)  # pdf路径
                                                try:
                                                    extract_pdf = get_pdf_table(pdf_path)
                                                    if not extract_pdf:
                                                        pass
                                                    else:
                                                        pdf_name = innerest_file
                                                        for content_num in range(len(extract_pdf)):
                                                            text_start_str = extract_pdf[content_num][5][-1].split('.')[
                                                                0]
                                                            if text_start_str.isdigit():
                                                                tecnologhsquirements_content = \
                                                                    extract_pdf[content_num][5][-1]
                                                                content_list.append(tecnologhsquirements_content)
                                                                original_file_name = file + '.zip'
                                                                content_title_list.append('技术要求')
                                                                original_order_list.append(original_order_name)
                                                                article_name_list.append(pdf_name)
                                                                original_file_list.append(original_file_name)
                                                                bo74_df = pd.DataFrame(
                                                                    np.arange(len(content_list) * 4).reshape(
                                                                        len(content_list), 4))
                                                                bo74_df['原始订单名'] = original_order_list
                                                                bo74_df['原始文件名'] = original_file_list
                                                                bo74_df['doc文件名'] = article_name_list
                                                                bo74_df['提取项'] = content_title_list
                                                                bo74_df['提取内容'] = content_list
                                                                bo74_df.to_excel('datas_of_B074.xlsx')

                                                except Exception as e:
                                                    print(e)

        except Exception as e:
            print(e)



if __name__ == "__main__":
    decompress()
