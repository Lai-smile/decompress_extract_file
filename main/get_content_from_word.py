# created by lynn at 01/22/2019

import win32com.client
import pickle
import zipfile
import os
import chardet
import shutil


def word2txt(filename):
    app = win32com.client.Dispatch('Word.Application')
    app.Visible = 0
    app.DisplayAlerts = 0
    doc = app.Documents.Open(filename)
    txt_out = filename + ".txt"
    doc.SaveAs(txt_out, 2)
    # 2 for wdFormatText
    doc.Close()
    app.Quit()

    return txt_out


def get_file_encoding(filename):
    raw_data = open(filename, 'rb').read()
    return chardet.detect(raw_data)['encoding']


def get_texts_from_txt(filename):
    encoding = get_file_encoding(filename)
    lines = []
    # res={}
    with open(filename, encoding=encoding, errors='ignore') as f:
        for y, line in enumerate(f):
            # res[(0,0,y,0,0)]=line
            lines.append((0, 0, y, 0, 0, line))
    #         (i,x1,y1,x2,y2)

    return lines, encoding


def doc2docx(doc_in):
    docx_out = doc_in[:doc_in.rfind('.')] + ".docx"
    app = win32com.client.Dispatch('Word.Application')
    app.Visible = 0
    app.DisplayAlerts = 0
    doc_in = app.Documents.Open(doc_in)
    doc_in.SaveAs(docx_out, 16)
    # 16 for docx
    doc_in.Close()
    app.Quit()

    return docx_out


def extract_images(docx, dest_path=""):
    zf = os.path.splitext(docx)[0] + ".zip"
    if not os.path.exists(zf):
        os.rename(docx, zf)
    # zip docx

    extracted_path = os.path.splitext(docx)[0] + "-extracted"
    print(extracted_path)
    if not os.path.exists(extracted_path):
        os.makedirs(extracted_path)
    with zipfile.ZipFile(os.path.splitext(docx)[0] + ".zip", "r") as zip:
        zip.extractall(extracted_path)
    #     unzip xlsx

    img_path = os.path.join(extracted_path, "word", "media")
    if not os.path.exists(img_path):
        return {'img_count': 0, 'paths': dest_path}

    pic_path = os.path.splitext(docx)[0] + '-images'
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    path_out = []
    i = 0

    for file in os.listdir(img_path):
        print(file)
        if file.endswith("png") or file.endswith(
                "jpg") or file.endswith("jpeg"):
            extracted_pic = os.path.join(extracted_path, 'word', 'media', file)
            saved_pic = os.path.join(pic_path, file)
            shutil.copyfile(extracted_pic, saved_pic)
            # move image from extracted path to destination path
            path_out.append(saved_pic)
            i += 1

    os.remove(zf)
    shutil.rmtree(extracted_path, ignore_errors=True)
    # delete zip file and unzip folder

    return {'img_count': i, 'paths': path_out}


def get_images_from_word(docfile, dest_path):
    if docfile.endswith('doc'):
        docxfile = doc2docx(docfile)
    else:
        docxfile = docfile
    return extract_images(docxfile, dest_path)


def get_word_content(filename):
    txt = word2txt(filename)
    data, encoding = get_texts_from_txt(txt)
    return data


if __name__ == '__main__':
    in_path = r"C:\Users\Administrator\Desktop\test.doc"
    for file in os.listdir(in_path):
        filepath = os.path.join(in_path, file)
        data = get_word_content(filepath)
        with open(in_path + file + ".pickle", "wb") as f:
            pickle.dump(data, f)
