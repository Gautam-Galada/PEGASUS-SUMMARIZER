
from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
from typing import List
import regex as re
from transformers import pipeline, PegasusForConditionalGeneration, PegasusTokenizer
import torch
from transformers import AutoConfig
import re
import shutil
import os, wave, math, collections
from os import path
import splitter




tok = PegasusTokenizer.from_pretrained('models')
config = AutoConfig.from_pretrained('models')
dummy_model = PegasusForConditionalGeneration(config)


quantized_model = torch.quantization.quantize_dynamic(
    dummy_model, {torch.nn.Linear}, dtype=torch.qint8
)

#quantization code is availabe in a different repo
quantized_state_dict = torch.load('pegasus-quantized.h5')
quantized_model.load_state_dict(quantized_state_dict)

#if you want to run the summarizer
app = Flask(__name__)
run_with_ngrok(app)

main_dir = ""
filename = "test.txt" #input file

def preparing_transcript(text):

    transcript = text
    if os.path.exists(main_dir):
        shutil.rmtree(main_dir)  # delete output folder
    os.makedirs(main_dir)


    doge = open(filename, "wt")
    g = doge.write(transcript)
    doge.close()


    n = len(transcript.split())

    if n<7000:
        print("a")
        AR = open('test.txt', 'r').read()
        batch = tok.prepare_seq2seq_batch(src_texts=[AR], truncation=True, padding='longest', return_tensors='pt')
        gen = quantized_model.generate(**batch)
        summary: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
        print("0")
        sum = str(summary)

    elif 7000<n<14000:
        print("b")
        n_words = (len(transcript.split())//1)
        input = 'test.txt'
        output = ''
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('test_split0.txt', 'r').read()
        AR2 = open('test_split1.txt', 'r').read()


        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')



        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)

        print("1")
        sum = str(summary1+summary2)

    elif 14000<n<21000:
        print("c")
        n_words = (len(transcript.split())//2)
        input = 'test.txt'
        output = ''
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('test_split0.txt', 'r').read()
        AR2 = open('test_split1.txt', 'r').read()
        AR3 = open('test_split2.txt', 'r').read()


        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')
        batch3 = tok.prepare_seq2seq_batch(src_texts=[AR3], truncation=True, padding='longest', return_tensors='pt')


        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)
        gen3 = quantized_model.generate(**batch3)
        summary3: List[str] = tok.batch_decode(gen3, skip_special_tokens=True)

        print("2")
        sum = str(summary1+summary2+summary3)


    elif 21000<n<28000:
        print("d")
        n_words = (len(transcript.split())//3)
        input = 'test.txt'
        output = ''
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('test_split0.txt', 'r').read()
        AR2 = open('test_split1.txt', 'r').read()
        AR3 = open('test_split2.txt', 'r').read()
        AR4 = open('test_split3.txt', 'r').read()

        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')
        batch3 = tok.prepare_seq2seq_batch(src_texts=[AR3], truncation=True, padding='longest', return_tensors='pt')
        batch4 = tok.prepare_seq2seq_batch(src_texts=[AR4], truncation=True, padding='longest', return_tensors='pt')

        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)
        gen3 = quantized_model.generate(**batch3)
        summary3: List[str] = tok.batch_decode(gen3, skip_special_tokens=True)
        gen4 = quantized_model.generate(**batch4)
        summary4: List[str] = tok.batch_decode(gen4, skip_special_tokens=True)

        print("3")
        sum = str(summary1+summary2+summary3+summary4)

    elif 35000<n<42000:
        print("e")
        n_words = (len(transcript.split())//4)
        input = 'test.txt'
        output = ''
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('test_split0.txt', 'r').read()
        AR2 = open('test_split1.txt', 'r').read()
        AR3 = open('test_split2.txt', 'r').read()
        AR4 = open('test_split3.txt', 'r').read()
        AR5 = open('test_split4.txt', 'r').read()

        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')
        batch3 = tok.prepare_seq2seq_batch(src_texts=[AR3], truncation=True, padding='longest', return_tensors='pt')
        batch4 = tok.prepare_seq2seq_batch(src_texts=[AR4], truncation=True, padding='longest', return_tensors='pt')
        batch5 = tok.prepare_seq2seq_batch(src_texts=[AR5], truncation=True, padding='longest', return_tensors='pt')

        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)
        gen3 = quantized_model.generate(**batch3)
        summary3: List[str] = tok.batch_decode(gen3, skip_special_tokens=True)
        gen4 = quantized_model.generate(**batch4)
        summary4: List[str] = tok.batch_decode(gen4, skip_special_tokens=True)
        gen5 = quantized_model.generate(**batch5)
        summary5: List[str] = tok.batch_decode(gen5, skip_special_tokens=True)

        print("4")
        sum = str(summary1+summary2+summary3+summary4+summary5)

    else:
        print("f")
        n_words = (len(transcript.split())//5)
        input = 'test.txt'
        output = ''
        pr = True
        splitter.splitter(input,output,n_words,pr)

        print(n_words)

        AR1 = open('test_split0.txt', 'r').read()
        AR2 = open('test_split1.txt', 'r').read()
        AR3 = open('test_split2.txt', 'r').read()
        AR4 = open('test_split3.txt', 'r').read()
        AR5 = open('test_split4.txt', 'r').read()
        AR6 = open('test_split5.txt', 'r').read()


        batch1 = tok.prepare_seq2seq_batch(src_texts=[AR1], truncation=True, padding='longest', return_tensors='pt')
        batch2 = tok.prepare_seq2seq_batch(src_texts=[AR2], truncation=True, padding='longest', return_tensors='pt')
        batch3 = tok.prepare_seq2seq_batch(src_texts=[AR3], truncation=True, padding='longest', return_tensors='pt')
        batch4 = tok.prepare_seq2seq_batch(src_texts=[AR4], truncation=True, padding='longest', return_tensors='pt')
        batch5 = tok.prepare_seq2seq_batch(src_texts=[AR5], truncation=True, padding='longest', return_tensors='pt')
        batch6 = tok.prepare_seq2seq_batch(src_texts=[AR6], truncation=True, padding='longest', return_tensors='pt')

        gen1 = quantized_model.generate(**batch1)
        summary1: List[str] = tok.batch_decode(gen1, skip_special_tokens=True)
        gen2 = quantized_model.generate(**batch2)
        summary2: List[str] = tok.batch_decode(gen2, skip_special_tokens=True)
        gen3 = quantized_model.generate(**batch3)
        summary3: List[str] = tok.batch_decode(gen3, skip_special_tokens=True)
        gen4 = quantized_model.generate(**batch4)
        summary4: List[str] = tok.batch_decode(gen4, skip_special_tokens=True)
        gen5 = quantized_model.generate(**batch5)
        summary5: List[str] = tok.batch_decode(gen5, skip_special_tokens=True)
        gen6 = quantized_model.generate(**batch6)
        summary6: List[str] = tok.batch_decode(gen6, skip_special_tokens=True)

        print("5")
        sum = str(summary1+summary2+summary3+summary4+summary5+summary6)
        os.remove(filename)

    return sum

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():

    text1 = request.form['text1']
    text2 = request.form['text2']
    text3 = request.form['text3']

    print(text1,text2,text3)

    new_transcript1 = text1.replace("\n", " ")
    new_transcript2 = text2.replace("\n"," ")
    new_transcript3 = text3.replace("\n"," ")

    summary1 = preparing_transcript(new_transcript1)
    summary2 = preparing_transcript(new_transcript2)
    summary3 = preparing_transcript(new_transcript3)

    new_summary1 = re.sub(r'[^\x00-\x7f]+', ' ', str(summary1))
    new_summary2 = re.sub(r'[^\x00-\x7f]+', ' ', str(summary2))
    new_summary3 = re.sub(r'[^\x00-\x7f]+', ' ', str(summary3))

    return render_template('form.html',text1=new_summary1, text2=new_summary2, text3=new_summary3)
if __name__ == "__main__":
    app.run()
