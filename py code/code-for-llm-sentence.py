import os
import re
import pandas as pd
import xmltodict
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from string import punctuation
from nltk.tokenize import RegexpTokenizer
import gc
from itertools import islice
import json
import glob

# import of the basis lib
from ollama import chat
from ollama import ChatResponse
from ollama import Client

# 指定文件夹路径
folder_path = 'C:/Users/Pratt/Desktop/申请内容/科研内容/Harry/small'  # 替换为你的实际文件夹路径

# 获取文件夹中的所有文件名
bbb = os.listdir(folder_path)

def import_deceptivewords(file):
    df = pd.read_csv(file, sep=',')
    df["Word"]=df["Word"].str.strip()
    tokeep = ['Word','Positive']
    positive = set([x['Word'].lower() for idx, x in df[tokeep].iterrows()
                    if x['Positive'] > 0])
    tokeep = ['Word','Negative']
    negative = set([x['Word'].lower() for idx, x in df[tokeep].iterrows()
                    if x['Negative'] > 0])
    return {'positive':positive, 'negative':negative}

def import_politicalbigrams(file):
    df = pd.read_csv(file, sep=',', encoding='utf-8')
    df = df.assign(bigram=df['bigram'].str.replace('_', ' '))
    df = df.assign(bigram=df['bigram'].str.replace('-', ' '))
    df.rename(columns={'politicaltbb':'tfidf'}, inplace=True)
    df['bigram'] = df['bigram'].str.strip()
    df['bigram'] = df['bigram'].str.lower()
    df = df.drop_duplicates().reset_index()
    df = df[["bigram", "tfidf"]]
    df.set_index('bigram', inplace=True)
    return df.to_dict(orient='index')

def clean_title(string):
    
    # Remove leading and trailing white space
    string = re.sub(r'^\s+','', string)
    string = re.sub(r'\s+$', '', string)
    
    # Remove The Motley Fool
    string = re.sub(r' \| [\s\S]+$', '', string)
    
    return string

def delete_names(text_ori):
    ##删除以[数字]结尾的行
    parts = text_ori.split('--------------------------------------------------------------------------------')
    parts_without_names = ''
    numeric_tail = re.compile(r'\[\d+\]$')
    for i in parts:
        #print(i)
        #print(parts[i])
        piece = i.rstrip()
        if not numeric_tail.search(piece):
            parts_without_names = parts_without_names + piece        
    ##print(parts_without_names) 
    return parts_without_names

def non_articles(text):
    tokens = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    no_articles = [t for t in tokens]
    # articles_list 为一层过滤，过滤需求的article #
    ##ps = PorterStemmer()
    ##lemmatized = [ps.stem(t) for t in no_articles]
    ##wnl = WordNetLemmatizer()
    ##lemmatized = [wnl.lemmatize(t) for t in no_articles]
    return no_articles

def split_narr_qa(text_ori):
    ## 将MD与Q&A区分开来
    no_indexes = 0
    text_split = text_ori.split()

    a = text_ori.find("Presentation")
    b = text_ori.find("Questions and Answers")
    b0 = text_ori.find("QandA")
    b1 = text_ori.find("Q-and-A")
    b2 = text_ori.find("q-n-a")
    b3 = text_ori.find("q and a")
    c = text_ori.find("Transcript")

    if a !=-1 and b !=-1:
        text_narr1 = text_ori[a+12:b]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        text_qa1 = text_ori[b+21:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        #text_all = text_narr + text_qa

    if a != -1 and b == -1 and b0 != -1:
        text_narr1 = text_ori[a+12:b0]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        text_qa1 = text_ori[b0+5:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        #text_all = text_narr + text_qa

    if a != -1 and b == -1 and b0 == -1 and b1 != -1:
        text_narr1 = text_ori[a+12:b1]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        text_qa1 = text_ori[b1+7:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        #text_all = text_narr + text_qa
    
    if a != -1 and b == -1 and b0 == -1 and b1 == -1 and b2 != -1:
        text_narr1 = text_ori[a+12:b2]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        text_qa1 = text_ori[b2+5:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        #text_all = text_narr + text_qa
        
    if a != -1 and b == -1 and b0 == -1 and b1 == -1 and b2 == -1 and b3 != -1:
        text_narr1 = text_ori[a+12:b3]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        text_qa1 = text_ori[b3+7:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        #text_all = text_narr + text_qa

    if a != -1 and b == -1 and b0 == -1 and b1 == -1 and b2 == -1 and b3 == -1:
        text_narr1 = text_ori[a+12:]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        text_qa =''
        #text_all = text_narr

    if a == -1 and b != -1:
        text_qa1 = text_ori[b+21:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        text_narr1 = text_ori[:b]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        #text_all = text_narr + text_qa            

    if a == -1 and b == -1 and b0 != -1:
        text_qa1 = text_ori[b0+5:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        text_narr1 = text_ori[:b0]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        #text_all = text_narr + text_qa

    if a == -1 and b == -1 and b0 == -1 and b1 != -1:
        text_qa1 = text_ori[b1+7:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        text_narr1 = text_ori[:b1]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        #text_all = text_narr + text_qa
    
    if a == -1 and b == -1 and b0 == -1 and b1 == -1 and b2 != -1:
        text_qa1 = text_ori[b2+5:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        text_narr1 = text_ori[:b2]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        #text_all = text_narr + text_qa
    
    if a == -1 and b == -1 and b0 == -1 and b1 == -1 and b2 == -1 and b3 != -1:
        text_qa1 = text_ori[b3+7:]
        text_split2 = text_qa1.split()
        text_qa = ' '.join(text_split2)
        text_narr1 = text_ori[:b3]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        #text_all = text_narr + text_qa
    
    ##有问题，未解决。存在presentation和Q&A都没有标识的情况
    if a == -1 and b == -1 and b0 == -1 and b1 == -1 and b2 == -1 and b3 == -1:
        text_narr1 = text_ori[c+10:]
        text_split1 = text_narr1.split()
        text_narr = ' '.join(text_split1)
        ##print(text_narr1)
        #text_all = text_narr
        text_qa = ''
        no_indexes = 1    
    return [no_indexes, text_narr, text_qa]

def extra_execu_narr_qa(text_narr, text_qa, corporate_participants):
    ##1. 提取narrative part的高管语言
    parts_narr = text_narr.split('--------------------------------------------------------------------------------')
    ##print(parts_narr)
    parts_narr_execu = ''    
    speaking_index_narr = []
    numeric_tail = re.compile(r'\[\d+\]$')
    for i in range(len(parts_narr)):
        #print(i)
        #print(parts[i])
        piece_narr = parts_narr[i].rstrip()
        if (piece_narr.startswith(tuple(corporate_participants)) and numeric_tail.search(piece_narr)):
            speaking_index_narr.append(i+1)
            ##print(parts_narr[i+1])
    #print(speaking_index_narr)       
    for i in speaking_index_narr:
        parts_narr_execu = parts_narr_execu + parts_narr[i]           
    #print(parts_narr_execu)
    
    ##2. 提取Q&A的高管语言
    parts_qa = text_qa.split('--------------------------------------------------------------------------------')
    #print(parts_qa)
    parts_qa_execu = ''    
    speaking_index_qa = []
    numeric_tail = re.compile(r'\[\d+\]$')
    for i in range(len(parts_qa)):
        #print(i)
        #print(parts[i])
        piece_qa = parts_qa[i].rstrip()
        if (piece_qa.startswith(tuple(corporate_participants)) and numeric_tail.search(piece_qa)):
            #print(parts_narr[i])
            speaking_index_qa.append(i+1)
    #print(speaking_index_qa)        
    for i in speaking_index_qa:
        parts_qa_execu = parts_qa_execu + parts_qa[i]           
    ##print(parts_qa_execu)                   
    parts_all_execu = parts_narr_execu + parts_qa_execu    
    return [parts_all_execu, parts_narr_execu, parts_qa_execu]

def load_transcripts_xml_new_2(folder):
    #corporate_participants = [' Unidentified Speaker',' Unidentified Company Representative']
    results = {} 
    domain = os.path.abspath(folder)
    for info in os.listdir(folder):
        info1 = os.path.join(domain,info) 
        # files_all = [x for x in os.listdir(info1) if '.xml' in x] 
        files_all = [os.path.basename(info1)]  # Direct load the file name (since info1 being files)
        number_of_file=0
        for name in files_all:
            if name not in bbb: continue
            with open(info1, 'r', errors = 'ignore', encoding='utf-8') as fd:
                original_data = xmltodict.parse(fd.read())
                aa_data = pd.json_normalize(original_data)
                text_ori = aa_data['Event.EventStory.Body'][0]
                title = clean_title(aa_data['Event.eventTitle'][0])
                ticker = aa_data['Event.companyTicker'][0]
                date = aa_data['Event.startDate'][0]
                time = aa_data['Event.@lastUpdate'][0]
                file_name = name
                number_of_file=number_of_file+1
                if number_of_file%10==0:
                    print("Preprocess Working On:", file_name, ticker, date)
                fd.close()
            
            #text_ori = aa_data['Event.EventStory.Body'][0]     

        # 3. extract the executive name list: "corporate_participants"
        # 3.1 some transcripts use these phrases to identify corporate participants
                corporate_participants = [' Unidentified Speaker',' Unidentified Company Representative'] 
        # 3.2 dummy for whether we can find any executive names. Default is no.
                find_corporate_participants = 0
                try:
                    corporate_participants_text = [x for x in re.findall('(?<=\=).+?(?=\=)', text_ori, re.DOTALL) if '*' in x][0].split('*')
                    for i in corporate_participants_text[1:]: 
                        corporate_participant = i.split('\n')[0].strip()
                        corporate_participant_fir2 = ' '.join(corporate_participant.split()[:2])
                        # 3.3 generate the executive name list - "corporate_participants"
                        corporate_participants.append(" " + corporate_participant_fir2)
                except:
                    #print("Fail to look for corporate participants")
                    pass
        # 3.4 if we can identify any specific executive name, change the dummy to one
                if  len(corporate_participants) > 2:
                    find_corporate_participants = 1

        # 4. split the original transcript into narrative and Q&A part. Combine them to the whole part
        # 4.1 apply the function "split_narr_qa" to get the narrative part and Q&A part
                split_narr_qa_list = split_narr_qa(text_ori)
                no_indexes = split_narr_qa_list[0]
                text_narr = split_narr_qa_list[1]
                text_qa = split_narr_qa_list[2]

        # 5. get the executive wording in the all content, narrative part, and Q&A part, respectively: "parts_narr_execu", "parts_qa_execu", and "parts_all_execu"
        # 5.1 Cases that we cannot find any executive names in the transcript
                if find_corporate_participants == 0:
                    parts_narr_execu = delete_names(text_narr)
                    parts_qa_execu = delete_names(text_qa)
                    parts_all_execu = parts_narr_execu + parts_qa_execu
                    tokens_all = non_articles(parts_all_execu)
                    if len(tokens_all) < 150: continue
                    tokens_narr = non_articles(parts_narr_execu)
                    tokens_qa = non_articles(parts_qa_execu)
            # 5.1.1 if the length of the respective part is less than 150, assume there is no material content in that part
                    if len(tokens_narr) < 150 :
                        parts_narr_execu = ""
                        tokens_narr = []
                    if len(tokens_qa) < 150:
                        parts_qa_execu = ""
                        tokens_qa = []
        # 5.2 Cases that we can find any executive names in the transcript
                if find_corporate_participants == 1:
                    split_execu_narr_qa = extra_execu_narr_qa(text_narr, text_qa, corporate_participants)
                    parts_all_execu = split_execu_narr_qa[0]
                    parts_narr_execu =  split_execu_narr_qa[1]
                    parts_qa_execu = split_execu_narr_qa[2]
                    tokens_all = non_articles(parts_all_execu)
                    if len(tokens_all) < 150: continue            
                    tokens_narr = non_articles(parts_narr_execu)
                    tokens_qa = non_articles(parts_qa_execu)
                    if len(tokens_narr) < 150 :
                        parts_narr_execu = ""
                        tokens_narr = []
                    if len(tokens_qa) < 150:
                        parts_qa_execu = ""
                        tokens_qa = []

            results[title] = {'text_all':parts_all_execu,
                              'text_narr':parts_narr_execu,
                              'text_qa':parts_qa_execu,
                          'date':date,
                          'time':time,
                          'ticker':ticker,
                          'filename': file_name}
            #fd.close()
    return results 

def split_and_store_transcripts(transcripts_raw, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for title, content in transcripts_raw.items():
        filename = content['filename']
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_seperate.json"
        output_path = os.path.join(output_folder, output_filename)
        data_to_save = {
            'title': title,
            'content': content
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    print(f"成功拆分并存储了 {len(transcripts_raw)} 个文件到 {output_folder}")

def preprocess_all_new(input_file, output_folder, window_size=20):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    title = data['title']
    content = data['content']
    
    # 创建结果字典
    result = {
        'title': title,
        'content': content.copy()  # 创建内容副本
    }
    
    # 获取文本内容并按句号分割
    text_str_all = content['text_all']
    sentences_all = [s.strip() for s in text_str_all.split('.') if s.strip()]
    
    text_str_narr = content['text_narr']
    sentences_narr = [s.strip() for s in text_str_narr.split('.') if s.strip()]
    
    text_str_qa = content['text_qa']
    sentences_qa = [s.strip() for s in text_str_qa.split('.') if s.strip()]
    
    # 预处理每个句子：转换为小写并移除标点
    processed_sentences_all = []
    for sentence in sentences_all:
        cleaned = re.sub(r'[^a-zA-Z ]', '', sentence.lower())
        processed_sentences_all.append(cleaned.split())
    
    processed_sentences_narr = []
    for sentence in sentences_narr:
        cleaned = re.sub(r'[^a-zA-Z ]', '', sentence.lower())
        processed_sentences_narr.append(cleaned.split())
    
    processed_sentences_qa = []
    for sentence in sentences_qa:
        cleaned = re.sub(r'[^a-zA-Z ]', '', sentence.lower())
        processed_sentences_qa.append(cleaned.split())
    
    # 更新结果
    result['content']['sentences_all'] = processed_sentences_all
    result['content']['sentences_narr'] = processed_sentences_narr
    result['content']['sentences_qa'] = processed_sentences_qa
    result['content']['original_sentences_all'] = sentences_all
    result['content']['original_sentences_narr'] = sentences_narr
    result['content']['original_sentences_qa'] = sentences_qa

    # 创建输出文件名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    base_name = base_name.replace('_seperate', '_process')
    output_path = os.path.join(output_folder, f"{base_name}.json")
    
    # 写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"成功处理并存储了 {output_path}")

# stream response # 
# 调用本地的性能进行 #
def LLM_judge(sentence): 
    answer_llama = []
    question = f"""Please read this sentence:{sentence}. Please judge whether this sentence is talking about climate change, your answer should only be a word: yes/no. Do not return any explanation."""

    # 指定特定模型
    stream = chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 
                'content': question}],
        stream=False,
    )
    answer_llama.append(stream['message']['content'].strip().lower())
    cleaned_list = [re.sub(r'[^\w\s]', '', s) for s in answer_llama]
    return cleaned_list

def LLM_Score(sentence): 
    question = f"""Please read this text: {sentence}. Please rate the text on a scale of 0-100, with 0 being completely honest and 100 being extremely deceptive. Your answer MUST be a single integer from 0 to 100. Do not return any explanation or words, just the integer."""

    # 指定特定模型
    stream = chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 
                'content': question}],
        stream=False,
    )
    score = int(stream['message']['content'].strip())
    return score

def process_scores_file(seperate_file, processed_folder, political_bigrams):
    """处理单个文件并计算scores"""
    scores = {}
    answer_all = []
    answer_narr = []
    answer_qa = []
    try:
        # 第一步：生成中间处理文件
        preprocess_all_new(seperate_file, processed_folder)
        
        # 第二步：加载刚生成的中间文件
        processed_file = os.path.join(
            processed_folder,
            os.path.basename(seperate_file).replace('_seperate', '_process')
        )
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        content = data['content']
        title = data['title']
        # 提取标题
        progressfile = os.path.basename(seperate_file)
        print('Working on:', progressfile)
        scores[title] = {}
        
        # 获取所有句子数据
        sentences_all = content['sentences_all']
        original_sentences_all = content['original_sentences_all']
        sentences_narr = content['sentences_narr']
        original_sentences_narr = content['original_sentences_narr']
        sentences_qa = content['sentences_qa']
        original_sentences_qa = content['original_sentences_qa']
        cc_date = content['date']
        cc_ticker = content['ticker']
        file_name = content['filename']
        
        # 计算总词数
        totalwords = sum(len(s) for s in sentences_all)
        totalwords_narr = sum(len(s) for s in sentences_narr)
        totalwords_qa = sum(len(s) for s in sentences_qa)
        
        # 初始化scores
        scores[title] = {
            'Total words': totalwords,
            'Total narr_words': totalwords_narr,
            'Total qa_words': totalwords_qa,
            'DATE': cc_date,
            "tic": cc_ticker,
            "filename": file_name
        }

        # 1. 首先对原始文本进行欺骗性评分
        if original_sentences_all:
            full_text_all = '. '.join(original_sentences_all) + '.' if original_sentences_all else ''
            scores[title]['Deceptive_original_all'] = LLM_Score(full_text_all)
        if original_sentences_narr:
            full_text_narr = '. '.join(original_sentences_narr) + '.' if original_sentences_narr else ''
            scores[title]['Deceptive_original_narr'] = LLM_Score(full_text_narr)
        if original_sentences_qa:
            full_text_qa = '. '.join(original_sentences_qa) + '.' if original_sentences_qa else ''
            scores[title]['Deceptive_original_qa'] = LLM_Score(full_text_qa)

        # 处理all文本
        processed_sentences_all = set()  # 用于去重
        for words, original_sentence in zip(sentences_all, original_sentences_all):
            # 生成unigrams, bigrams, trigrams
            unigrams = words
            bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            
            # 检查是否包含政治相关词汇
            has_political = (any(word in political_bigrams for word in unigrams) or 
                           any(bigram in political_bigrams for bigram in bigrams) or 
                           any(trigram in political_bigrams for trigram in trigrams))
            
            if has_political and original_sentence not in processed_sentences_all:
                answer = LLM_judge(original_sentence)
                if answer[0] == 'yes':
                    answer_all.append(original_sentence)
                    answer_all.append(". ")
                    processed_sentences_all.add(original_sentence)

        # 处理narrative文本
        processed_sentences_narr = set()
        for words, original_sentence in zip(sentences_narr, original_sentences_narr):
            unigrams = words
            bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            
            has_political = (any(word in political_bigrams for word in unigrams) or 
                           any(bigram in political_bigrams for bigram in bigrams) or 
                           any(trigram in political_bigrams for trigram in trigrams))
            
            if has_political and original_sentence not in processed_sentences_narr:
                answer = LLM_judge(original_sentence)
                if answer[0] == 'yes':
                    answer_narr.append(original_sentence)
                    answer_narr.append(". ")
                    processed_sentences_narr.add(original_sentence)

        # 处理qa文本
        processed_sentences_qa = set()
        for words, original_sentence in zip(sentences_qa, original_sentences_qa):
            unigrams = words
            bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            
            has_political = (any(word in political_bigrams for word in unigrams) or 
                           any(bigram in political_bigrams for bigram in bigrams) or 
                           any(trigram in political_bigrams for trigram in trigrams))
            
            if has_political and original_sentence not in processed_sentences_qa:
                answer = LLM_judge(original_sentence)
                if answer[0] == 'yes':
                    answer_qa.append(original_sentence)
                    answer_qa.append(". ")
                    processed_sentences_qa.add(original_sentence)

        # 2. 对气候相关文本进行欺骗性评分
        if answer_all:
            combined_climate_text_all = ''.join(answer_all)
            scores[title]['Deceptive_climate_filtered_all'] = LLM_Score(combined_climate_text_all)
            scores[title]['Deceptive_climate_text_all'] = combined_climate_text_all
        else:
            scores[title]['Deceptive_climate_filtered_all'] = 0
            scores[title]['Deceptive_climate_text_all'] = None
            
        if answer_narr:
            combined_climate_text_narr = ''.join(answer_narr)
            scores[title]['Deceptive_climate_filtered_narr'] = LLM_Score(combined_climate_text_narr)
            scores[title]['Deceptive_climate_text_narr'] = combined_climate_text_narr
        else:
            scores[title]['Deceptive_climate_filtered_narr'] = 0
            scores[title]['Deceptive_climate_text_narr'] = None
            
        if answer_qa:
            combined_climate_text_qa = ''.join(answer_qa)
            scores[title]['Deceptive_climate_filtered_qa'] = LLM_Score(combined_climate_text_qa)
            scores[title]['Deceptive_climate_text_qa'] = combined_climate_text_qa
        else:
            scores[title]['Deceptive_climate_filtered_qa'] = 0
            scores[title]['Deceptive_climate_text_qa'] = None

        # 删除中间文件
        os.remove(processed_file)
        print(f"Processed and cleaned: {os.path.basename(seperate_file)}")
        
    except Exception as e:
        print(f"Error processing {seperate_file}: {str(e)}")
    
    return scores, answer_all, answer_narr, answer_qa


# Files
earningscall_dir_new = 'C:/Users/Pratt/Desktop/申请内容/科研内容/Harry/small'
deceptivewords_file = 'deceptive_list_clean.csv'
polbigrams_file = 'cc_words.csv'
deceptive_words = import_deceptivewords(deceptivewords_file)
allwords1 = dict(deceptive_words)

# Import political bigrams
political_bigrams = import_politicalbigrams(polbigrams_file)
transcripts_raw = load_transcripts_xml_new_2(earningscall_dir_new)
split_and_store_transcripts(transcripts_raw, 'seperate_files_small_byllm_V2_S')
seperate_files = glob.glob('seperate_files_small_byllm_V2_S/*_seperate.json')
processed_folder = 'processed_files_small_byllm_V2_S'
os.makedirs(processed_folder, exist_ok=True)

scores = {}
for seperate_file in seperate_files:
    file_scores,answers_all,answers_narr,answers_qa = process_scores_file(seperate_file, processed_folder, political_bigrams)
    scores.update(file_scores)

# 创建DataFrame
df_scores = pd.DataFrame().from_dict(scores, orient='index')
df_scores.index.name = 'event name'

print(df_scores)