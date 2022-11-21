
import time, csv
import numpy as np
from text_rankcopy import query_predict
import evaluate, pickle
from pathlib import Path
import nltk
from datasets import load_dataset
metric = evaluate.load('rouge')

excluded_phrases=["objective", "background", "method", "conclusions", "introduction", "purpose", "methods", ":"]
extra= [ex + "." for ex in excluded_phrases]
excluded_phrases.extend(extra)
exclude_idx = [2320, 4923, 5210]
def main():
    data_split = 'test'
    use_bert = True
    use_textrank = True 
    fine_tune = False 
    make_personalization=True 
    length = 175 
    
    start_time = time.perf_counter()
    quries = list()
    #215 = 150
    name = 'sums215'
    if data_split == 'test':
        name += 'Test'
    name += '.pickle'
    with open(name, 'rb') as f:
        summaries = pickle.load(f)
    summaries = [summ for pair in summaries for summ in pair]
    for i in range(len(summaries)):
        summ = summaries[i]
        result = [word for word in summ.split() if word not in excluded_phrases] 
        summaries[i] = ' '.join(result)


        
    
    # with open('outputnew400.csv', 'r') as f:
    #     reader = csv.DictReader(f)
    #     i = 0
    #     for row in reader:
    #         quries.append((row['index'],summaries[i], row['abstract']))
    #         i+=1


    if data_split == 'validation':
        for i in sorted(exclude_idx, reverse=True):
            del summaries[i]

        raw_dataset = load_dataset("scientific_papers",'pubmed').filter(lambda example, i: i not in exclude_idx, with_indices=True)
    elif data_split == 'test':
        raw_dataset = load_dataset("scientific_papers",'pubmed')
    else:
        raise Exception

    raw_dataset.remove_columns(
        'section_names'
    )
    print('num examples ' ,len(summaries))
    raw_dataset = raw_dataset[data_split]['abstract']
    print(len(summaries))
    for i in range(len(raw_dataset)):
        quries.append((i, summaries[i], raw_dataset[i]))
    # length = 165
    print('datasplit: ', data_split, 'name: ', name, 'length', length)
    #print(quries[0][1])
    
    #processes = [pool.apply_async(query_predict, args=(quries[x][1], x, length,)) for x in range(0,10) if x not in exclude_idx]
    #print(quries)

    if fine_tune:
        file_name = f'text_rank_results_{length}.csv'


        file = (Path(__file__).parent / file_name).open('w')
        writer = csv.writer(file)
        header = ['use_bert', 'use_textrank', 'target_length', 'avg_length_length', 'alpha', 'results']
        writer.writerow(header) 
        for alpha in np.arange(.05, 1, .05):
            result, avg_length = eval(alpha, use_textrank, quries, use_bert, length, data_split)
            writer.writerow([use_bert, use_textrank, length, avg_length, alpha, result])
        file.close()
    else:
        result, avg_length = eval(.85, use_textrank, quries, use_bert, length, data_split, make_personalization) 
        print('result: ', result, 'avg_length', avg_length)
    assert(False)
    #{'rouge1': 0.25052002720900657, 'rouge2': 0.022333150302001697, 'rougeL': 0.12211520451117444, 'rougeLsum': 0.1695929493182961} 
    with open('output3', 'w') as f:
                writer = csv.writer(f)
                header = ['index', 'abstract', 'summary']
                writer.writerow(header) 
                for i in range(len(result)):
                    writer.writerow([result[i][0], result[i][1], result[i][2]])
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def eval(alpha, use_textrank, quries, use_bert, length, split, make_person_vec=True):
    alpha = round(alpha,2)
    preds = list()
    labels = list()
    lens = list()
    print('quries',len(quries))
    n_sums = len(quries)
    #n_sums = 5
    met = 0
    results= np.zeros((n_sums,2))
    j = 16
    for i in range(j,j+1): 
        if use_textrank:
            pred = query_predict(quries[i][1], int(quries[i][0]), length, use_bert, alpha, split, make_person_vec)
        else:
            #Use abstractive summary
            pred = quries[i][1]
        
        label = quries[i][2]
        preds.append(pred)
        labels.append(label)
        lens.append(len(pred.split()))
        res1 = metric.compute(predictions=[pred], references=[label], use_stemmer=True)['rouge1']
        res2 = metric.compute(predictions=[quries[i][1]], references=[label], use_stemmer=True)['rouge1']
        if i == 2175:
            print('TEST')
            print(preds[i])
    print(results.shape)
    print('diff')
    print(quries[j][1],'\n------------------------------\n' ,preds[0], 'good')
    print(results)
    print(res1, 'pred')
    print(res2, 'query')
    print('label',labels[0])
    assert(False)
    dif = results[:,0] - results[:,1]
    print(dif)
    good = np.argmax(dif)
    print(labels[good])
    print(quries[good][1],'\n------------------------------\n' ,preds[good], 'good')
    print('#################################################################')
    bad = np.argmin(dif)
    print(results[bad][0], 'pred')
    print(results[bad][1], 'query')
    print(labels[bad])
    print(quries[bad][1], '\n------------------------------\n', preds[bad], 'bad', bad)
    assert(False)
    print('labels', labels[0])
    print('pred', preds[0])
    print('query', quries[0][1])
    avg_length = sum(lens)/len(lens)
    print('avg',avg_length)
    print('Used SBERT:', use_bert) 
    preds, labels = postprocess_text(preds, labels) 
    result = metric.compute(predictions=preds, references=labels, use_stemmer=True)
    # result_q =metric.compute(predictions=[quries[0][1]], references=labels, use_stemmer=True) 
    print(result)
    # print('q',result_q)
    return result, avg_length
if __name__ == "__main__":
    main()

#alpha = .65 
#text 150 result:  {'rouge1': 0.4070705000529794, 'rouge2': 0.15159560136623784, 'rougeL': 0.22243375289372452, 'rougeLsum': 0.36379194448776786} avg_length 150.57284469810753
#Base 150 result:  {'rouge1': 0.4246288910409251, 'rouge2': 0.1678186785195749, 'rougeL': 0.2457939975369589, 'rougeLsum': 0.38559672043164006} avg_length 153.50540702913787 
#alt 150 result:  {'rouge1': 0.4193315798837814, 'rouge2': 0.167791312435773, 'rougeL': 0.23633832774953273, 'rougeLsum': 0.37617395543467724} avg_length 149.47116251126465

#alpha = .55
#text 200 result:  result:  {'rouge1': 0.4295850209554639, 'rouge2': 0.16322533725768304, 'rougeL': 0.22904994348352706, 'rougeLsum': 0.38597169614902643} avg_length 200.29483328326825
#BASE 200: result:  {'rouge1': 0.4384297459114671, 'rouge2': 0.17165549308935663, 'rougeL': 0.24742049645667477, 'rougeLsum': 0.3988893505336946} avg_length 200.29648543106038 
#alt 200  result:  {'rouge1': 0.4394663521152846, 'rouge2': 0.17699925008612438, 'rougeL': 0.24115463134820786, 'rougeLsum': 0.3958173951947783} avg_length 199.5806548513067

#alpha = .55
#text result:  {'rouge1': 0.40252383125137203, 'rouge2': 0.16906013803666806, 'rougeL': 0.2165120046576241, 'rougeLsum': 0.36745473390049355} avg_length 399.3552117753079
#base 400 result:  {'rouge1': 0.36356715999224615, 'rouge2': 0.14373190886084092, 'rougeL': 0.19893087486609184, 'rougeLsum': 0.33371071097728333} avg_length 397.7711024331631
#alt 400 result:  {'rouge1': 0.40940240429803865, 'rouge2': 0.17779892652457036, 'rougeL': 0.22473145583172444, 'rougeLsum': 0.3742918365028922} avg_length 398.9755181736257 