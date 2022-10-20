import multiprocessing, time, csv
import numpy as np
from text_rankcopy import query_predict
import evaluate, pickle
from pathlib import Path
import nltk
metric = evaluate.load('rouge')


exclude_idx = [2320, 4923, 5210]
def main():
    pool = multiprocessing.Pool(4)
    start_time = time.perf_counter()
    quries = list()
    with open('sums500.pickle', 'rb') as f:
        summaries = pickle.load(f)
    summaries = [summ for pair in summaries for summ in pair]
    print(len(summaries[0].split()))
    with open('outputnew400.csv', 'r') as f:
        reader = csv.DictReader(f)
        i = 0
        for row in reader:
            quries.append((row['index'],summaries[i].replace("<S>", "").replace("</S>",""), row['abstract']))
            i+=1

    
    length = 400 
    #print(quries[0][1])
    
    #processes = [pool.apply_async(query_predict, args=(quries[x][1], x, length,)) for x in range(0,10) if x not in exclude_idx]
    #print(quries)

    use_bert = True 
    use_textrank = True
    file_name = f'text_rank_results_{length}.csv'


    file = (Path(__file__).parent / file_name).open('w')
    writer = csv.writer(file)
    header = ['use_bert', 'use_textrank', 'target_length', 'avg_length_length', 'alpha', 'results']
    writer.writerow(header) 

    for alpha in np.arange(.05, 1, .05):
        alpha = round(alpha,2)
        preds = list()
        labels = list()
        lens = list()
        
        for i in range(30): 
            if use_textrank:
                pred, label = query_predict(quries[i][1], int(quries[i][0]), length, use_bert, alpha)
            else:
                #Use abstractive summary
                pred = quries[i][1]
                label = quries[i][2]
            
            preds.append(pred)
            labels.append(label)
            lens.append(len(pred.split()))

        avg_length = sum(lens)/len(lens)
        print('avg',avg_length)
        print('Used SBERT:', use_bert) 
        preds, labels = postprocess_text(preds, labels) 
        result = metric.compute(predictions=preds, references=labels, use_stemmer=True)

        writer.writerow([use_bert, use_textrank, length, avg_length, alpha, result])
    file.close()
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
if __name__ == "__main__":
    main()