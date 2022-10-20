import multiprocessing, time, csv
from text_rankcopy import query_predict
import evaluate, pickle
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

    preds = list()
    labels = list()
    lens = list()
    for i in range(6630): 
        pred, label = query_predict(quries[i][1], int(quries[i][0]), length)
        
        
        
        #print('######################')
        #print('pred', pred)
        #print('label', label)
        # pred = quries[i][1]
        # label = quries[i][2]
        preds.append(pred)
        labels.append(label)

        lens.append(len(pred.split()))

    print('avg',sum(lens)/len(lens))
    
    preds, labels = postprocess_text(preds, labels) 
    #print('preds', preds[0])
    #print('labels', labels[0])
    result = metric.compute(predictions=preds, references=labels, use_stemmer=True)
    print(result)
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