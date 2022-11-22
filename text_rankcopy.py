from math import exp
import torch
from pickle import FALSE
from pydoc import doc
from typing import Set
import numpy as np
from collections import defaultdict
from math import log
from graph import Graph
from datasets import load_dataset
from datasets import load_metric
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
from sklearn.preprocessing import normalize
import pickle, csv
from sentence_transformers import SentenceTransformer, util
import multiprocessing, time
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def main():
    

    example = '''    
Aug 30 (Reuters) - Mikhail Gorbachev, who ended the Cold War without bloodshed but failed to prevent the collapse of the Soviet Union, died on Tuesday at the age of 91, hospital officials in Moscow said.

Gorbachev, the last Soviet president, forged arms reduction deals with the United States and partnerships with Western powers to remove the Iron Curtain that had divided Europe since World War Two and bring about the reunification of Germany.

Advertisement · Scroll to continue
But his internal reforms helped weaken the Soviet Union to the point where it fell apart, a moment that President Vladimir Putin has called the "greatest geopolitical catastrophe" of the twentieth century.

"Mikhail Gorbachev passed away tonight after a serious and protracted disease," said Russia's Central Clinical Hospital.

Putin expressed "his deepest condolences", Kremlin spokesman Dmitry Peskov told Interfax. "Tomorrow he will send a telegram of condolences to his family and friends," he said.

Putin said in 2018 he would reverse the Soviet Union's disintegration if he could, news agencies reported.

World leaders were quick to pay tribute. European Commission chief Ursula von der Leyen said Gorbachev, who was awarded the Nobel Peace Prize in 1990, had opened the way for a free Europe.

U.S. President Joe Biden said he had believed in "glasnost and perestroika – openness and restructuring – not as mere slogans, but as the path forward for the people of the Soviet Union after so many years of isolation and deprivation."

British Prime Minister Boris Johnson, citing Putin's invasion of Ukraine, said Gorbachev's "tireless commitment to opening up Soviet society remains an example to us all".

WESTERN PARTNERSHIPS
After decades of Cold War tension and confrontation, Gorbachev brought the Soviet Union closer to the West than at any point since World War Two.

"He gave freedom to hundreds of millions of people in Russia and around it, and also half of Europe," said former Russian liberal opposition leader Grigory Yavlinsky. "Few leaders in history have had such a decisive influence on their time."

But Gorbachev saw his legacy wrecked late in life, as the invasion of Ukraine brought Western sanctions crashing down on Moscow, and politicians in both Russia and the West began to speak of a new Cold War.

"Gorbachev died in a symbolic way when his life's work, freedom, was effectively destroyed by Putin," said Andrei Kolesnikov, senior fellow at the Carnegie Endowment for International Peace.'''


    sentences = [sent for sent in doc.sents]
    print(sentences)
    
    tf_idf_vec_table = create_norm_tfidf_vec(sentences)
    assert(False)
    xv, yv = np.meshgrid(np.arange(len(sentences)),np.arange(len(sentences)), indexing='ij')
    print(xv, yv)
    similarity_table = np.zeros(xv.shape)
    print(similarity_table)

    for i in range(len(xv)):
        for j in range(len(yv)):
            if xv[i,j] != yv[i,j]:
                similarity_table[xv[i,j], yv[i,j]] = tf_idf_vec_table[sentences[xv[i,j]]] @ tf_idf_vec_table[sentences[yv[i,j]]]
    
    # test = nx.Graph(similarity_table)
    # print('test', test)
    # pr = nx.pagerank(nx.Graph(similarity_table))
    # print(pr.values())
    # ranked = np.argsort(list(pr.values()))[::-1][:5]
    # print(ranked)
    # for rank in ranked:
    #     print(sentences[rank])

    # assert(False)
    graph = Graph(similarity_table)
    graph = generate_score(graph)
    print(sum(graph.get_scores()))
    ranked = np.argsort(graph.get_scores())[::-1][:5]
    
    #print(ranked)
    for rank in ranked:
        print(sentences[rank])
    #generate_score(graph,d)
    #print(sum(generate_score(graph, d).get_scores()))

    similarity_table = dict()    
    for sent1 in sentences:
       for sent2 in sentences: 
           similarity_table[(sent1,sent2)] = tf_idf_vec_table[sent1] @ tf_idf_vec_table[sent2]

#    print(similarity_table)

def generate_score(graph: Graph, personalization = None, d:float=.85):
    i = 0
    n = graph.get_num_nodes()

    if type(personalization) == type(None):
        personalization = torch.Tensor(np.full((n),1/n))
    assert(personalization.shape[0] == n)

    M = torch.Tensor(graph.get_negihbor_weights() * graph.get_weighted_list())


    for i in range(50):
        old_score = torch.Tensor(graph.get_scores())


        new_score = d*(M @ old_score) + (1-d)*personalization
        graph.set_scores(new_score)

        if torch.sum(torch.abs(old_score-new_score)) <= .0000001:
            return graph

    return graph





def create_norm_tfidf_vec(sentences):
    print(sentences)
    tf_idf_vec = {sent: [] for sent in sentences}
    terms = sorted(set([tok.lemma_ for sent in sentences for tok in sent if not tok.is_stop]))
    

    tf_idf = compute_doc_tf_idf(sentences)
    
    for sent in sentences:
        for term in terms:
            tf_idf_vec[sent].append(tf_idf[(term,sent)])
        tf_idf_vec[sent] /= np.linalg.norm(tf_idf_vec[sent])

    return tf_idf_vec


def compute_setence_tf(sentence):
    sentence_tf_table = defaultdict(lambda: 0)

    for tok in sentence:
        lem = tok.lemma_
        if lem in sentence_tf_table:
            sentence_tf_table[lem] += 1
        else:
            sentence_tf_table[lem] = 1

    return sentence_tf_table


def compute_tf_table(sentences):
    tf_table = dict()
    
    for sent in sentences:
        tf_table[sent] = compute_setence_tf(sent)

    return tf_table

def compute_doc_freq(terms, tf_table):
    document_freq = dict.fromkeys(terms, 0)

    for term in terms:
        for sent, sentence_tf_table in tf_table.items():
            if sentence_tf_table[term] > 0:
                document_freq[term] += 1
        print(term, document_freq[term])
    print(sum(document_freq.values()))
    return document_freq

def compute_doc_tf_idf(sentences):
    tf_table = compute_tf_table(sentences)
    terms = set([tok.lemma_ for sent in sentences for tok in sent if not tok.is_stop])
    document_freq = compute_doc_freq(terms, tf_table)

    return create_tf_idf(terms, tf_table, document_freq, len(tf_table.keys()))

    

def create_tf_idf(terms, tf_table, document_freq, N):
    tf_idf_table = defaultdict(lambda: 0)
    for term in terms:
        for sent, sentence_tf_table in tf_table.items():
            if(sentence_tf_table[term] == 0):
                tf_idf = 0
            else:
                tf_idf = sentence_tf_table[term] * (log(N / (document_freq[term]+1)) + 1)
    #        print('TERMS: ', term, 'Sentence: ', sent, 'TF:', sentence_tf_table[term], 'Doc_freq:',  document_freq[term], 'TF_IDF',tf_idf)
            tf_idf_table[(term, sent)] = tf_idf 

    return tf_idf_table, document_freq

def compute_query_tf_idf(query, doc_freq, N):
    terms = sorted(doc_freq.keys())
    tf_table = compute_tf_table(query)
    print(tf_table.keys())
    print('--------------------------------------------------------------------------')
    query_tfidf, _ = create_tf_idf(terms, tf_table, doc_freq, N)
    final_query_tfidf = dict()
    print(query_tfidf)

    query_total = sum(final_query_tfidf.values())
    for k, v in query_tfidf.items():
        final_query_tfidf[k[0]] = v/query_total
    
    
    return final_query_tfidf



def soft_max(x):
    return np.exp(x)/sum(np.exp(x))

    
dataset = None 

def load_data(split): 
    global dataset 
    if dataset == None:
        if split == 'test':
            dataset = load_dataset("scientific_papers",'pubmed')
        elif split == 'validation':
            dataset = load_dataset("scientific_papers",'pubmed').filter(lambda example, i: i not in exclude_idx, with_indices=True)
        else:
            raise Exception
        dataset.remove_columns(
            'section_names'
        )
        print(dataset)
        print(split)
    return dataset
#when2meet
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
torch.set_default_tensor_type('torch.FloatTensor')
def query_predict(abs_sum, doc_id, max_len, bart_embed=False, alpha=.1, split='validation', make_personalizaton=True) -> str:
    dataset = load_data(split)
    print('Doc', doc_id, flush=True, end='\r')
    if type(abs_sum) == str:
        abs_sum = [abs_sum]    


    #Removes white space since dataset has some problems with repeating lines
    pred_sum = ' '.join(dataset[split][doc_id]['article'].split())
    pred_sum = pred_sum.replace('.', '. ')
    
    
    #Remove sentence duplicates
    pred_sum_sentences = pd.unique(sent_tokenize(pred_sum))
    pred_sum_sentences = [sent for sent in pred_sum_sentences if len(sent.split()) > 5]
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())
    
    if not bart_embed:
        try:
            tfidf_mat = vectorizer.fit_transform(pred_sum_sentences).toarray()
            embeded_sentences = torch.Tensor(tfidf_mat)

        except Exception as e:
            print('------------------ERROR-------------')
            print(e)
            print(pred_sum)
            print(pred_sum_sentences)
            raise e

        try:
            query_tfidf = vectorizer.transform(abs_sum).toarray()
            embeded_query = torch.Tensor(query_tfidf)

        except ValueError as e:
            print(e)
            print('Query error',abs_sum)
            raise e
    else:
        embeded_sentences = torch.Tensor(sentence_transformer.encode(pred_sum_sentences))
        embeded_query = torch.Tensor(sentence_transformer.encode(abs_sum))



    #Create personalization vector that will influence text rank algorithm. 
    #Biases teleportion based on the query
   # personalization_vec = soft_max(personalization_vec)

    old_personalization_vec = (embeded_sentences @ embeded_query.t()).flatten()
    personalization_vec = util.cos_sim(embeded_query, embeded_sentences).flatten()
    #print('DIF', sum((personalization_vec - new_person).flatten()))
    assert(torch.sum(abs(personalization_vec - old_personalization_vec)) < .1)
    
    old_personalization_vec =  personalization_vec / torch.sum(personalization_vec)
    personalization_vec =  personalization_vec / torch.sum(personalization_vec)
    assert(torch.sum(personalization_vec) - 1 <= .001)
    assert(torch.sum(abs(personalization_vec - old_personalization_vec)) < .1)



    # xv, yv = np.meshgrid(np.arange(len(pred_sum_sentences)),np.arange(len(pred_sum_sentences)), indexing='ij')
    # similarity_table = np.zeros(xv.shape)
    # #Calulate cosine similarities between every sentence
    # for i in range(len(xv)):
    #     for j in range(len(yv)):
    #         if xv[i,j] != yv[i,j]:
    #             similarity_table[xv[i,j], yv[i,j]] = tfidf_mat[xv[i,j],:] @ tfidf_mat[yv[i,j],:]

    ind = np.diag_indices(embeded_sentences.shape[0])

    #Remove edges that point to itself
    similarity_table = util.cos_sim(embeded_sentences,embeded_sentences)
    similarity_table[ind[0], ind[1]] = 0
 
    graph = Graph(similarity_table)
    #graph = generate_score(graph, personalization_vec, .1)
    if(make_personalizaton):
        graph = generate_score(graph, personalization_vec,d=alpha)
    else:
        graph = generate_score(graph,d=alpha)

    #Get top 5 ranked sentences and arrange in order they appear in article.

    ranked = np.argsort(graph.get_scores().tolist())[::-1]
    
    final_len = 0
    j = 0
    #print(ranked)
    for rank in ranked:
        if final_len + len(pred_sum_sentences[rank].split()) > max_len:
            if final_len == 0:
                continue
            break
        final_len += len(pred_sum_sentences[rank].split())
        j+=1
    
    
    final_summary = ' '.join([pred_sum_sentences[r] for r in sorted(ranked[:j])])



    #print('max_len', max_len, 'FINAL_SUM', len(final_summary.split()))
    # return final_summary, dataset['validation'][doc_id]['abstract']
    return final_summary
    


#[6, 7, 9, 30, 40]
exclude_idx = [2320, 4923, 5210]


#rouge = load_metric('rouge')
query = "<s> backgroundthe aim of the present study was to test whether coenzyme q10 supplementation could decrease mild - to - moderate statin - associated muscle pain.material/methodsthis was a double - blind, placebo - controlled study with balanced randomization. </s> <s> fifty patients of both sexes, aged between 40 and 65 years, were recruited in this study. before the inclusion to the study, all possible efforts to decrease symptoms and to identify possible association<s> backgroundthe aim of the present study was to test whether coenzyme q10 supplementation could decrease mild - to - moderate statin - associated muscle pain.material/methodsthis was a double - blind, placebo - controlled study with balanced randomization. </s> <s> fifty patients of both sexes, aged between 40 and 65 years, were recruited in this study. before the inclusion to the study, all possible efforts to decrease symptoms and to identify possible association"

# def page_rank_test():
#     weights = np.array([[0, .4, .3], [.4, 0, .8], [.3, .8, 0]])
#     graph = graph(weights)
#     generate_score(graph)


if __name__ == "__main__":
    exclude_idx = [2320, 4923, 5210]
    # print(dataset['validation'][5210])
    # assert(false)
    test2()
    assert(False)
    #main()
    t = 0
 #   for i in range(150):
  #      t+= len(dataset['validation'][i]['article'].split())
#    print('avg', t/150)
    #print(len(dataset['validation'][215]['article'].split()))
#    query_predict(query, 215, 1000)
    pool = multiprocessing.pool(4)
    start_time = time.perf_counter()
    quries = list()
    with open('output2', 'r') as f:
        reader = csv.dictreader(f)
        for row in reader:
            quries.append((row['index'],row['query'].replace("<s>", "").replace("</s>","")))
    
    processes = [pool.apply_async(query_predict, args=(quries[x][1], x, 200,)) for x in range(0,6633) if x not in exclude_idx]
    result = [p.get() for p in processes]
    with open('output3', 'w') as f:
                writer = csv.writer(f)
                header = ['index', 'abstract', 'summary']
                writer.writerow(header) 
                for i in range(len(result)):
                    writer.writerow([result[i][0], result[i][1], result[i][2]])
    finish_time = time.perf_counter()
    print(f"program finished in {finish_time-start_time} seconds")

#[7, 20, 31, 38, 54]
 #a total of 39% ( 215/549 ) patients were diagnosed with vte during their hospital stay , 54% ( 296/549 ) were admitted to hospital with a diagnosis of vte , and 7% ( 38/549 ) were diagnosed and continued to be managed in the outpatient department [ figure 2 ] .co - morbidities in venous thromboembolism patients of the 476 patients with dvt , 2% ( 9 ) had upper extremity dvt , 97% ( 462 ) had lower extremity dvt and the site of dvt was not known in 5 patients .in a study from north india , men constituted 70% of our registry , more than those reported from vellore registry ( 48% ) , but similar to those reported in the endorse ( epidemiologic international day for the evaluation of patients at risk for vte in the acute hospital care setting ) study ( 69% ) .of the 476 patients with dvt , 2% ( 9 ) had upper extremity dvt , 97% ( 462 ) had lower extremity dvt and the site of dvt was not known in 5 patients .of those diagnosed beyond 6 weeks diagnosis of venous thromboembolism during the postoperative period ( n = 81 ) the most common ( 73% ) symptom was swelling of the limb among patients with vte [ table 6 ] pe was confirmed by pulmonary angiography in 27% of all the patients [ table 7 ] .
# ***** eval metrics *****
#   eval_gen_len            =      128.0
#   eval_loss               =      1.408
#   eval_rouge1             =      43.75
#   eval_rouge2             =     21.519
#   eval_rougel             =      28.75
#   eval_rougelsum          =       40.0
#   eval_runtime            = 0:00:00.97
#   eval_samples            =          1
#   eval_samples_per_second =      1.025
#   eval_steps_per_second   =      1.025


