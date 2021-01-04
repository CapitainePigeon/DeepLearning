import pandas as pd
import numpy as np
import unidecode
import re
import string

#import nltk
from torch import tensor
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem.snowball import EnglishStemmer
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
data_path = "/content/drive/MyDrive/DL_Corti/data/"



def tokenize(cell):
  regex_str = [
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',   # numbers
    r"(?:[a-z][a-z\-_]+[a-z])",     # words with -
    r'(?:[\w_]+)',                  # other words
    r'(?:\S)'                       # anything else
  ]
  cell = unidecode.unidecode(cell)
  tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
  return tokens_re.findall(cell.lower())
    
def remove_stop_words(cell):
  stop_words = stopwords.words('english')
  return [word for word in cell if word not in stop_words]

def remove_punctation(cell):
  punctuation = string.punctuation
  return [word for word in cell if word not in punctuation]

def get_stemmed_text(cell):
  stemmer = EnglishStemmer()
  return [stemmer.stem(word) for word in cell]

def clean(col):
  col = col.apply(lambda x: tokenize(x))
  col = col.apply(lambda x: remove_stop_words(x))
  col = col.apply(lambda x: remove_punctation(x))
  col = col.apply(lambda x: get_stemmed_text(x))
  col = col.apply(lambda x: ' '.join(x))
  return col

def cleanAll(df):
  df['text'] = clean(df['text'])
  return df


from gensim.models import KeyedVectors

wordEmbedingModel=KeyedVectors.load_word2vec_format('/content/drive/MyDrive/DL_Corti/code/wiki-news-300d-1M.vec', limit=100000)
#word_vectors = model.wv

def itertuples_impl(df):
  #☺cutoff_date = datetime.date.today() + datetime.timedelta(days=2)  
  return pd.Series(
    vectorise(row)
    for row in df.itertuples()
  )

def vectorise_list(tweetList):
  #☺cutoff_date = datetime.date.today() + datetime.timedelta(days=2)]
    returnList=np.ndarray(shape=(len(tweetList),20,300), dtype=np.float32)
    i=0
    for row in tweetList.itertuples():
        
        returnList[i]=vectorise(row)
        i=i+1
    return tensor(returnList)

empty_array=np.zeros(300)
tweetsize=20
def vectorise(tweet):
    vectorisedTweet=np.ndarray(shape=(20,300), dtype=np.float32)
    words=tweet.text.lower().split()
    i=0
    for word in words:
        if i>=tweetsize:
            break
        if(wordEmbedingModel.__contains__(word)):
            vectorisedTweet[i]=wordEmbedingModel.get_vector(word).astype(np.float32)
            i=i+1
    while i<tweetsize:
        
        vectorisedTweet[i]=empty_array
        i=i+1
    return vectorisedTweet

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return vectorise(self.dataframe.iloc[index])
    

################## to run only once in order to create the txt files ##########################
#################################  first read files from csv ...slow ##########################
#### save text in a txt file
#### delete the file before rerunning, it overwrites!!
##### clean text and save cleaned text in a txt file

# filename = "Political-media-DFE.csv"
# df_politic = pd.read_csv(
#   filename,
#   sep = ',',
#   usecols=["text"],
#   na_filter = False,
#     encoding ="ISO-8859-1")

# filename = "export_dashboard_x_uk.xlsx"
# df_tweet_UK = pd.read_excel(
#   filename,
#   sheet_name="Stream",
#   usecols=[6],
#   na_filter = False)

# filename = "../data/dashboard_x_usa.xlsx"
# df_tweet_USA = pd.read_excel(
#   filename,
#   sheet_name="Stream",
#   usecols=[6],
#   na_filter = False)
#   encoding ="ISO-8859-1")

### save text in a txt file
# df_politic.to_csv('../data/df_politic.txt',  index=None, sep=' ', mode='a')
# df_tweet_UK.to_csv('../data/df_tweet_uk.txt', index=None, sep=' ', mode='a')
# df_tweet_USA.to_csv('../data/df_tweet_usa.txt', index=None, sep=' ', mode='a')

###### clean text and save cleaned text in a txt file
# df_politic_c = cleanAll(df_politic)
# df_tweetUK_c = cleanAll(df_tweet_UK)
# df_tweetUSA_c = cleanAll(df_tweet_USA)

# df_politic_c.to_csv(r'df_politic_c.txt', index = None, sep=' ', mode='a')
# df_tweetUK_c.to_csv(r'df_tweetUK_c.txt', index = None, sep=' ', mode='a')
# df_tweetUSA_c.to_csv(r'../data/df_tweetUSA_c.txt', index = None, sep=' ', mode='a')

cleaner = False

if not cleaner:
    ##Faster reading txt !!!
    df_politic = pd.read_csv(
      data_path+'df_politic.txt')
    
    df_tweetUK = pd.read_csv(
      data_path+'df_tweet_uk.txt', dtype=str, na_filter = False)
    df_tweetUK=df_tweetUK.rename(columns={'Tweet content': 'text'})
    
    df_tweetUSA = pd.read_csv(
      data_path+'df_tweet_usa.txt', dtype=str, na_filter = False)
    df_tweetUSA=df_tweetUSA.rename(columns={'Tweet content': 'text'})
else:
    ##Faster reading txt !!!
    df_politic = pd.read_csv(
      data_path+'df_politic_c.txt', dtype=str, na_filter = False)
    
    df_tweetUK = pd.read_csv(
      data_path+'df_tweetUK_c.txt', dtype=str, na_filter = False)
    df_tweetUK = df_tweetUK.rename(columns={'Tweet content': 'text'})
    
    df_tweetUSA = pd.read_csv(
      data_path+'df_tweetUSA_c.txt', dtype=str, na_filter = False)
    df_tweetUSA = df_tweetUK.rename(columns={'Tweet content': 'text'})

df_politic = df_politic.sample(frac=1).reset_index(drop=True)
df_tweetUK = df_tweetUK.sample(frac=1).reset_index(drop=True)
df_tweetUSA = df_tweetUSA.sample(frac=1).reset_index(drop=True)

batch_size=64
persentageTrain=0.8
#serie_politic= itertuples_impl(df_politic)
dataset_politic = PandasDataset(df_politic)
test_loader_politic = DataLoader(dataset_politic, batch_size=batch_size)

#serie_USA_train= itertuples_impl(df_tweetUSA.iloc[0:int(169033*persentageTrain)])
dataset_USA_train = PandasDataset(df_tweetUSA.iloc[0:int(169033*persentageTrain)])
train_loader = DataLoader(dataset_USA_train, batch_size=batch_size)

#serie_USA_test= itertuples_impl(df_tweetUSA.iloc[int(169033*persentageTrain):])
dataset_USA_test = PandasDataset(df_tweetUSA.iloc[int(169033*persentageTrain):])
test_loader = DataLoader(dataset_USA_test, batch_size=batch_size)

#serie_UK= itertuples_impl(df_tweetUK)
dataset_UK = PandasDataset(df_tweetUK)
test_loader_UK = DataLoader(dataset_UK, batch_size=batch_size)



#cat1= "election"
#cat2= "peace"
#cat3= "technology"

cat1= "love"
cat2= "peace"
cat3= "technology"

matching1 = df_politic[df_politic['text'].str.contains(cat1)]
matching2 = df_politic[df_tweetUK['text'].str.contains(cat2)]
matching3 = df_politic[df_tweetUSA['text'].str.contains(cat3)]
matching1 = matching1[0:10]
matching2 = matching2[0:10]
matching3 = matching3[0:10]
matching = pd.concat([matching1 , matching2 , matching3])
embeding_matching= vectorise_list(matching)
y = [1 for i in range(len(matching1))] + [2 for i in range(len(matching2))] + [3 for i in range(len(matching3))]

#flattening nested list -> list of strings

# print(tfidf_politic)
    


