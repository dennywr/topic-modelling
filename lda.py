import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup

join = []
temp = []

with st.sidebar:
  selected = option_menu('Topic Modelling', ['Crawling Data', 'Load Data', 'Preprocessing', 'Ekstraksi Fitur', 'LDA', 'Clustering & Klasifikasi'], default_index=0)
st.title("Topic Modeling")
##### Crwaling Data
def crawlingPta():
  st.subheader("Crawling PTA UTM")
  url = st.text_input('Inputkan url pta utm berdasarkan prodi di sini', 'https://pta.trunojoyo.ac.id/c_search/byprod/10')

  headers = {
      'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Mobile Safari/537.36'
  }

  req = requests.get(url, headers=headers)

  bs = BeautifulSoup(req.content, "html5lib")
  bs.prettify()
  item = bs.find("div", {"id": "wrapper"})
  pagination = item.findAll("a", attrs = {"class": "pag_button"})
  reqPagination = st.number_input('Masukkan jumlah halaman yang ingin di crawling (max: 172): ', min_value=1, max_value=172, value=1, step=1)
  totalPages = int(pagination[reqPagination]["href"].split("/")[-1])
  goCrawlingButton = st.button('Mulai Crawling')
  if(goCrawlingButton):
    global temp
    for pages in range(1, totalPages+1):

        url = "https://pta.trunojoyo.ac.id/c_search/byprod/10"
        idProdi = url.split("/")[-1]
        nameOfProdi = ""
        if (idProdi == "10"):
            nameOfProdi = "Teknik Informatika"

        reqPages = requests.get(f"{url}/{pages}", headers=headers)
        bsPages = BeautifulSoup(reqPages.content, "html5lib")
        bsPages.prettify()

        items = bsPages.find("div", {"id": "wrapper"})
        for item in items.findAll("a", attrs = {"class":"gray button"}):
            reqItem = requests.get(item["href"], headers=headers)
            bsItem = BeautifulSoup(reqItem.content, "html5lib")
            bsItem.prettify()
            getData = bsItem.find("li", attrs = {"data-id":"id-1", "data-cat":"#luxury"})
            title = getData.find('a', attrs = {"class":"title", "href":"#"}).string
            writer = getData.find_all('div', style='padding:2px 2px 2px 2px;')[0].span.get_text().replace("Penulis : ", "")
            dp1 = getData.find_all('div', style='padding:2px 2px 2px 2px;')[1].span.get_text().replace("Dosen Pembimbing I : ", "")
            dp2 = getData.find_all('div', style='padding:2px 2px 2px 2px;')[2].span.get_text().replace("Dosen Pembimbing II :", "")
            abstract = getData.find('p', attrs = {"align":"justify"}).string
            temp.append([title, writer, dp1, dp2, abstract])
    st.dataframe(temp)

##### Load Data
def loadData():
  st.subheader("Load Data:")
  data_url = st.text_input('Enter URL of your CSV file here', 'https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/ptautm-fix.csv')

  @st.cache_resource
  def load_data():
      data = pd.read_csv(data_url, index_col=False)
      # data['nomor\ufeff'] += 1
      return data

  df = load_data()
  # df.set_index('nomor\ufeff', inplace=True)
  # df.index += 1
  df['Abstrak'] = df['Abstrak'].fillna('').astype(str)
  # if(selected == 'Load Data'):
  st.dataframe(df)
  return (df['Judul'])
    

##### Preprocessing
def preprocessing():
  st.subheader("Preprocessing:")
  st.text("Menghapus karakter spesial")

  ### hapus karakter spesial
  @st.cache_resource
  def load_data():
      data = pd.read_csv('https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/ptautm-fix.csv', index_col=False)
      # data['nomor\ufeff'] += 1
      return data

  df = load_data()
  def removeSpecialText (text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")
  
  df['Abstrak'] = df['Abstrak'].astype(str).apply(removeSpecialText)
  df['Abstrak'] = df['Abstrak'].apply(removeSpecialText)
  # df.index += 1
  st.dataframe(df['Abstrak'])

  ### hapus tanda baca
  st.text("Menghapus tanda baca")
  def removePunctuation(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text

  df['Abstrak'] = df['Abstrak'].apply(removePunctuation)
  st.dataframe(df['Abstrak'])

  ### hapus angka pada teks
  st.text("Menghapus angka pada teks")
  def removeNumbers (text):
    return re.sub(r"\d+", "", text)
  df['Abstrak'] = df['Abstrak'].apply(removeNumbers)
  st.dataframe(df['Abstrak'])

  ### case folding
  st.text("Mengubah semua huruf pada teks menjadi huruf kecil")
  def casefolding(Comment):
    Comment = Comment.lower()
    return Comment
  df['Abstrak'] = df['Abstrak'].apply(casefolding)
  st.dataframe(df['Abstrak'])

  ### Tokenizing dan stopwords removal
  st.text("Tokenisasi dan penghapusan stopwords")
  more_stopword = ["ada", "adanya", "adalah", "k", "cf", "z", "zf", "adapun", "agak", "agaknya", "agar", "akan", "akankah", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar", "diantaranya", "antara", "antaranya", "diantara", "apa", "apaan", "mengapa", "apabila", "apakah", "apalagi", "apatah", "atau", "ataukah", "ataupun", "bagai", "bagaikan", "sebagai", "sebagainya", "bagaimana", "bagaimanapun", "sebagaimana", "bagaimanakah", "bagi", "bahkan", "bahwa", "bahwasanya", "sebaliknya", "banyak", "sebanyak", "beberapa", "seberapa", "begini", "beginian", "beginikah", "beginilah", "sebegini", "begitu", "begitukah", "begitulah", "begitupun", "sebegitu", "belum", "belumlah", "sebelum", "sebelumnya", "sebenarnya", "berapa", "berapakah", "berapalah", "berapapun", "betulkah", "sebetulnya", "biasa", "biasanya", "bila", "bilakah", "bisa", "bisakah", "sebisanya", "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya", "cuma", "percuma", "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "dekat", "demi", "demikian", "demikianlah", "sedemikian", "dengan", "depan", "di", "dia", "dialah", "dini", "diri", "dirinya", "terdiri", "dong", "dulu", "enggak", "enggaknya", "entah", "entahlah", "terhadap", "terhadapnya", "hal", "hampir", "hanya", "hanyalah", "harus", "haruslah", "harusnya", "seharusnya", "hendak", "hendaklah", "hendaknya", "hingga", "sehingga", "ia", "ialah", "ibarat", "ingin", "inginkah", "inginkan", "ini", "inikah", "inilah", "itu", "itukah", "itulah", "jangan", "jangankan", "janganlah", "jika", "jikalau", "juga", "justru", "kala", "kalau", "kalaulah", "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah", "kapanpun", "dikarenakan", "karena", "karenanya", "ke", "kecil", "kemudian", "kenapa", "kepada", "kepadanya", "ketika", "seketika", "khususnya", "kini", "kinilah", "kiranya", "sekiranya", "kita", "kitalah", "kok", "lagi", "lagian", "selagi", "lah", "lain", "lainnya", "melainkan", "selaku", "lalu", "melalui", "terlalu", "lama", "lamanya", "selama", "selama", "selamanya", "lebih", "terlebih", "bermacam", "macam", "semacam", "maka", "makanya", "makin", "malah", "malahan", "mampu", "mampukah", "mana", "manakala", "manalagi", "masih", "masihkah", "semasih", "masing", "mau", "maupun", "semaunya", "memang", "mereka", "merekalah", "meski", "meskipun", "semula", "mungkin", "mungkinkah", "nah", "namun", "nanti", "nantinya", "nyaris", "oleh", "olehnya", "seorang", "seseorang", "pada", "padanya", "padahal", "paling", "sepanjang", "pantas", "sepantasnya", "sepantasnyalah", "para", "pasti", "pastilah", "per", "pernah", "pula", "pun", "merupakan", "rupanya", "serupa", "saat", "saatnya", "sesaat", "saja", "sajalah", "saling", "bersama", "sama", "sesama", "sambil", "sampai", "sana", "sangat", "sangatlah", "saya", "sayalah", "se", "sebab", "sebabnya", "sebuah", "tersebut", "tersebutlah", "sedang", "sedangkan", "sedikit", "sedikitnya", "segala", "segalanya", "segera", "sesegera", "sejak", "sejenak", "sekali", "sekalian", "sekalipun", "sesekali", "sekaligus", "sekarang", "sekarang", "sekitar", "sekitarnya", "sela", "selain", "selalu", "seluruh", "seluruhnya", "semakin", "sementara", "sempat", "semua", "semuanya", "sendiri", "sendirinya", "seolah", "seperti", "sepertinya", "sering", "seringnya", "serta", "siapa", "siapakah", "siapapun", "disini", "disinilah", "sini", "sinilah", "sesuatu", "sesuatunya", "suatu", "sesudah", "sesudahnya", "sudah", "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tak", "tanpa", "setelah", "telah", "tentang", "tentu", "tentulah", "tentunya", "tertentu"]

  from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
  import nltk
  nltk.download('punkt')
  from nltk.tokenize import word_tokenize
  #Inisialisasi fungsi stop words
  stop_factory = StopWordRemoverFactory()
  words = []
  #Membuat perulangan untuk memasukkan dataset ke dalam tekonisasi dan list stopwords
  for i in range (len(df['Abstrak'])):
    #Inisialisai fungsi tokenisasi dan stopword
    tokens = word_tokenize(df['Abstrak'][i])
    data = stop_factory.get_stop_words()+more_stopword
    stopword = stop_factory.create_stop_word_remover()
    # Melakukan removed kata
    removed = []
    for t in tokens:
        if t not in data:
            removed.append(t)
    # Memasukkan hasil removed kedalem variable words
    words.append(removed)
    # st.write(removed)  
    dfRemoved = pd.DataFrame(removed, columns=['Tokenisasi dan Stopwords']).T
  # Display the DataFrame
  st.dataframe(dfRemoved.head(5))

  ### menggabungkan kata hasil tokenisasi
  st.text("Menggabungkan kata hasil tokenisasi")
  join = []
  for i in range(len(words)):
    join_words = ' '.join(words[i])
    join.append(join_words)

  st.dataframe({"Join Words":join})
  return df["Abstrak"]


def preprocessingTanpaOutput():

  ### hapus karakter spesial
  @st.cache_resource
  def load_data():
      data = pd.read_csv('https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/ptautm-fix.csv')
      return data

  df = load_data()
  ### if(hapusKarakterSpesial):
  def removeSpecialText (text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")
  
  df['Abstrak'] = df['Abstrak'].astype(str).apply(removeSpecialText)
  df['Abstrak'] = df['Abstrak'].apply(removeSpecialText)


  # hapusTandaBaca = st.button("Hapus Tanda Baca")
  def removePunctuation(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text

  df['Abstrak'] = df['Abstrak'].apply(removePunctuation)

  ### hapus angka pada teks
  def removeNumbers (text):
    return re.sub(r"\d+", "", text)
  df['Abstrak'] = df['Abstrak'].apply(removeNumbers)

  ### case folding
  def casefolding(Comment):
    Comment = Comment.lower()
    return Comment
  df['Abstrak'] = df['Abstrak'].apply(casefolding)

  ### Tokenizing dan stopwords removal
  more_stopword = ["ada", "adanya", "adalah", "k", "cf", "z", "zf", "adapun", "agak", "agaknya", "agar", "akan", "akankah", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar", "diantaranya", "antara", "antaranya", "diantara", "apa", "apaan", "mengapa", "apabila", "apakah", "apalagi", "apatah", "atau", "ataukah", "ataupun", "bagai", "bagaikan", "sebagai", "sebagainya", "bagaimana", "bagaimanapun", "sebagaimana", "bagaimanakah", "bagi", "bahkan", "bahwa", "bahwasanya", "sebaliknya", "banyak", "sebanyak", "beberapa", "seberapa", "begini", "beginian", "beginikah", "beginilah", "sebegini", "begitu", "begitukah", "begitulah", "begitupun", "sebegitu", "belum", "belumlah", "sebelum", "sebelumnya", "sebenarnya", "berapa", "berapakah", "berapalah", "berapapun", "betulkah", "sebetulnya", "biasa", "biasanya", "bila", "bilakah", "bisa", "bisakah", "sebisanya", "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya", "cuma", "percuma", "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "dekat", "demi", "demikian", "demikianlah", "sedemikian", "dengan", "depan", "di", "dia", "dialah", "dini", "diri", "dirinya", "terdiri", "dong", "dulu", "enggak", "enggaknya", "entah", "entahlah", "terhadap", "terhadapnya", "hal", "hampir", "hanya", "hanyalah", "harus", "haruslah", "harusnya", "seharusnya", "hendak", "hendaklah", "hendaknya", "hingga", "sehingga", "ia", "ialah", "ibarat", "ingin", "inginkah", "inginkan", "ini", "inikah", "inilah", "itu", "itukah", "itulah", "jangan", "jangankan", "janganlah", "jika", "jikalau", "juga", "justru", "kala", "kalau", "kalaulah", "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah", "kapanpun", "dikarenakan", "karena", "karenanya", "ke", "kecil", "kemudian", "kenapa", "kepada", "kepadanya", "ketika", "seketika", "khususnya", "kini", "kinilah", "kiranya", "sekiranya", "kita", "kitalah", "kok", "lagi", "lagian", "selagi", "lah", "lain", "lainnya", "melainkan", "selaku", "lalu", "melalui", "terlalu", "lama", "lamanya", "selama", "selama", "selamanya", "lebih", "terlebih", "bermacam", "macam", "semacam", "maka", "makanya", "makin", "malah", "malahan", "mampu", "mampukah", "mana", "manakala", "manalagi", "masih", "masihkah", "semasih", "masing", "mau", "maupun", "semaunya", "memang", "mereka", "merekalah", "meski", "meskipun", "semula", "mungkin", "mungkinkah", "nah", "namun", "nanti", "nantinya", "nyaris", "oleh", "olehnya", "seorang", "seseorang", "pada", "padanya", "padahal", "paling", "sepanjang", "pantas", "sepantasnya", "sepantasnyalah", "para", "pasti", "pastilah", "per", "pernah", "pula", "pun", "merupakan", "rupanya", "serupa", "saat", "saatnya", "sesaat", "saja", "sajalah", "saling", "bersama", "sama", "sesama", "sambil", "sampai", "sana", "sangat", "sangatlah", "saya", "sayalah", "se", "sebab", "sebabnya", "sebuah", "tersebut", "tersebutlah", "sedang", "sedangkan", "sedikit", "sedikitnya", "segala", "segalanya", "segera", "sesegera", "sejak", "sejenak", "sekali", "sekalian", "sekalipun", "sesekali", "sekaligus", "sekarang", "sekarang", "sekitar", "sekitarnya", "sela", "selain", "selalu", "seluruh", "seluruhnya", "semakin", "sementara", "sempat", "semua", "semuanya", "sendiri", "sendirinya", "seolah", "seperti", "sepertinya", "sering", "seringnya", "serta", "siapa", "siapakah", "siapapun", "disini", "disinilah", "sini", "sinilah", "sesuatu", "sesuatunya", "suatu", "sesudah", "sesudahnya", "sudah", "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tak", "tanpa", "setelah", "telah", "tentang", "tentu", "tentulah", "tentunya", "tertentu"]

  from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
  import nltk
  nltk.download('punkt')
  from nltk.tokenize import word_tokenize
  #Inisialisasi fungsi stop words
  stop_factory = StopWordRemoverFactory()
  words = []
  #Membuat perulangan untuk memasukkan dataset ke dalam tekonisasi dan list stopwords
  for i in range (len(df['Abstrak'])):
    #Inisialisai fungsi tokenisasi dan stopword
    tokens = word_tokenize(df['Abstrak'][i])
    data = stop_factory.get_stop_words()+more_stopword
    stopword = stop_factory.create_stop_word_remover()
    # Melakukan removed kata
    removed = []
    for t in tokens:
        if t not in data:
            removed.append(t)
    # Memasukkan hasil removed kedalem variable words
    words.append(removed)

  ### menggabungkan kata hasil tokenisasi
  join = []
  for i in range(len(words)):
    join_words = ' '.join(words[i])
    join.append(join_words)
  return (df["Abstrak"], df["Judul"])


##### Ekstraksi Fitur
def ekstraksiFitur():
  import nltk
  from nltk.tokenize import RegexpTokenizer
  from sklearn.decomposition import TruncatedSVD
  from sklearn.feature_extraction.text import TfidfVectorizer
  from nltk.corpus import stopwords

  nltk.download('stopwords', quiet=True)

  st.subheader("Ekstraksi Fitur (TF-IDF):")
  stopwords = stopwords.words('indonesian')

  from sklearn.feature_extraction.text import CountVectorizer

  coun_vect = CountVectorizer(stop_words=stopwords)
  count_matrix = coun_vect.fit_transform(preprocessingTanpaOutput()[0])
  count_array = count_matrix.toarray()

  df = pd.DataFrame(data=count_array, columns=coun_vect.vocabulary_.keys())

  # Menampilkan DataFrame menggunakan streamlit
  st.text(ekstraksiFiturTanpaOutput()[0].shape)

  df = pd.concat([preprocessingTanpaOutput()[1], df], axis=1)

  st.dataframe(df)

  tokenizer = RegexpTokenizer(r'\w+')
  vectorizer = TfidfVectorizer(lowercase=True,
                          stop_words=stopwords,
                          tokenizer = tokenizer.tokenize)

  tfidf_matrix = vectorizer.fit_transform(preprocessingTanpaOutput()[0])
  tfidf_terms = vectorizer.get_feature_names_out()
  st.text(tfidf_matrix)
  


##### Ekstraksi Fitur
def ekstraksiFiturTanpaOutput():
  import nltk
  from nltk.tokenize import RegexpTokenizer
  from sklearn.decomposition import TruncatedSVD
  from sklearn.feature_extraction.text import TfidfVectorizer
  from nltk.corpus import stopwords

  nltk.download('stopwords', quiet=True)

  stopwords = stopwords.words('indonesian')

  tokenizer = RegexpTokenizer(r'\w+')
  vectorizer = TfidfVectorizer(lowercase=True,
                          stop_words=stopwords,
                          tokenizer = tokenizer.tokenize)

  tfidf_matrix = vectorizer.fit_transform(preprocessingTanpaOutput()[0])
  tfidf_terms = vectorizer.get_feature_names_out()
  return [tfidf_matrix, tfidf_terms]
  
def lda():
  coba1 = ekstraksiFiturTanpaOutput()[0]
  coba2 = ekstraksiFiturTanpaOutput()[1]
  coba3 = coba1
  coba4 = coba2
  import numpy as np
  from sklearn.decomposition import LatentDirichletAllocation
  numberLda = st.number_input('Masukkan jumlah topik: ', min_value=1, max_value=1000, value=5, step=1)
  wordLda = st.number_input('Masukkan jumlah kata tiap topik: ', min_value=1, max_value=1000, value=5, step=1)
  # jumlah topik
  n_topics = int(numberLda)

  # jumlah kata per topik
  n_words = int(wordLda)

# def lda():
  proporsiKataDalamTopik = st.button('Tampilkan proporsi kata dalam topik')
  if(proporsiKataDalamTopik):
    ###### baru
    # jumlah topik
    # n_topics = n_topics

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(coba3)

    # membuat dataframe untuk menampilkan proporsi
    topics_df = pd.DataFrame(lda.components_, columns=coba4)

    # menambahkan nama topik sebagai indeks
    topics_df.index = ['Topik {}'.format(i) for i in range(n_topics)]
    st.subheader("Proporsi seluruh kata dalam topik")
    # menampilkan dataframe
    st.dataframe(topics_df)
    st.text(topics_df.shape)
    # topics_df
    ######
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(coba3)

    # menampilkan topik
    st.subheader("Proporsi kata dalam topik")
    for idx, topic in enumerate(lda.components_):
        st.write("Topic %d:" % (idx))
        st.write({(coba4[i], topic[i]) for i in topic.argsort()[:-n_words - 1:-1]})

    #####
    st.subheader("Proporsi topik dalam dokumen")
    # mendapatkan proporsi topik untuk setiap dokumen
    doc_topic_distrib = lda.transform(coba3)
    topic_word_distrib = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    # dataframe
    doc_topic_df = pd.DataFrame(doc_topic_distrib, columns=[f'Topic_{i}' for i in range(n_topics)])


    doc_topic_df[[f'Topic_{i}' for i in range(n_topics)]].apply(lambda x: x.argmax(), axis=1)

    doc_topic_df = pd.concat([preprocessingTanpaOutput()[1], doc_topic_df], axis=1)

    doc_topic_df

    st.text(doc_topic_df.shape)

    ekstraksiFitur()

def clusteringDanKlasifikasi():
  from sklearn.cluster import KMeans

  tfidf_matrix = ekstraksiFiturTanpaOutput()[0]
  st.subheader('Clustering (K-Means)')
  numOfCluster = st.number_input('Masukkan jumlah cluster: ', min_value=1, max_value=100, value=2, step=1)
  # numOfCluster = st.slider("Pick a number", 0, 100)

  resultOfCluster = st.button('Tampilkan hasil clustering')
  if(resultOfCluster):
    # Menjalankan algoritma k-means
    kmeans = KMeans(n_clusters=numOfCluster, n_init=10, random_state=0)
    kmeans.fit(tfidf_matrix)

    # Menampilkan hasil clustering
    abstrak = []
    cluster = []
    for i, doc in enumerate(preprocessingTanpaOutput()[0]):
      abstrak.append(doc)
      cluster.append(kmeans.labels_[i])

    data = {'Abstrak': abstrak, 'Cluster': cluster}
    dfCluster = pd.DataFrame(data)
    st.dataframe(dfCluster)

    from sklearn.metrics import silhouette_score

    # Prediksi cluster untuk setiap sampel
    preds = kmeans.predict(tfidf_matrix)

    # Hitung skor silhouette
    st.subheader('Evaluasi (Silhouette)')
    score = silhouette_score(tfidf_matrix, preds)
    st.text("Silhouette score: {}".format(score))

    ### klasifikasi
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report

    # Membagi data menjadi data training dan data testing
    X_train, X_test, y_train, y_test = train_test_split(dfCluster['Abstrak'], dfCluster['Cluster'], test_size=0.2, random_state=42)

    # Mengubah teks menjadi vektor fitur menggunakan CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Membuat dan melatih model Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)

    # Melakukan prediksi pada data testing
    y_pred = nb.predict(X_test_vec)

    # Menampilkan laporan klasifikasi
    # print(classification_report(y_test, y_pred))
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.subheader('Klasifikasi (Naive Bayes)')
      # assuming y_test and y_pred are defined
    report = classification_report(y_test, y_pred)

    st.text(report)
    st.text("")

    st.subheader('Evaluasi (Confusion Matrix)')

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Hitung confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualisasikan confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    # Tampilkan plot di Streamlit
    st.pyplot(plt)
    # return numOfCluster
  
# def clusteringTanpaOutput():
#   from sklearn.cluster import KMeans

#   tfidf_matrix = ekstraksiFiturTanpaOutput()[0]
#   # Menjalankan algoritma k-means
#   kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
#   kmeans.fit(tfidf_matrix)

#   # Menampilkan hasil clustering
#   abstrak = []
#   cluster = []
#   for i, doc in enumerate(preprocessingTanpaOutput()[0]):
#     abstrak.append(doc)
#     cluster.append(kmeans.labels_[i])

#   data = {'Abstrak': abstrak, 'Cluster': cluster}
#   dfCluster = pd.DataFrame(data)
#   return dfCluster

# def klasifikasi():
#   from sklearn.model_selection import train_test_split
#   from sklearn.feature_extraction.text import CountVectorizer
#   from sklearn.naive_bayes import MultinomialNB
#   from sklearn.metrics import classification_report

#   # Membagi data menjadi data training dan data testing
#   X_train, X_test, y_train, y_test = train_test_split(clusteringTanpaOutput()['Abstrak'], clusteringTanpaOutput()['Cluster'], test_size=0.2, random_state=42)

#   # Mengubah teks menjadi vektor fitur menggunakan CountVectorizer
#   vectorizer = CountVectorizer()
#   X_train_vec = vectorizer.fit_transform(X_train)
#   X_test_vec = vectorizer.transform(X_test)

#   # Membuat dan melatih model Naive Bayes
#   nb = MultinomialNB()
#   nb.fit(X_train_vec, y_train)

#   # Melakukan prediksi pada data testing
#   y_pred = nb.predict(X_test_vec)

#   # Menampilkan laporan klasifikasi
#   # print(classification_report(y_test, y_pred))
#   st.subheader('Klasifikasi')
#     # assuming y_test and y_pred are defined
#   report = classification_report(y_test, y_pred)

#   st.text(report)

#   st.subheader('Evaluasi (Confusion Matrix)')

#   from sklearn.metrics import confusion_matrix
#   import seaborn as sns
#   import matplotlib.pyplot as plt

#   # Hitung confusion matrix
#   cm = confusion_matrix(y_test, y_pred)

#   # Visualisasikan confusion matrix
#   plt.figure(figsize=(10,7))
#   sns.heatmap(cm, annot=True, fmt='d')
#   plt.xlabel('Predicted')
#   plt.ylabel('Truth')

#   # Tampilkan plot di Streamlit
#   st.pyplot(plt)



   
   

def main():
  if(selected == 'Load Data'):
     loadData()
  if(selected == 'Preprocessing'):
     preprocessing()

  if(selected == 'Ekstraksi Fitur'):
    #  preprocessingOutputHidden()
     ekstraksiFitur()

  if(selected == 'LDA'):
     lda()
  if(selected == 'Crawling Data'):
     crawlingPta()
  if(selected == 'Clustering & Klasifikasi'):
     clusteringDanKlasifikasi()
  # if(selected == 'Klasifikasi'):
  #    klasifikasi()
    # st.text(tfidf_matrix)




if __name__ == "__main__":
    main()

