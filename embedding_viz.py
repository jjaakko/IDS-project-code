import numpy as np
import matplotlib.pyplot as plt
import gensim
plt.style.use('ggplot')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


models = ['real','fake']

arr = np.empty((0, 256), dtype='f')

word = "climatechange"
close_words = []
word_labels = []
for label in models:
    try:
        model = gensim.models.KeyedVectors.load(label+ '.model')
        word_labels.append(word+"_"+label)
        close_words.append(model.similar_by_word(word))
    except KeyError:
        pass
# get close words
# using list comprehension
#print(close_words)

#close_words= [j for i in close_words for j in i]
# add the vector for each of the closest words to the array
for label in models:
    try:
        model = gensim.models.KeyedVectors.load(label + '.model')
        arr = np.append(arr, np.array([model[word]]), axis=0)
    except KeyError:
        pass

#print(len(close_words))
checked_words = []
for i in range(len(close_words)):
    print(close_words[i])
    close_words_year = close_words[i]
    model = gensim.models.KeyedVectors.load(models[i] + '.model')
    new_model = gensim.models.KeyedVectors.load('compass.model')
    print(model, models[i])
    for wrd_score in close_words_year:
        try:
            print(wrd_score[0])
            if wrd_score[0] not in checked_words:
                print(wrd_score[0], "is a new word")
                checked_words.append(wrd_score[0])
                print(checked_words)
                wrd_vector = new_model[wrd_score[0]]
                word_labels.append(wrd_score[0])
                print(word_labels)
                arr = np.append(arr, np.array([wrd_vector]), axis=0)
            else:
                print(wrd_score[0], "already in word_labels")
        except KeyError:
            pass

# find tsne coords for 2 dimensions
tsne = TSNE(n_components=2, random_state=0, init='pca')
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(arr)

x_coords = Y[:, 0]
y_coords = Y[:, 1]
# display scatter plot
plt.scatter(x_coords, y_coords)

for label, x, y in zip(word_labels, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=20)
plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
plt.show()

