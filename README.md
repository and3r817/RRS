# Personalized Recipe Recommender System using Recurrent Neural Network
Empirical experiment for my master thesis "Personalized Recipe Recommender system using RNN".


Code are adapted from the orginal code in "[Collaborative filtering based on sequences](https://github.com/rdevooght/sequence-based-recommendations)" and "[Improving-RNN-recommendation-model](https://github.com/kwonmha/Improving-RNN-recommendation-model)" and implemented in Keras on top of Tensorflow 2.0 framework.
  
<p align="center">
	<img width="1000" height="400" src="https://i.imgur.com/mtdD0ou.png">



## Requirements

- Python 3
- Tensorflow >= 2.0.0
- Keras >=2.1.2
- Numpy
- pickle



## Usage



**Explanation below are almost copied from original github and simplified**

### train.py

This script is used to train models and offers many options regarding when to save new models and when to stop training.
The basic usage is the following:
````
python train.py -d path/to/dataset/ -b 32 --n_epoch 20 --max_length 60 --r_l 100 --r_emb 100 --r_emb_opt lstm
````

The argument `-d` is used to specify the path to the folder that contains the "data", "models" and "results" subfolders created by preprocess.py. 
If you have multiple datasets with a partly common path (e.g. path/to/dataset1/, path/to/dataset2/, etc.) you can specify this common path in the variable DEFAULT_DIR of helpers/data_handling.py. For example, setting DEFAULT_DIR = "path/to/" and using the argument `-d dataset1` will look for the dataset in "path/to/dataset1/".

Other important arguments are the following:

Option | Description
------ | ----------
`-b ` | {int}, batch_size in model.fit(). (default=32)
`--n_epoch ` | {int}, Number of epochs in model.fit(). (default=20)
`--max_length ` | {int}, Maximum length of sub-sequences during training (for RNNs). (default=60)
`--r_t ` | {str}, Type of recurrent layer, choose from [LSTM, GRU, Vanilla]. (default=LSTM)
`--r_l ` | {str}, Size and number of layers. for example, `--r_l 100-50-50` creates a layer with 50 hidden neurons on top of another layer with 50 hidden neurons on top of a layer with 100 hidden neurons. (default=100).
`--r_emb ` | {int}, Output dimension of the embedding layer. (default=100)
`--r_emb_opt ` | {str}, Embedding options, choose from [own, lstm, tf-idf]. (default=own)



`--max_length int` | Maximum length of sequences (default: 30)

### neural_networks/rnn_base.py

This script is used to define non-model-structure related functions, e.g. train(), which defines the training process.

### neural_networks/rnn_core_keras.py

This script is used to define model-structure related functions, e.g. prepare_networks(), which defines the model layers and compile the model before training.


### preprocess.py

This script takes a file containing a dataset of user/item interactions and split it into training/validation/test sets and save them in the format used by train.py and test.py.
The original dataset must be in a format where each line correspond to a single user/item interaction.

The only required argument is `-f path/to/dataset`, which is used to specify the original dataset. The script will create subfolders named "data", "models" and "results" in the folder containing the original dataset. "data" is used by preprocess.py to store all the files it produces, "models" is used by train.py to store the trained models and "results" is used by test.py to store the results of the tests.

The optional arguments are the following:

Option | Desciption
------ | ----------
`--columns` | Order of the columns in the file (eg: "uirt"), u for user, i for item, t for timestamp, r for rating. If r is not present a default rating of 1 is given to all interaction. If t is not present interactions are assumed to be in chronological order. Extra columns are ignored. Default: uit
`--sep` | Separator between the column. If unspecified pandas will try to guess the separator
`--min_user_activity` | Users with less interactions than this will be removed from the dataset. Default: 2
`--min_item_pop` | Items with less interactions than this will be removed from the dataset. Default: 5
`--val_size` | Number of users to put in the validation set. If in (0,1) it will be interpreted as the fraction of total number of users. Default: 0.1
`--test_size` | Number of users to put in the test set. If in (0,1) it will be interpreted as the fraction of total number of users. Default: 0.1
`--seed` | Seed for the random train/val/test split

#### Example
In the movielens 1M dataset each line has the following format:
````
UserID::MovieID::Rating::Timestamp
````
To process it you have to specify the order of the columns, in this case uirt (for user, item, rating, timestamp), and the separator ("::"). If you want to use a hundred users for the validation set and a hundred others for the test set, you'll have to use the following command:
````
python preprocess.py -f path/to/datafile(.dat or .csv, etc) --columns uirt --sep :: --val_size 100 --test_size 100
````

