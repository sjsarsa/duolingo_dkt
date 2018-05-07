# Deep knowledge tracing of Duolingo competition data for deep nlp seminar

The actual data is not included since the training data file is over 100MB. There is a small version of FR_EN dataset for testing in the data_fr_en directory. The whole competition data can be downloaded [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO).

## DKT model
The DKT model is a slightly modified version of Going Deeper with Deep Knowledge Tracing [paper](http://www.educationaldatamining.org/EDM2016/proceedings/paper_133.pdf)'s [code](https://github.com/siyuanzhao/2016-EDM).

The main differences include updating for python3 and TensorFlow 1.6.0, tuned parameters and increased verbosity.
Also the code is in ipython(Jupyter) notebooks.

Usage instructions:
1. Start Jupyter notebook
2. Run ```file_formatter.ipynb``` in the ```data_fr_en``` directory with proper train and test files.
    * Requires lots of memory, the amount of distinct reduced exercises given by the formatter is the main factor here.
3. Run ```tensorflow_dkt.ipynb``` with the correct path for formatted files. 

## LSTM model
Implemented with keras to compare with DKT model
Requires a lot of memory for the whole dataset, should be updated to use sparse data representation.

Usage instructions:
1. Run main.py with proper file paths. These can be passed as argument
```
python3 main.py --train=your_directory/your_file.train --test=your_directory/your_file.dev  --keys=your_directory/your_file.dev.key
```
or edit the defaults in the code,
```
    folder = '../'
    lang = 'fr_en'
```
which translate to ```../data_fr_en/fr_en.slam.20171218.train``` and respective paths.
