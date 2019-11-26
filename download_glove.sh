if [[ ! -d glove ]]; then
	mkdir glove
fi

cd glove
#wget http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
wget http://nlp.stanford.edu/data/glove.6B.zip
#unzip glove.42B.300d.zip
unzip glove.6B.zip
