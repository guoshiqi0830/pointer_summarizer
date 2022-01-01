git clone https://github.com/andersjo/pyrouge.git
pip install pyrouge
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
apt-get install libxml-parser-perl
cd pyrouge/tools/ROUGE-1.5.5/data/ && \
rm WordNet-2.0.exc.db && \
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db