#!/bin/bash
# from https://gist.githubusercontent.com/nehalecky/9258c01fb2077f51545a/raw/789f08141dc681cf1ad5da05455c2cd01d1649e8/install-py3-dataproc.sh

apt-get -y install python3
apt-get -y install python3-pip
pip3 install -y tweepy=3.10.0
pip3 install google
pip3 install google-cloud-language
pip3 install reload
pip3 install nltk
echo "export PYSPARK_PYTHON=python3" | tee -a  /etc/profile.d/spark_config.sh  /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "Adding PYTHONHASHSEED=0 to profiles and spark-defaults.conf..."
echo "export PYTHONHASHSEED=0" | tee -a /etc/profile.d/spark_config.sh /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "spark.executorEnv.PYTHONHASHSEED=0" >> /etc/spark/conf/spark-defaults.conf