#!/bin/sh
ENV_NAME=env

#prepare python venv environment
if [ ! -d "$ENV_NAME" ];then
	echo "create venv $ENV_NAME"
	python -m venv $ENV_NAME
fi
source $ENV_NAME/bin/activate
pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt


#prepare elastic search

