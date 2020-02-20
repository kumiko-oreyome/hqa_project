#!/bin/sh
sudo sysctl -w vm.max_map_count=262144
sudo docker pull samejack/docker-elasticsearch-kibana:latest
sudo docker run -p 5601:5601 -p 9200:9200 -it samejack/elasticsearch-kibana:latest
