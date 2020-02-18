#!/bin/sh
HOST=localhost
PORT=9200
CMDB_INDEX_NAME=cmkb
CMDB_LIB_DOC_NAME=library
curl -X PUT http://${HOST}:${PORT}/${CMDB_INDEX_NAME}
curl -X PUT "http://${HOST}:${PORT}/${CMDB_INDEX_NAME}/${CMDB_LIB_DOC_NAME}/_mapping" -H 'Content-Type:application/json' -d '{
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "ik_smart",
                "search_analyzer": "ik_smart"
            },
			"tags":{"type":"text", "analyzer": "ik_smart","search_analyzer": "ik_smart"},
			"paragraphs":{"type":"text", "analyzer": "ik_smart","search_analyzer": "ik_smart"}
        }
}'
curl -X GET "localhost:9200/$CMDB_INDEX_NAME?pretty"

