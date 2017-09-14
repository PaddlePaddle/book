#!/bin/bash
function abort(){
    echo "The deploy process is failed" 1>&2
    exit 1
}

trap 'abort' 0

openssl enc -in ubuntu.pem.enc -out ubuntu.pem -d -aes256 -k $ENC_PASSWD
chmod 400 ubuntu.pem
ssh -i ubuntu.pem ubuntu@52.76.173.135 '/bin/bash -c echo OK'
trap : 0
