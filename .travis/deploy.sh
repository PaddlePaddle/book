#!/bin/bash
function abort(){
    echo "The deploy process is failed" 1>&2
    exit 1
}

trap 'abort' 0

openssl enc -in ubuntu.pem.enc -out ubuntu.pem -d -aes256 -k $ENC_PASSWD
eval "$(ssh-agent -s)"
chmod 400 ubuntu.pem
ssh-add ubuntu.pem
ssh ubuntu@52.76.173.135 '/bin/bash -c touch /tmp/ok'
trap : 0
