#!/bin/bash
function abort(){
    echo "The deploy process is failed" 1>&2
    exit 1
}

trap 'abort' 0

directory_name="build"

if [ -d $directory_name ]
then
    rm -rf $directory_name
fi

mkdir $directory_name

cp -r .tools/ $directory_name/.tools

for i in `ls -F | grep /` ; do
    should_copy=false
    cd $i

    if [ -e index.html ] && [ -e index.cn.html ] && [ -d image ]
    then
        should_copy=true
    fi

    cd ..

    if $should_copy ; then
      mkdir $directory_name/$i
      cp $i/index.html $directory_name/$i
      cp $i/index.cn.html $directory_name/$i
      cp -r $i/image $directory_name/$i
    fi

    cp index.html $directory_name/
    cp index.cn.html $directory_name/

done

openssl aes-256-cbc -d -a -in ubuntu.pem.enc -out ubuntu.pem -k $DEC_PASSWD

eval "$(ssh-agent -s)"
chmod 400 ubuntu.pem

ssh-add ubuntu.pem
rsync -r build/ ubuntu@52.76.173.135:/tmp/book

rm -rf $directory_name

chmod 644 ubuntu.pem
rm ubuntu.pem

trap : 0
