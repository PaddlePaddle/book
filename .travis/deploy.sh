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
#cp ../01.fit_a_line/ $directory_name
for i in `ls -F | grep /`
do
echo $i
should_copy=false
cd $i
if [ -e index.html ] && [ -e index.cn.html ] && [ -d image ]
then
    echo "ok"
    should_copy=true
else
    echo "nok"
fi
cd ..

if $should_copy ; then
  echo 'should_copy'
  mkdir $directory_name/$i
  cp $i/index.html $directory_name/$i
  cp $i/index.cn.html $directory_name/$i
  cp -r $i/image $directory_name/$i
fi

cp index.html $directory_name/
cp index.cn.html $directory_name/

done

#openssl enc -in ubuntu.pem.enc -out ubuntu.pem -d -aes256 -k $DEC_PASSWD
##openssl enc -in ubuntu.pem.enc -out ubuntu.pem -d -aes256 -k 'SOME_PASSWD'
#eval "$(ssh-agent -s)"
#chmod 400 ubuntu.pem
#ssh-add ubuntu.pem

#rsync -r --delete-after --quiet ../01.fit_a_line ubuntu@52.76.173.135


#ssh ubuntu@52.76.173.135 '/usr/bin/touch /tmp/ok'
trap : 0
