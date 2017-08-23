#########################################################################
# File Name: run.sh
# Method: 
# Author: Jerry Shi
# Mail: jerryshi0110@gmail.com
# Created Time: 2017年08月23日 星期三 11时42分54秒
#########################################################################
#!/bin/bash


# check data and download
if [ ! -d "../../data/classify_toy" ]; then
    wget -c --referer=https://pan.baidu.com/s/1c1FVZe4 -O ../../data/data.zip \
        'http://www.baidupcs.com/rest/2.0/pcs/file?method=batchdownload&app_id=250528&zipcontent=%7B%22fs_id%22%3A%5B%22388358461187362%22%2C%22568140509196122%22%5D%7D&sign=DCb740ccc5511e5e8fedcff06b081203:Dv2cSWSk6eCDyU5fwwtPztmOfPI%3D&uid=2804074387&time=1503546043&dp-logid=5438406453897693993&dp-callid=0&vuk=2804074387&from_uk=2804074387&zipname=%E3%80%90%E6%89%B9%E9%87%8F%E4%B8%8B%E8%BD%BD%E3%80%91data%E7%AD%89.zip'
    cd ../../data/
    unzip data.zip -d data
    mv data/┐╬═т╤з╧░/github_data/classify_toy/ .
    rm -r data/
fi
