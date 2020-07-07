set -e
wget 'http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip' --output-document wikiart.zip
unzip wikiart.zip && rm wikiart.zip
mkdir -p synthetic_resource
mv wikiart synthetic_resource/
