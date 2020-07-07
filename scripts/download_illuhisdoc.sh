set -e
wget 'https://www.dropbox.com/s/bbpb9lzanjtj9f9/illuhisdoc.zip?dl=0' --output-document illuhisdoc.zip
unzip illuhisdoc.zip && rm illuhisdoc.zip
mkdir -p raw_data
mv illuhisdoc raw_data/
