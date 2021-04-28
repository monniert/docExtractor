set -e
gdown --id 1_goCKP5VeStjdDS0nGeZBPqPoLCMNyb6 -O syndoc.zip
unzip syndoc.zip && rm syndoc.zip
mkdir -p datasets
mv syndoc datasets/
