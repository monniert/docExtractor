# docExtractor

Pytorch implementation of "docExtractor: An off-the-shelf historical document element 
extraction" paper:

[Paper](http://imagine.enpc.fr/~monniert/docExtractor/docExtractor.pdf) | [Project
webpage](http://imagine.enpc.fr/~monniert/docExtractor)

![teaser.jpg](http://imagine.enpc.fr/~monniert/docExtractor/teaser.jpg)

## Installation :construction_worker:

### Prerequisites

Make sure you have Anaconda installed (version >= to 4.7.10, you may not be able to install 
correct dependencies if older). If not, follow the installation instructions provided at 
https://docs.anaconda.com/anaconda/install/.

### 1. Create conda environment

```
conda env create -f environment.yml
conda activate docExtractor
```

### 2. Download resources and models

The following command will download:

- synthetic resources needed to generate SynDoc (manually collected ones and Wikiart dataset)
- our trained model
- IlluHisDoc dataset

```
./download.sh
```

## How to use :rocket:

There are several main usages you may be interested in:

1. perform element extraction (off-the-shelf using our trained network or a fine-tuned one)
2. build our segmentation method from scratch
3. fine-tune network on custom datasets

### 1. Element extraction

```
CUDA_VISIBLE_DEVICES=gpu_id python src/extractor.py --input_dir inp --output_dir out
```

**Main args**
- `-i, --input_dir`: directory where images to process are stored (e.g. `raw_data/test`)
- `-o, --output_dir`: directory where extracted elements will be saved (e.g. 
  `results/output`)
- `-t, --tag`: model tag to use for extraction (default is our trained network)
- `-l, --labels`: labels to extract (default corresponds to illustration and text labels)
- `-s, --save_annot`: whether to save full resolution annotations while extracting

**Additionals**
- `-sb, --straight_bbox`: whether to use straight bounding boxes instead of rotated ones to 
  fit connected components
- `-dm, --draw_margin`: Draw the margins added during extraction (for visual or debugging 
  purposes)

**NB:** check `src/utils/constant.py` for labels mapping  
**NB:** the code will automatically run on CPU if no GPU are provided/found

### 2. Build our segmentation method from scratch

This would result in a model similar to the one located in `models/default/model.pkl`.

#### a) Generate SynDoc dataset

```
python src/syndoc_generator.py -d dataset_name -n nb_train --merged_labels
```

- `-d, --dataset_name`: name of the resulting synthetic dataset
- `-n, --nb_train`: nb of training samples to generate (0.1 x nb_train samples are generated 
  for val and test splits)
- `-m, --merged_labels`: whether to merge all graphical and textual labels into unique 
  `illustration` and `text` labels 

#### b) Train neural network on SynDoc

```
CUDA_VISIBLE_DEVICES=gpu_id python src/trainer.py --tag tag --config syndoc.yml
```

### 3. Fine-tune network on custom datasets

To fine-tune on a custom dataset, you have to:

1. annotate a dataset pixel-wise: we recommend using VGG Image Anotator 
   ([link](http://www.robots.ox.ac.uk/~vgg/software/via/)) and our `ViaJson2Image` tool
2. split dataset in `train`, `val`, `test` and move it to `datasets` folder
3. create a `configs/example.yml` config with corresponding dataset name and a pretrained 
   model name (e.g. `default`)
4. train segmentation network with `python src/trainer.py --tag tag --config example.yml`

Then you can perform extraction with the fine-tuned network by specifying the model tag.

**NB**: `val` and `test` splits can be empty, you just won't have evaluation metrics

## Annex tools :hammer_and_wrench:

### 1. ViaJson2Image - Convert VIA annotations into segmentation map images

```
python src/via_converter.py --input_dir inp --output_dir out --file via_region_data.json
```

**NB**: by default, it converts regions using the `illustration` label

### 2. SyntheticLineGenerator - Generate a synthetic line dataset for OCR training

```
python src/synline_generator.py -d dataset_name -n nb_doc
```

**NB:** you may want to remove text translation augmentations by modifying 
`synthetic.element.TEXT_FONT_TYPE_RATIO`.

### 3. IIIFDownloader - Download IIIF images data from manifest urls

```
python src/iiif_downloader.py -f filename.txt -o output_dir --width W --height H
```

- `-f, --file`: file where each line contains an url to a json manifest
- `-o, --output_dir`: directory where downloaded images will be saved
- `--width` and `--height`: image width and height respectively (default is full resolution)

**NB:** Be aware that if both `width` and `height` arguments are specified, aspect ratio 
won't be kept. Common usage is to specify a fixed height only.

## How to cite? :clipboard:

If you find this work useful in your research, please consider citing:

```
@inproceedings{monnier2020docExtractor,
  title={docExtractor: An off-the-shelf historical document element extraction},
  author={Monnier, Tom and Aubry, Mathieu},
  booktitle={ICFHR},
  year={2020},
}
```
