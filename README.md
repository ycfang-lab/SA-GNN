```
.
├── configs
├── dataset
├── evaluation
├── framework.png
├── __init__.py
├── LICENSE
├── main.py
├── modules
├── old_README.md
├── preprocess
├── README.md
├── seq_scripts.py
├── slr_network.py
├── software
├── utils
└── work_dir
```

### Data Preparation

1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it to ./dataset/phoenix, it is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phienix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess.py --process-image --multiprocessing
   ```

### Prerequisites

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite:    
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`
  We also provide a python version evaluation tool for convenience, but sclite can provide more detailed statistics.

### Reference

This code frame refers to [[VAC]](https://github.com/ycmin95/VAC_CSLR)

### Training



Our main program is in `./stgcn/__main__.py`.So run the command below,to train the SLR model on phoenix14:

`python ./stgcn --config configs/stgcn_Bn_VAC_au_CCE10_W5_re.yaml`


