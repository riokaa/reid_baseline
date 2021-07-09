# MODEL.PRETRAIN_PATH '/export/home/lxy/.cache/torch/checkpoints/resnet50_ibn_a.pth.tar' \
python3 tools/train.py -cfg='configs/softmax_triplet.yml' \
DATASETS.NAMES '("market1501",)'  \
DATASETS.TEST_NAMES 'market1501' \
INPUT.DO_LIGHTING 'False' \
MODEL.GPUS 4 \
MODEL.WITH_IBN 'False'  \
MODEL.STAGE_WITH_GCB '(False, False, False, False)' \
SOLVER.MAX_EPOCHS 1 \
SOLVER.IMS_PER_BATCH '256' \
SOLVER.LOSSTYPE '("softmax_smooth", "triplet", "center")' \
OUTPUT_DIR 'logs/2021.7.9/market/baseline'
