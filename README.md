## Install the environment
conda create -n ostrack python=3.12
conda activate ostrack
bash install.sh

## create local file
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output

## training 
python tracking/train.py --script test --config vitb_256_mae_ce_32x4_ep300_vis --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0
python tracking/train.py --script test --config vitb_256_mae_ce_32x4_ep300_event --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0

python tracking/train.py --script test --config vitb_256_mae_ce_32x4_ep300_hybrid_ef --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0
python tracking/train.py --script test --config vitb_256_mae_ce_32x4_ep300_hybrid_hw_mf --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 0 --nohup 1

## test
python tracking/test.py test vitb_256_mae_ce_32x4_ep300_hybrid_mf --dataset visevent --threads 16 --num_gpus 4
python tracking/test.py test vitb_256_mae_ce_32x4_ep300_hybrid_hw_mf --dataset visevent --threads 16 --num_gpus 4
python tracking/test.py test vitb_256_mae_ce_32x4_ep300_hybrid_lf --dataset visevent --threads 16 --num_gpus 4

## speed
python tracking/profile_model.py --script test --config vitb_256_mae_ce_32x4_ep300_hybrid_mf
python tracking/profile_model.py --script test --config vitb_256_mae_ce_32x4_ep300_hybrid_hw_mf

## analysis
python tracking/analysis_results.py

##tensorboard
tensorboard --logdir=tensorboard/train/test


## hardware test
python3 /usr/local/lynxi/sdk/tools/modelTool/modelTool.py --modelPath ./Net_0/ --input ./Net_0/apu_0/case0/net0/chip0/tv_mem/data/input.dat --output ./output.dat --repeat 10
python3 /usr/local/lynxi/sdk/tools/modelTool/modelTool.py --modelPath ./onnx.mdl --input ./pytorch/Net_0/apu_0/case0/net0/chip0/tv_mem/data/input.dat --output ./output.dat --repeat 10

## analysis
python tracking/analysis_results.py

## debug
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset vot22 --threads 1 --num_gpus 1 --debug 1

## compare 
compare multi -f ONNX -m ../Test_ep0300.onnx -i "{'template':(1,3,128,128),'search':(1,3,256,256),'template_event':(1,3,128,128),'search_event':(1,3,256,256)}" -a -o ./cpmpare_result



## hardware convert

first
lib/models/test/test.py : function forward : return out
lib/models/layers/head.py : class CenterPredictor : function forward : avoid box mapback
lib/models/test/vit_hybrid.py : class VisionTransformerHWMF : cross_index adaptive