python -m torch.distributed.run --nproc_per_node=8 --master_port=25641 train.py --cfg-path ./lavis/projects/blip2/train/vg_coco_vicuna7b.yaml

