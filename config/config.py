from yacs.config import CfgNode as CN

config = {"windows": CN(), "linux": CN()}



# windows config

config["windows"] = CN()

config["windows"].project_path = "D:/python/python_data/SiamRPN_rebuild"

config["windows"].datasets = CN()
config["windows"].datasets.TC_dataset_path = "E:/dataset/tc"
config["windows"].datasets.DTB70_dataset_path = "E:/dataset/DTB70/DTB70"
config["windows"].datasets.UAV123_dataset_path = "E:/dataset/Dataset-UAV-123/data/UAV123"

config["windows"].datasets.Coco_Dataset_Path = "E:/dataset/coco"
config["windows"].datasets.Coco_Dataset_GT_Path = "E:/dataset/coco/gt/refine_simple_coco2017.json"
config["windows"].datasets.Coco_Pre_50000_Dataset_GT_Path = "E:/dataset/coco/gt/refine_simple_coco2017_pre_50000.json"
config["windows"].datasets.Coco_Pre_200000_Dataset_GT_Path = "E:/dataset/coco/gt/refine_simple_coco2017_pre_200000.json"
config["windows"].datasets.Coco_Pre_250000_Dataset_GT_Path = "E:/dataset/coco/gt/refine_simple_coco2017_pre_250000.json"

config["windows"].cuda_no = 0


config["windows"].train = CN()
config["windows"].train.lr = 1e-5
config["windows"].train.epoch = 1
config["windows"].train.batch_size = 1
config["windows"].train.num_workers = 1
config["windows"].train.model_save_folder = "D:/python/python_data/SiamRPN_rebuild/save"

config["windows"].build_feature_map = CN()
config["windows"].build_feature_map.batch_size = 8
config["windows"].build_feature_map.num_workers = 2

# linux_config

config["linux"] = CN()

config["linux"].project_path = "/home/python_data/SiamRPNPP_Change"

config["linux"].datasets = CN()
config["linux"].datasets.TC_dataset_path = "/home/datasets/tc"
config["linux"].datasets.DTB70_dataset_path = "/home/datasets/DTB70"

config["linux"].datasets.TC_FeatureMap_dataset_path = "/home/datasets/tc_resnet50_backbone_neck_feature_map"
config["linux"].datasets.DTB70_FeatureMap_dataset_path = "/home/datasets/dtb70_resnet50_backbone_neck_feature_map"
config["linux"].datasets.Coco_Dataset_Path = "/home/datasets/coco"
config["linux"].datasets.Coco_Dataset_GT_Path = "/home/datasets/coco/gt/refine_simple_coco2017.json"
config["linux"].datasets.Coco_Pre_50000_Dataset_GT_Path = "/home/datasets/coco/gt/refine_simple_coco2017_pre_50000.json"
config["linux"].datasets.Coco_Pre_200000_Dataset_GT_Path = "/home/datasets/coco/gt/refine_simple_coco2017_pre_200000.json"
config["linux"].datasets.Coco_Pre_250000_Dataset_GT_Path = "/home/datasets/coco/gt/refine_simple_coco2017_pre_250000.json"

config["linux"].cuda_no = 2

config["linux"].train = CN()
config["linux"].train.lr = 1e-5
config["linux"].train.epoch = 5
config["linux"].train.batch_size = 6
config["linux"].train.num_workers = 4
config["linux"].train.nproc_per_node = 4
config["linux"].train.model_save_folder = "/home/python_data/SiamRPNPP_Change/save"

config["linux"].build_feature_map = CN()
config["linux"].build_feature_map.batch_size = 24
config["linux"].build_feature_map.num_workers = 4








