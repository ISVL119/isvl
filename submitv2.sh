# python 1_image_splitter.py
# python 1_restructure_mvtec_dataset.py

# python isvl.py  --item_list can --total_epochs 10
# python isvl.py  --item_list fabric --total_epochs 10
# python isvl.py  --item_list rice --total_epochs 10
# python isvl.py  --item_list sheet_metal --total_epochs 10
# python isvl.py  --item_list wallplugs --total_epochs 10
# python isvl.py  --item_list walnuts --total_epochs 10


# foreground
# python tools/generate_foreground.py -lp log/foreground/foreground_mvtec_test_vial_fruit \
# --dataset-name mvtec_test_vial_fruit \
# --layer features.denseblock1 \
# -pm DenseNet

# # retrieval
# python tools/generate_retrieval.py -lp log/retrieval/retrieval_mvtec_test_vial_fruit \
# --dataset-name mvtec_test_vial_fruit \
# --layer features.denseblock1 \
# -pm DenseNet


# synthesize_data_fruit
# python tools/generate_synthesize_hand.py --output_dir log/synthesized/synthesized_mvtec_test_vial_fruit \
# --dataset-name mvtec_test_vial_fruit \
# --normal_dir data/mvtec_test_vial_fruit \
# --mask_dir log/foreground/foreground_mvtec_test_vial_fruit \
# --num_per_image 5 \
# --resize 640 \
# --seed 42 \
# --category fruit_jelly

# # synthesize_data_vial
# python tools/generate_synthesize_hand.py --output_dir log/synthesized/synthesized_mvtec_test_vial_fruit \
# --dataset-name mvtec_test_vial_fruit \
# --normal_dir data/mvtec_test_vial_fruit \
# --mask_dir log/foreground/foreground_mvtec_test_vial_fruit \
# --num_per_image 3 \
# --resize 640 \
# --seed 66 \
# --category vial



# train
# python train.py \
#  -fd log/foreground/foreground_mvtec_test_vial_fruit \
#  --steps 2000 \
#  -tps 2000 \
#  --data-dir log/synthesized/synthesized_mvtec_test_vial_fruit \
#  --retrieval-dir log/retrieval/retrieval_mvtec_test_vial_fruit \
#  --dataset-name mvtec_test_vial_fruit \
#  --category fruit_jelly
# #


# # train
# python train.py \
#   -fd log/foreground/foreground_mvtec_test_vial_fruit \
#   --steps 1300 \
#   -tps 1300 \
#   --data-dir log/synthesized/synthesized_mvtec_test_vial_fruit \
#   --retrieval-dir log/retrieval/retrieval_mvtec_test_vial_fruit \
#   --dataset-name mvtec_test_vial_fruit \
#   --category vial

# test
# python isvl.py  --item_list can --total_epochs 10  --phase val
# python isvl.py  --item_list fabric --total_epochs 10   --phase val
# python isvl.py  --item_list rice --total_epochs 10 --phase val
# python isvl.py  --item_list sheet_metal --total_epochs 10  --phase val
# python isvl.py  --item_list wallplugs --total_epochs 10    --phase val
# python isvl.py  --item_list walnuts --total_epochs 10  --phase val


# python test_new.py -fd log/foreground/foreground_mvtec_test_vial_fruit \
# --checkpoints log/chekpoints/chekpoints_mvtec_test_vial_fruit_True/fruit_jelly/02000.pth \
# -rd log/retrieval/retrieval_mvtec_test_vial_fruit \
# -dn mvtec_test_vial_fruit \
# --sub-categories fruit_jelly

# python test_new.py -fd log/foreground/foreground_mvtec_test_vial_fruit \
# --checkpoints log/chekpoints/chekpoints_mvtec_test_vial_fruit_True/vial/01300.pth \
# -rd log/retrieval/retrieval_mvtec_test_vial_fruit \
# -dn mvtec_test_vial_fruit \
# --sub-categories vial


# python 2_image_reconstruction.py
# python 3_replace_and_rename_folders.py
python 4_threshold_mapv2.py
# python 5_post_image_process.py
# python 6_replace_and_rename_folders.py
python 7_convert_tiff_to_float16.py
# python 8_check_and_prepare_data_for_upload.py "./results/"