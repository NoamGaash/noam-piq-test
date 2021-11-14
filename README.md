# my python image quality test

* the script run recursively on both given folders.
* absolute and relative pathes can both be used 

```shell
    python lpips_2dirs.py \
        -d0 "/home/noam/Documents/4-4 dataset/output/casia-no-bg/all_subjects/results/WS30/CASIA/Dataset_IUV_from_appearance_0000/0000/00/001/" \
        -d1 "/home/noam/Documents/4-4 dataset/output/casia-no-bg/all_subjects/Crop256/rgb/0000/00/001/" \
        -o out.txt
    python FID_2dirs.py \
        -d0 "/home/noam/Documents/4-4 dataset/output/casia-no-bg/all_subjects/results/WS30/CASIA/Dataset_IUV_from_appearance_0000/0000/00/001/" \
        -d1 "/home/noam/Documents/4-4 dataset/output/casia-no-bg/all_subjects/Crop256/rgb/0000/00/001/" \
        -o out.txt
```

