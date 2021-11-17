from glob import glob
import pandas as pd

folder = 'results/iuv/'

files = glob(folder + '*')

lpips_array = []
ssim_array = []

for filename in files:
    file = open(filename, "r").read()
    file = file.split("\n")
    try:
        lpips = float(file[1].split(' ')[1])
        ssim = float(file[2].split(' ')[1])
        pose = filename.split("_")[2]
        appearence = filename.split("_")[-1][:4]
        print(filename, lpips, ssim, int(pose), int(appearence))
        
        lpips_array.append({
            'pose': pose,
            'appearence': appearence,
            'lpips': lpips
        })
        ssim_array.append({
            'pose': pose,
            'appearence': appearence,
            'ssim': ssim
        })


    except:
        print(filename, ' - error')

lpips_df = pd.DataFrame(lpips_array)
lpips_df.set_index(['pose', 'appearence'], inplace=True)
lpips_df = lpips_df.unstack()

ssim_df = pd.DataFrame(ssim_array)
ssim_df.set_index(['pose', 'appearence'], inplace=True)
ssim_df = ssim_df.unstack()

print(ssim_df)
pd.to_pickle(lpips_df, folder + 'lpips.pickle')
pd.to_pickle(ssim_df, folder + 'ssim.pickle')