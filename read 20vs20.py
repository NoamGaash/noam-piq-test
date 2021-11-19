from glob import glob
import pandas as pd

folder = 'results/sticks/'

files = glob(folder + '*out_pose*.txt')

metrics_array = []

for filename in files:

    file = open(filename, "r").read()
    file += open(filename.replace("out_pose", "out_IS_FID_pose"), "r").read()
    file = file.split()
    pose = filename.split("_")[-3]
    appearence = filename.split("_")[-1][:4]

    try: # try get lpips and ssim
        lpips_ind = file.index('lpips:') + 1
        lpips = float(file[lpips_ind])

        ssim_ind = file.index('ssim:') + 1
        ssim = float(file[ssim_ind])

        fid_ind = file.index('fid:') + 1
        fid = float(file[fid_ind])
        IS_ind = file.index('IS:') + 1
        IS = float(file[IS_ind])
        
        metrics_array.append({
            'pose': pose,
            'appearence': appearence,
            'lpips': lpips,
            'ssim': ssim,
            'fid': fid,
            'IS': IS
        })

    except:
        print(filename, ' - error')


metrics_df = pd.DataFrame(metrics_array)
metrics_df.set_index(['pose', 'appearence'], inplace=True)
metrics_df = metrics_df.unstack()

print(metrics_df)
pd.to_pickle(metrics_df, folder + 'metrics.pickle')
