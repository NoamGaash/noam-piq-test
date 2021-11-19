import argparse
import os
import pathlib
import lpips
from piq import SSIMLoss

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

# crawl directories
f = open(opt.out,'w')
#files = os.listdir(opt.dir0)
files = [os.path.relpath(f, opt.dir0) for f in pathlib.Path(opt.dir0).glob("**/*") if f.is_file()]

if len(files) is 0:
	raise "no files in directory 0"

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

def getMetric(loss_fn):
	count = 0.
	sum = 0.

	for file in files:
		if(os.path.exists(os.path.join(opt.dir1,file))):
			# Load images
			img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
			img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))

			img0 = img0 * .5 + .5
			img1 = img1 * .5 + .5

			if(opt.use_gpu):
				img0 = img0.cuda()
				img1 = img1.cuda()

			# Compute distance
			dist01 = loss_fn(img0, img1)
			sum += dist01
			count += 1

	return sum/count


print('compare %s vs %s'%(opt.dir0, opt.dir1))
f.writelines('compare %s vs %s \n'%(opt.dir0, opt.dir1))


lpips_loss = getMetric(loss_fn.forward)
print('lpips: %.3f'%(lpips_loss))
f.writelines('lpips: %.3f \n'%(lpips_loss))


ssim_loss = getMetric(SSIMLoss(data_range=1.))
print('ssim: %.3f'%(ssim_loss))
f.writelines('ssim: %.3f \n'%(ssim_loss))

f.close()
