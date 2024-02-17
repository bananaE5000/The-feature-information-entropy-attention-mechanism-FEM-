import os
class Config(object):
	def __init__(self):
		self.num_epochs = 50
		self.init_lr = 1e-4
		self.batch_size = 1

		self.dpi=(1.0, 1.0)

		self.n_label = 6

		self.seed = 5
		self.augment = True



		self.size_train = (800, 800)

		# image channel-wise mean to subtract, the order is BGR
		self.img_channel_mean = [103.939, 116.779, 123.68]

	def check_folder(self, log_dir):
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		return log_dir