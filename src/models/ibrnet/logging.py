from src.models.ibrnet.utils import *


def convert_image(ret):
    ret = ret.numpy()
    ret = np.clip(ret, 0, 1)
    ret *= 255
    ret = ret.astype(np.uint8)
    ret = ret.transpose(2, 0, 1)
    return ret


def log_scalars(writer, loss, global_step):
    loss = loss.detach().cpu().numpy()
    scalars_to_log = {}
    mse_error = loss
    scalars_to_log['train/mse'] = mse_error
    scalars_to_log['train/psnr'] = mse2psnr(mse_error)
    for k in scalars_to_log.keys():
        writer.experiment.add_scalar(k, scalars_to_log[k], global_step)


def log_img(renderer, writer, batch, global_step, mode='train'):
    local_rank = 0
    renderer.eval()
    with torch.no_grad():
        ret = renderer(batch, local_rank)
    rend_img, rend_depth = ret['rgb'], ret['depth']
    render_depth = img_HWC2CHW(colorize(rend_depth, cmap_name='jet', append_cbar=True))
    render_img = convert_image(rend_img)
    writer.experiment.add_image(f'{mode}/render_img', render_img, global_step)
    writer.experiment.add_image(f'{mode}/render_depth', render_depth, global_step)

    renderer.train()

class LossLogger():
    def __init__(self):
        self.loss = {'loss': 0, 'mse_loss': 0, 'entropy_reg': 0}
        self.count = 0

    def log(self, loss):
        for k in self.loss.keys():
            self.loss[k] += loss[k]
        self.count += 1

    def write(self, logger, global_step):
        for k in self.loss.keys():
            logger.experiment.add_scalar(k, self.loss[k] / self.count, global_step)
        mse_error = self.loss['mse_loss'] / self.count
        logger.experiment.add_scalar('psnr', mse2psnr(mse_error), global_step)
        self.loss = {'loss': 0, 'mse_loss': 0, 'entropy_reg': 0}
        self.count = 0


class PoseLogger():
    def __init__(self):
        self.loss = {'loss': 0, 'loss_r': 0, 'loss_t': 0, 'loss_mc': 0, 'loss_sim': 0, 'loss_kl': 0, 'loss_mse': 0, 'conf_max': 0}
        self.count = 0

    def log(self, loss):
        for k in self.loss.keys():
            self.loss[k] += loss[k]
        self.count += 1

    def write(self, logger, global_step):
        for k in self.loss.keys():
            logger.experiment.add_scalar(k, self.loss[k] / self.count, global_step)
        self.loss = {'loss': 0, 'loss_r': 0, 'loss_t': 0, 'loss_mc': 0, 'loss_sim': 0, 'loss_kl': 0, 'loss_mse': 0, 'conf_max': 0}
        self.count = 0