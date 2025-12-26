import hydra
import sys
import os
import logging
import inspect
import torch
torch.cuda.empty_cache()
print(f"Initial memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

log = logging.getLogger(__name__)
sys.path.append(os.getcwd())
from prior.utils.train_utils import setup_training, init_trainer, init_pretrain_model
from prior.utils.wandb_utils import finish_run



@hydra.main(config_path="../../configs", config_name="base_pre_train")
def pre_train(config):
    setup_training(config)  # 是否初始化seed 或者 是否开启debug模式
    model = init_pretrain_model(config)    # 实例化模型
    ckpt_path = None
    if 'resume_from' in config.keys():
        model_paras = inspect.signature(model.__init__).parameters.keys()
        model_kwargs = {}
        for k in model_paras:
            try:
                model_kwargs[k] = getattr(model, k)
            except:
                model_kwargs[k] = getattr(config.pretrain_model, k)
        log.info('Resuming training from {}'.format(config.resume_from))
        model.load_from_checkpoint(config.resume_from, **model_kwargs, strict=True)
        ckpt_path = config.resume_from
    trainer = init_trainer(model, config)
    if config.trainer.auto_lr_find:
        log.info(f"----- Tuning LR -----")
        trainer.tune(model)
        log.info(f"----- Completed LR Tuning -----")
    trainer.fit(model, ckpt_path=ckpt_path)
    # run_api, _ = finish_run(trainer)
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    pre_train()
     