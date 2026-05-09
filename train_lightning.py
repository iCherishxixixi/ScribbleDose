import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim

import data_loader
import yaml
import argparse
import os

from model.DoseSCP.dosesp import DoseSP

from utils.boundary_utils import boundary_loss
from utils.ptv_utils import dose_ranking_loss
'''
In this script, we provide a basic (and simple) pipeline designed for successful execution.
There are numerous advanced AI methodologies and strategies that could potentially improve the model's performance. 
We encourage participants to explore these AI technologies independently. The organizers will not provide much support for these explorations.
Please note that discussions/questions about AI tech explorations are not supposed to be raised in the repository issues.

Reminder: The information provided in the meta files is crucial, as it directly impacts how the reference is created. 
An example of how to use these information are provided in the data_loader.py. 
If you have questions related to clinical backgrounds, feel free to start a discussion.
'''

class GDPLightningModel(pl.LightningModule):
    def __init__(self, cfig, modelname):
        super(GDPLightningModel, self).__init__()
        if modelname == 'dosesp':
            self.model = DoseSP(
                img_size=cfig['loader_params']['in_size'], 
                in_channels=cfig['model_params']['num_input_channels'], 
                out_channels=cfig['model_params']['out_channels']
            )      
        
        self.criterion = nn.L1Loss()
        self.lr = cfig['lr']
        self.num_epochs = cfig['num_epochs']
        self.cfig = cfig
        self.sig_act = nn.Sigmoid()

    def training_step(self, batch, batch_idx):

        inputs = batch['data']
        labels = batch['label']
        supervoxels = batch['supervoxels']

        outputs = self.model(inputs)

        if isinstance(self.model, DoseSP):

            dose_logits, coarse_seg, compact_loss, sep_loss, scribble_loss = outputs

            if self.cfig['act_sig']:
                dose_logits = self.sig_act(dose_logits)

            dose_loss = self.criterion(
                dose_logits * self.cfig['scale_out'],
                labels
            ) * self.cfig['scale_loss']

            boundary_weight = 0.05 if self.current_epoch > 15 else 0.0
            ranking_weight  = 0.05 if self.current_epoch > 20 else 0.0

            b_loss = torch.tensor(0.0, device=inputs.device)
            ranking_loss = torch.tensor(0.0, device=inputs.device)

            if boundary_weight > 0:
                b_loss = boundary_loss(coarse_seg, supervoxels)

            if ranking_weight > 0:
                ranking_loss = dose_ranking_loss(coarse_seg.detach(), labels)

            total_loss = (
                dose_loss
                + 0.3 * scribble_loss
                + 0.05 * compact_loss
                + 0.02 * sep_loss
                + boundary_weight * b_loss
                + ranking_weight * ranking_loss
            )

            self.log('train_dose_loss', dose_loss, prog_bar=True)
            
            if boundary_weight > 0:
                self.log('train_boundary_loss', b_loss)
                
            if ranking_weight > 0:
                self.log('train_dose_ranking_loss', ranking_loss)
                                
            self.log('train_scribble_loss', scribble_loss)
            self.log('train_loss', total_loss, prog_bar=True)

            return total_loss
        else:

            if self.cfig['act_sig']:
                outputs = self.sig_act(outputs)

            loss = self.criterion(
                outputs * self.cfig['scale_out'],
                labels
            ) * self.cfig['scale_loss']

            self.log('train_loss', loss, prog_bar=True)
            return loss
    
    def validation_step(self, batch, batch_idx):

        inputs = batch['data']
        labels = batch['label']
        supervoxels = batch['supervoxels']

        outputs = self.model(inputs)

        if isinstance(self.model, DoseSP):

            dose_logits, coarse_seg, compact_loss, sep_loss, scribble_loss = outputs

            if self.cfig['act_sig']:
                dose_logits = self.sig_act(dose_logits)

            dose_loss = self.criterion(
                dose_logits * self.cfig['scale_out'],
                labels
            ) * self.cfig['scale_loss']

            b_loss = boundary_loss(coarse_seg, supervoxels)
            ranking_loss = dose_ranking_loss(coarse_seg, labels)
            
            boundary_weight = 0.05 if self.current_epoch > 15 else 0.0
            ranking_weight  = 0.05 if self.current_epoch > 20 else 0.0

            total_loss = (
                dose_loss
                + 0.3 * scribble_loss
                + 0.05 * compact_loss
                + 0.02 * sep_loss
                + boundary_weight * b_loss
                + ranking_weight * ranking_loss
            )

            self.log('val_dose_loss', dose_loss)
            self.log('val_loss', total_loss, prog_bar=True)

            return total_loss

        else:
            if self.cfig['act_sig']:
                outputs = self.sig_act(outputs)
    
            loss = self.criterion(
                outputs * self.cfig['scale_out'],
                labels
            ) * self.cfig['scale_loss']

            self.log('val_loss', loss, prog_bar=True)
            return loss


    def configure_optimizers(self):
        optimizer = optim.Adam([{'params': self.model.parameters(), 'initial_lr': self.lr}], lr=self.lr)
        scheduler = {'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= self.num_epochs), 
                     'interval': 'epoch', 'frequency': 1}
        return [optimizer], [scheduler]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('cfig_dir', type=str)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--model', default='mednext', type=str)
    args = parser.parse_args()
    
    args.cfig_path = os.path.join(args.cfig_dir, f"config_{args.model}.yaml")    
    cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)

    # Data Loaders
    loaders = data_loader.GetLoader(cfig=cfig['loader_params'])
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()

    # Model
    model = GDPLightningModel(cfig, args.model)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"   
    if torch.cuda.device_count() > 1:
        stratgy = 'ddp_find_unused_parameters_true'
        sync_batchnorm = True
        use_distributed_sampler = True
        
    else:
        stratgy = 'auto' 
        sync_batchnorm = False
        use_distributed_sampler = False

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfig['save_model_root'],
        filename='best_model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=2,
        monitor='val_loss',
        mode='min', 
        save_last=True
    )

    mylogger = CSVLogger(cfig['save_model_root'])

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfig['num_epochs'],
        devices = 'auto', 
        accelerator=accelerator, strategy= stratgy, sync_batchnorm=sync_batchnorm,
        use_distributed_sampler=use_distributed_sampler, 
        logger=mylogger, 
        default_root_dir=cfig['save_model_root'],
        callbacks=[lr_monitor, checkpoint_callback]
    )

    # Training
    ckpt_path = cfig.get('pretrain_ckpt')
    if ckpt_path:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader)
