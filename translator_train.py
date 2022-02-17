from translator.preprocess import get_dataset, DataLoader, collate_fn_transformer
from translator.network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def main():

    dataset = get_dataset()
    global_step = 0
    
    m = nn.DataParallel(Model().cuda())

    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    pos_weight = t.FloatTensor([5.]).cuda()
    writer = SummaryWriter()

    if hp.checkpoint is not None:
        checkpoint = t.load(hp.checkpoint)
        m.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        global_step = checkpoint['global_step']
        loss = checkpoint['loss']
        print('Loaded checkpoint, starting at step %d'%global_step)
        print('Loss = %d'%loss)
    
    m.train()
    
    for epoch in range(hp.epochs):

        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)
                
            character, mel, embed, mel_input, pos_text, pos_mel, _ = data
            
            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
            
            character = character.cuda()
            mel = mel.cuda()
            embed = embed.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, embed, mel_input, pos_text, pos_mel)

            mel_loss = nn.L1Loss()(mel_pred, mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)
            
            loss = mel_loss + post_mel_loss
            
            writer.add_scalars('training_loss',{
                    'mel_loss':mel_loss,
                    'post_mel_loss':post_mel_loss,

                }, global_step)
                
            writer.add_scalars('alphas',{
                    'encoder_alpha':m.module.encoder.alpha.data,
                    'decoder_alpha':m.module.decoder.alpha.data,
                }, global_step)
            
            
            if global_step % hp.image_step == 1:
                
                for i, prob in enumerate(attn_probs):
                    
                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_%d_0'%global_step, x, i*4+j)
                
                for i, prob in enumerate(attns_enc):
                    num_h = prob.size(0)
                    
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)
            
                for i, prob in enumerate(attns_dec):

                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)
                
            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            
            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                t.save({'model':m.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'loss' : loss.item()
                        },
                        os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))
            pbar.set_postfix({
                                "loss": loss.item(),
                                "mel_loss": mel_loss.item(),
                                "post_mel_loss" : post_mel_loss.item()
                            })

if __name__ == '__main__':
    main()