import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler

import matplotlib as mpl
import matplotlib.pyplot as plt

from models.MyLOModel2D import find_level_set_constants

# remove top and right axis from plots
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def plot_training_curves(tlosses, vlosses=None, model_dir=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(tlosses)+1), tlosses, label='Train Loss')
    if vlosses is not None:
        plt.plot(range(1, len(vlosses)+1), vlosses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    if model_dir is None:
        plt.show()
    else:
        plt.savefig(f"{model_dir}training_loss_curve.png")
        plt.close()


def train(model, criterion, optimizer, epochs, train_loader, valid_loader=None, report_epochs=50, scheduler=None, patience=25, alpha=0, beta=0, plot=False, model_name='best_model.pth', model_dir=None):
    model_path = model_name if model_dir is None else f"{model_dir}{model_name}"
    
    tlosses, vlosses = [], []

    # Scheduler and patience
    if scheduler is None:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_loss = float('inf')
    epochs_no_improve = 0

    triplet_loss = nn.TripletMarginLoss(p=2)

    for epoch in range(epochs):
        # training step
        tloss = 0
        model.train()
        for batch_idx, ((data, positive, negative), _) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            
            embeddings = model.encode(data)
            embeddings_p, embeddings_n = model.encode(positive), model.encode(negative)
            output = model.decode(embeddings)
            
            rec_loss = criterion(output, data)
            reg_loss = torch.mean(torch.norm(embeddings, p=2, dim=1))
            tri_loss = triplet_loss(embeddings, embeddings_p, embeddings_n)

            loss = rec_loss + alpha * reg_loss + beta * tri_loss
            tloss += loss.item()
            
            loss.backward()
            optimizer.step()

        avg_tloss = tloss / len(train_loader)
        tlosses.append(avg_tloss)

        # validation step
        vloss = 0
        model.eval()
        for batch_idx, ((data, positive, negative), _) in enumerate(valid_loader):
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            
            embeddings = model.encode(data)
            embeddings_p, embeddings_n = model.encode(positive), model.encode(negative)
            output = model.decode(embeddings)
            
            rec_loss = criterion(output, data)
            reg_loss = torch.mean(torch.norm(embeddings, p=2, dim=1))
            tri_loss = triplet_loss(embeddings, embeddings_p, embeddings_n)

            loss = rec_loss + alpha * reg_loss + beta * tri_loss
            vloss += loss.item()

        avg_vloss = vloss / len(valid_loader)
        vlosses.append(avg_vloss)
        
        if (epoch + 1) % report_epochs == 0 or (epoch + 1) == epochs:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {tloss / len(train_loader)}, Validation Loss: {vloss / len(valid_loader)}')

        # Scheduler step
        scheduler.step(avg_tloss)

        # Early stopping
        if avg_vloss < best_loss:
            best_loss = avg_vloss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)  # Save best model
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            model.load_state_dict(torch.load(model_path))
            print("Early stopping triggered")
            break


    if plot:
        plot_training_curves(tlosses, vlosses, model_dir)


def joint_training(model, inference_model, criterion, optimizer, epochs, train_loader, valid_loader=None, report_epochs=50, scheduler=None, patience=25, alpha=0, beta=0, plot=False, model_name='best_model.pth', inference_model_name='best_inference_model.pth', model_dir=None):
    model_path = model_name if model_dir is None else f"{model_dir}{model_name}"
    inference_model_path = model_name if model_dir is None else f"{model_dir}{inference_model_name}"
    
    tlosses, vlosses = [], []

    # Scheduler and patience
    if scheduler is None:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_loss = float('inf')
    epochs_no_improve = 0

    triplet_loss = nn.TripletMarginLoss(p=2)
    flow_loss = inference_model._loss

    def default_calibration_kernel(x):
        return torch.ones_like(x[:, 0])

    for epoch in range(epochs):
        # training step
        tloss = 0
        model.train()
        for batch_idx, ((data, positive, negative), params) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            
            # Embeddings
            embeddings = model.encode(data)
            embeddings_p, embeddings_n = model.encode(positive), model.encode(negative)
            
            # Reconstruction
            output = model.decode(embeddings)

            # Compute combined loss
            rec_loss = criterion(output, data)
            tri_loss = triplet_loss(embeddings, embeddings_p, embeddings_n)
            est_loss = flow_loss(theta=params, x=embeddings, masks=torch.ones(params.shape[0], dtype=torch.bool, device=params.device), proposal=None, calibration_kernel=default_calibration_kernel).mean()

            loss = est_loss + alpha * rec_loss + beta * tri_loss
            tloss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_tloss = tloss / len(train_loader)
        tlosses.append(avg_tloss)

        # validation step
        vloss = 0
        model.eval()
        for batch_idx, ((data, positive, negative), params) in enumerate(valid_loader):
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            
            # Embeddings
            embeddings = model.encode(data)
            embeddings_p, embeddings_n = model.encode(positive), model.encode(negative)
            
            # Reconstruction
            output = model.decode(embeddings)

            # Compute combined loss
            rec_loss = criterion(output, data)
            tri_loss = triplet_loss(embeddings, embeddings_p, embeddings_n)
            est_loss = flow_loss(theta=params, x=embeddings, masks=torch.ones(params.shape[0], dtype=torch.bool, device=params.device), proposal=None, calibration_kernel=default_calibration_kernel).mean()

            loss = est_loss + alpha * rec_loss + beta * tri_loss
            vloss += loss.item()

        avg_vloss = vloss / len(valid_loader)
        vlosses.append(avg_vloss)
        
        if (epoch + 1) % report_epochs == 0 or (epoch + 1) == epochs:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {tloss / len(train_loader)}, Validation Loss: {vloss / len(valid_loader)}')

        # Scheduler step
        scheduler.step(avg_tloss)

        # Early stopping
        if avg_vloss < best_loss:
            best_loss = avg_vloss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)  # Save best model
            torch.save(inference_model._neural_net.state_dict(), inference_model_path)

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            model.load_state_dict(torch.load(model_path))
            inference_model._neural_net.load_state_dict(torch.load(inference_model_path))
            print(f"Early stopping triggered at epoch {epoch}")
            break


    if plot:
        plot_training_curves(tlosses, vlosses, model_dir)

    return 


def test(model, criterion, test_loader, n=3, plot=False, model_dir=None):
    # test step
    tloss = 0
    model.eval()
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.view(data.size(0), -1)
        
        embeddings = model.encode(data)
        output = model.decode(embeddings)
        
        loss = criterion(output, data)
        tloss += loss.item()

        if plot and batch_idx == 0:
            data, output = data.detach().numpy(), output.detach().numpy()

            plt.figure(figsize=(10, 5))
            fig, axs = plt.subplots(n)
            for k in range(n):
                axs[k].plot(range(len(data[k])), data[k], label='Original trace')
                axs[k].plot(range(len(output[k])), output[k], label='Reconstructed trace', ls='dashed')
                axs[k].set_ylabel(f'Trace {k+1}')
                axs[k].legend()
                axs[k].grid()
            plt.xlabel('Time')
            plt.suptitle('Trace reconstructions')
            if model_dir is None:
                plt.show()
            else:
                plt.savefig(f"{model_dir}test_reconstruction.png")
                plt.close()
    
    tloss /= len(test_loader)
        
    print(f'The Total Reconstruction Loss for the Test Dataset is: {tloss}')


def encode(model, test_loader, p=0, ls=False, model_dir=None):
    model.eval()
    codes, color = None, None
    for _, (data, labels) in enumerate(test_loader):
        data = data.view(data.size(0), -1)
        
        if codes is None:
            codes = model.encode(data).detach()
            if ls:
                levelset = find_level_set_constants(labels.detach())
                if p == 0:
                    color = levelset['K_a']
                else: # p == 1:
                    color = levelset['K_f']
            else:
                color = labels[:, p].detach()
        else:
            codes = torch.cat((codes, model.encode(data).detach()), dim=0)
            if ls:
                levelset = find_level_set_constants(labels.detach())
                if p == 0:
                    color = torch.cat((color, levelset['K_a']), dim=0)
                else: # p == 1:
                    color = torch.cat((color, levelset['K_f']), dim=0)
            else: 
                color = torch.cat((color, labels[:, p].detach()), dim=0)

    n = codes.shape[-1]
    label_names = [f'$K_a$', f'$K_f$'] if ls else [f'$\lambda$', f'$b$']

    fig, axs = plt.subplots(n-1, n-1)
    if n == 2:
        im = axs.scatter(codes[:, 0], codes[:, 1], c=color, cmap='viridis')
        plt.colorbar(im, label=label_names[p])
        plt.xlabel(f'Feature {2}')
        plt.ylabel(f'Feature {1}')
    else:
        for i in range(n-1):
            for j in range(n-1):
                if i <= j:
                    im = axs[i, j].scatter(codes[:, j+1], codes[:, i], c=color, cmap='viridis')
                    axs[i, j].grid()
                    if i == j:
                        axs[i, j].set_xlabel(f'Feature {j+2}')
                        axs[i, j].set_ylabel(f'Feature {i+1}')
                else:
                    axs[i, j].set_visible(False)
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical', label=label_names[p])

    plt.suptitle(f'Embedding Space ({label_names[p]})')
    if model_dir is None:
        plt.show()
    else:
        plt.savefig(f"{model_dir}test_encodings__color={label_names[p]}.png")
        plt.close()