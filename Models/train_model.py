import torch
import Utils.utils
import os
import copy


def train(model, criterion, optim, data_loaders, scheduler, num_epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = os.path.join(os.getcwd(), 'trained-models/net.pth')

    train_loss_history = []
    test_loss_history = []

    best_acc = 0.0

    for epoch in range(num_epochs):

        print(f'\nEpoch: {epoch+1}/{num_epochs}')

        phases = ['train', 'test']

        for phase in phases:

            print(phase.upper())

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for index, (inputs, targets) in enumerate(data_loaders[phase]):

                Utils.utils.update_progress(index+1, len(data_loaders[phase]))

                inputs = inputs.to(device)
                targets = targets.to(device)

                if phase == 'train':
                    optim.zero_grad()

                    # in case val is needed

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, targets)

                        if phase == 'train':

                            loss.backward()
                            optim.step()

                else:

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += preds.eq(targets).sum().item()
                total += targets.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total

            train_loss_history.append(epoch_loss) if phase == 'train' else test_loss_history.append(epoch_loss)

            print(f'Loss: {epoch_loss}, Accuracy: {epoch_acc}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                if not os.path.isdir('trained-models'):
                    os.mkdir('trained-models')

                if scheduler is not None:
                    state = {

                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optim.state_dict(),
                        'scheduler': scheduler.state_dict()

                    }
                else:
                    state = {

                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optim.state_dict(),

                    }

                if os.path.exists(save_path):
                    os.remove(save_path)

                torch.save(state, save_path)

    print(f'Best Accuracy: {best_acc}')

    model.load_state_dict(best_model_wts)

    return model, train_loss_history, test_loss_history
