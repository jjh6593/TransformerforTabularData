import torch
import torch.optim as optim
from torch.nn import MSELoss
from model import FTTransformerNew
import os

def posttrain(args, train_loader, test_loader, num_numerical_cols, device, epochs, posttrain_model_name):
    old_model_path = os.path.join(args['save_folder'], args['pretrain_model_name'])

    new_model = FTTransformerNew(
        num_numerical_cols=num_numerical_cols,
        hidden_dim=args['hidden_dim'],
        pf_dim=args['pf_dim'],
        num_heads=args['num_heads'],
        num_layers=args['num_layers'],
        dropout_ratio=args['dropout_ratio'],
        device=device
    ).to(device)

    old_model_dict = torch.load(old_model_path)
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in old_model_dict.items() if k in new_model_dict and 'output_layer1' not in k}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)

    criterion = MSELoss()
    optimizer = optim.Adam(new_model.fc.parameters(), lr=args['learning_rate'])
    best_test_loss = float('inf')

    for epoch in range(epochs):
        new_model.train()
        train_loss = 0.0
        for masked_features, mask, original_features, y in train_loader:
            masked_features, mask, original_features, y = masked_features.to(device), mask.to(device), original_features.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = new_model(original_features)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * original_features.size(0)

        train_loss /= len(train_loader.dataset)

        new_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for masked_features, mask, original_features, y in test_loader:
                masked_features, mask, original_features, y = masked_features.to(device), mask.to(device), original_features.to(device), y.to(device)
                outputs = new_model(original_features)
                loss = criterion(outputs, y)
                test_loss += loss.item() * original_features.size(0)

        test_loss /= len(test_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_path = os.path.join(args['save_folder'], posttrain_model_name)
            torch.save(new_model.state_dict(), save_path)

    print(f'Posttraining complete. Best Test Loss: {best_test_loss:.4f}')
