import torch
import torch.optim as optim
from torch.nn import MSELoss
from model import FTTransformer
import os

def pretrain(args, train_loader, test_loader, num_numerical_cols, device, epochs, pretrain_model_name):
    model = FTTransformer(
        num_numerical_cols=num_numerical_cols,
        hidden_dim=args['hidden_dim'],
        pf_dim=args['pf_dim'],
        num_heads=args['num_heads'],
        num_layers=args['num_layers'],
        output_dim=1,
        dropout_ratio=args['dropout_ratio'],
        device=device
    ).to(device)

    criterion = MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_test_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for masked_features, mask, original_features, _ in train_loader:
            masked_features, mask, original_features = masked_features.to(device), mask.to(device), original_features.to(device)

            optimizer.zero_grad()
            outputs = model(masked_features, mask)

            masked_loss = criterion(outputs * mask, original_features * mask)
            unmasked_loss = criterion(outputs * (1 - mask), original_features * (1 - mask))
            loss = masked_loss + unmasked_loss
            loss = loss.sum()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * masked_features.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for masked_features, mask, original_features, _ in test_loader:
                masked_features, mask, original_features = masked_features.to(device), mask.to(device), original_features.to(device)
                outputs = model(masked_features, mask)
                masked_loss = criterion(outputs * mask, original_features * mask)
                unmasked_loss = criterion(outputs * (1 - mask), original_features * (1 - mask))
                loss = masked_loss + unmasked_loss
                loss = loss.sum()
                test_loss += loss.item() * masked_features.size(0)

        test_loss /= len(test_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_path = os.path.join(args['save_folder'], pretrain_model_name)
            torch.save(model.state_dict(), save_path)

    print(f'Pretraining complete. Best Test Loss: {best_test_loss:.4f}')
