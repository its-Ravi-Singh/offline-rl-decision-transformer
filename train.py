import torch
import matplotlib.pyplot as plt

def train(model, dataloader, epochs=5):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0

        for states, actions in dataloader:

            logits = model(states)
            loss = loss_fn(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

    plt.figure()
    plt.plot(range(epochs), epoch_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()