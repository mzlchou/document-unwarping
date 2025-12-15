from dataset_loader import get_dataloaders, visualize_batch

def main():
    print("Testing dataloader…")
    train_loader, val_loader = get_dataloaders(
        data_dir='renders/synthetic_data_pitch_sweep',
        batch_size=2,
        img_size=(512, 512),
        use_border=False
    )

    batch = next(iter(train_loader))
    visualize_batch(batch)

if __name__ == "__main__":
    main()
