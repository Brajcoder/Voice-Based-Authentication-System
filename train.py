"""
Root training runner forwarding to ml.train.
Allows running either `python train.py` or `python -m ml.train`.
"""

from ml.train import main, DATASET_PATH, max_speakers, samples_per_speaker, EPOCHS, LR, BATCH_SIZE

if __name__ == "__main__":
    main()

