from pretraining import Autoencoder

test = Autoencoder.load_from_checkpoint("VisionRetrievalPretraining/srip_pt_extended/checkpoints/epoch=18-step=85576.ckpt")
test.encoder.save_pretrained("VisionRetrievalPretraining/logical_pt_extended/srip_best")