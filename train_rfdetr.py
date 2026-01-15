from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir="data/VNL_500Videos_RFDETR",
    epochs=10,
    batch_size=8,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="./finetune/",
)