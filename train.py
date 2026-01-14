from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="data/VNL_500Videos_RTDETR",
    epochs=100,
    batch_size=8,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="./finetune/",
)