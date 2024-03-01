#!/bin/bash

# HPX32 UNet 6var up-interp clrelu 300
#RUN_CMD="python -u scripts/train.py num_workers=4 port=29450 learning_rate=2e-4 batch_size=32 experiment_name=hpx32_unet_6var_up-inter_clrelu_300 model=hpx_unet model.encoder.n_channels=[90,180,360] model.decoder.n_channels=[360,180,90] model/modules/blocks@model.decoder.up_sampling_block=interpolation_upsample model/modules/activations@model.encoder.conv_block.activation=capped_leaky_relu model/modules/activations@model.decoder.conv_block.activation=capped_leaky_relu trainer.max_epochs=300 data=era5_hpx32_6var_6h_24h data.prefix=era5_1deg_3h_HPX32_1979-2021_  data.prebuilt_dataset=False data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam"

# HPX32 UNet 6var 300
#RUN_CMD="python -u scripts/train.py num_workers=4 port=29450 learning_rate=2e-4 batch_size=32 experiment_name=hpx32_unet_6var_300 model=hpx_unet model.encoder.n_channels=[87,174,348] model.decoder.n_channels=[348,174,87] trainer.max_epochs=300 data=era5_hpx32_6var_6h_24h data.prefix=era5_1deg_3h_HPX32_1979-2021_  data.prebuilt_dataset=False data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam"

# HPX32 UNet 6var dil 300
#RUN_CMD="python -u scripts/train.py num_workers=4 port=29451 learning_rate=2e-4 batch_size=32 experiment_name=hpx32_unet_6var_dil_300 model=hpx_unet model.encoder.n_channels=[87,174,348] model.decoder.n_channels=[348,174,87] model.encoder.dilations=[1,2,4] model.decoder.dilations=[4,2,1] trainer.max_epochs=300 data=era5_hpx32_6var_6h_24h data.prefix=era5_1deg_3h_HPX32_1979-2021_  data.prebuilt_dataset=False data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam"

# HPX32 UNet 6var dil cnxt 300
#RUN_CMD="python -u scripts/train.py num_workers=4 port=29453 learning_rate=2e-4 batch_size=32 experiment_name=hpx32_unet_6var_dil_cnxt_300 model=hpx_unet model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[19,38,76] model.decoder.n_channels=[76,38,19] model.encoder.dilations=[1,2,4] model.decoder.dilations=[4,2,1] trainer.max_epochs=300 data=era5_hpx32_6var_6h_24h data.prefix=era5_1deg_3h_HPX32_1979-2021_  data.prebuilt_dataset=False data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam"

# HPX32 UNet 6var dil cnxt inv 300
#RUN_CMD="python -u scripts/train.py num_workers=4 port=29454 learning_rate=2e-4 batch_size=32 experiment_name=hpx32_unet_6var_dil_cnxt_inv_300 model=hpx_unet model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[72,36,18] model.decoder.n_channels=[18,36,72] model.encoder.dilations=[1,2,4] model.decoder.dilations=[4,2,1] trainer.max_epochs=300 data=era5_hpx32_6var_6h_24h data.prefix=era5_1deg_3h_HPX32_1979-2021_  data.prebuilt_dataset=False data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam"

# HPX32 UNet 6var dil cnxt inv gru 300
#RUN_CMD="python -u scripts/train.py num_workers=4 port=29455 learning_rate=2e-4 batch_size=32 experiment_name=hpx32_unet_6var_dil_cnxt_inv_gru_300 model=hpx_rec_unet model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[72,36,18] model.decoder.n_channels=[18,36,72] model.encoder.dilations=[1,2,4] model.decoder.dilations=[4,2,1] trainer.max_epochs=300 data=era5_hpx32_6var_6h_24h data.prefix=era5_1deg_3h_HPX32_1979-2021_  data.prebuilt_dataset=False data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam"

# HPX64 UNet 7var inv dil gru 9.8m cnxt 300
#RUN_CMD="python -u scripts/train.py num_workers=8 port=29455 learning_rate=2e-4 batch_size=16 experiment_name=hpx64_unet_7var_inv_dil_gru_9.8m_cnxt_300 model=hpx_rec_unet model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[136,68,34] model.decoder.n_channels=[34,68,136] model.encoder.dilations=[1,2,4] model.decoder.dilations=[4,2,1] trainer.max_epochs=300 data=era5_hpx64_7var_6h_24h data.prefix=era5_0.25deg_3h_HPX64_1979-2021_  data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam"

# HPX64 UNet 136,68,34 cnxt GRU 6h
#RUN_CMD="python -u scripts/train.py num_workers=8 port=29450 learning_rate=2e-4 batch_size=16 experiment_name=hpx64_unet_136-68-34_cnxt_skip_dil_gru_6h_300 model=hpx_rec_unet model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[136,68,34] model.decoder.n_channels=[34,68,136] trainer.max_epochs=300 data=era5_hpx64_7var_6h_24h data.prefix=era5_0.25deg_3h_HPX64_1979-2021_  data.prebuilt_dataset=False data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam"

# HPX64 UNet 136,68,34 cnxt GRU 3h
RUN_CMD="python -u scripts/train.py num_workers=8 port=29450 learning_rate=2e-4 batch_size=16 experiment_name=hpx64_unet_136-68-34_cnxt_skip_dil_gru_3h_300 model=hpx_rec_unet model/modules/blocks@model.encoder.conv_block=conv_next_block model/modules/blocks@model.decoder.conv_block=conv_next_block model.encoder.n_channels=[136,68,34] model.decoder.n_channels=[34,68,136] trainer.max_epochs=300 data=era5_hpx64_7var_3h_24h data.prefix=era5_0.25deg_3h_HPX64_1979-2021_  data.prebuilt_dataset=False data.module.drop_last=True trainer/lr_scheduler=cosine trainer/optimizer=adam"


# Specify number of GPUs and CPUs
NUM_GPU=2
TOTALGPU=$(( ${NUM_GPU} ))
TOTALCORES=16  # Number of CPUs

# srun command for CPU
#srun -u --ntasks=1 \
#     --ntasks-per-node=1 \
#     --cpu_bind=sockets \
#     -c $(( ${TOTALCORES} )) \
#     bash -c "
#     export WORLD_RANK=\${SLURM_PROCID}
#     export HDF5_USE_FILE_LOCKING=True
#     export CUDA_VISIBLE_DEVICES=
#     export HYDRA_FULL_ERROR=1 
#     ${RUN_CMD}"
#exit

# srun command for GPU
srun -u --ntasks=${TOTALGPU} \
     --ntasks-per-node=${NUM_GPU} \
     --gres=gpu:A100:${NUM_GPU} \
     --cpu_bind=sockets \
     -c $(( ${TOTALCORES} / ${NUM_GPU} )) \
     bash -c "
     export WORLD_RANK=\${SLURM_PROCID}
     export HDF5_USE_FILE_LOCKING=False
     export CUDA_VISIBLE_DEVICES=0,1
     export HYDRA_FULL_ERROR=1 
     ${RUN_CMD}"
