# Fit with the real mask raster at 593.336 and 637.045 um/arcsec and a fitted FPM axial offset.
# The mount is dropped unless MOUNT=1. Usage: FPM_RASTER=/path/to/raster.npz bash experiments/run_real_mask.sh
cd "$(dirname "$0")/.."
: "${FPM_RASTER:?set FPM_RASTER to the mask raster .npz}"
MOUNT_FLAG=--no_mount
if [ "${MOUNT:-0}" = 1 ]; then MOUNT_FLAG=; fi
for UM in 593.336 637.045; do
    CUDA_VISIBLE_DEVICES=0 python run_FLIP_displacement.py \
        --data_dir HIP65426/model_setup  \
        --measurement_file HIP65426/demo_data/HIP-65426/full_im/target_002_00001COADDALL.npy  \
        --reference_file HIP65426/demo_data/HIP-68245/full_im/reference_001_00001COADDALL.npy \
        --opd_filename observation_opd.npy \
        --scene_name DEMO \
        --stage_fit_flat_fields_start_iter 3500 \
        --stage_fit_flat_fields_end_iter 5500 \
        --px_mask_file HIP65426/model_setup/pixel_masks/pixel_masks.npy \
        --ref_which_int 0 --sci_which_int 0 \
        --exp_name "HIP65426_real${UM%.*}" \
        --iters 15000 --sci_targ_name HIP65426 \
        --ref_cutoff_iter 5500 --sci_lr_weight 0.005  \
        --forced_stop_iter 11000 \
        --inj_xpos 17 --inj_ypos 5 --inject_how_many_sigma_flux 30 --track_injected_planet \
        --measurement_noisemap HIP65426/demo_data/HIP-65426/noise_maps/noisemap_target_002_00001COADDALL.npy \
        --lr 0.000000004 --stage_fluxpos_cutoff_iter 600 \
        --element fpm --offset 0 \
        --fpm_raster "$FPM_RASTER" --um_per_arcsec $UM $MOUNT_FLAG
done
