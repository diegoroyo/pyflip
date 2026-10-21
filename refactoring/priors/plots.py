import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # the refactoring folder (pyFLIP modules), for the field figures

# Metrics and figures for the runs of testing_priors_plot.py (one metrics.npz per run in pyflip-nw/outputs/<run>/).
# Usage: python priors/plots.py [-o name] [--fields] outputs/noprior outputs/prior [...]
#   -> outputs/<name>_{opd,lyot,injected,real}.png (name: 'fig'); with --fields also <name>_field_{fpm,lyot}.png, the fields
#      recomputed from the model states saved at state_iters (slow: one forward model per run and iteration)
#   python priors/plots.py --table outputs/noprior [...]: only the summary table (one line per run)

def opd_bandpass(opd, pupil, cutoff=18):
    """
    Split an OPD map (n x n, metres) into its low (<= cutoff cycles per pupil) and high frequency parts, inside the pupil
    (bool mask, n x n; the pupil spans the array). Returns low, high (n x n, zero outside the pupil).
    """
    n = opd.shape[-1]
    opd = np.where(pupil, opd, 0.)
    opd = np.where(pupil, opd - opd[pupil].mean(), 0.)
    k = np.fft.fftfreq(n) * n # cycles per array = cycles per pupil
    kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    low = np.fft.ifft2(np.fft.fft2(opd) * (kr <= cutoff)).real * pupil
    high = (opd - low) * pupil
    return low, high

def opd_rms_nm(opd, pupil):
    return 1e9 * np.sqrt(np.mean(opd[pupil] ** 2))

def layer_hf_nm(total, static, pupil, cutoff=18):
    """
    High-frequency rms (nm, inside the pupil) of a fitted OPD layer (total - static maps, metres), band-passed on the full
    array: unlike opd_bandpass this does not mask the map by the pupil first, so a smooth layer (coarse grid, prior) gives
    nothing from the pupil edge. A free grid is zero outside the pupil anyway, so for it both agree.
    """
    layer = (total - static).astype(np.float64)
    low, high = opd_bandpass(layer, np.ones_like(pupil, dtype=bool), cutoff)
    return opd_rms_nm(high, pupil)

def load_runs(run_dirs, snr_max=1e3):
    runs = {}
    for d in run_dirs:
        z = np.load(os.path.join(d, 'metrics.npz'), allow_pickle=True)
        r = {k: z[k] for k in z.files if k not in ('opd_snaps', 'lyot_snaps')} # the snapshots are large and not plotted here
        if 'lyot_snaps' in z.files: # edge-free HF of the fitted Lyot layer, from the snapshots (coarser in time than the logged metric)
            static, pupil = z['lyot_snaps'][0], z['lyot_pupil'].astype(bool)
            r['lyot_layer_iters'] = np.append(z['opd_snap_iters'], z['stage_cutoffs'][-1])
            r['lyot_layer_hf_nm'] = np.array([layer_hf_nm(m, static, pupil) for m in z['lyot_snaps']] + [layer_hf_nm(z['lyot_final'], static, pupil)])
            r['lyot_layer_rms_nm'] = opd_rms_nm((z['lyot_final'] - static).astype(np.float64), pupil)
        for k in ('snrs', 'signal_losses', 'signal_peaks', 'snrs_smooth', 'real_snrs', 'real_snrs_smooth'):
            if k in r:
                r[k] = np.where(np.abs(r[k]) < snr_max, r[k], np.nan) # calc_snr occasionally returns absurd values (e.g. empty annulus)
        runs[os.path.basename(os.path.normpath(d))] = r
    return runs

def fig_opd(runs, out_path, cutoff=18, which='ote'):
    # which = 'ote' (OTE OPD layer: WFS map + prior + grid) or 'lyot' (Lyot-plane layer: NIRCam OPD at the central wavelength + Lyot grid)
    hf, final, pupil_key, label = ('opd_hf_rms_nm', 'opd_final', 'pupil', 'OTE OPD') if which == 'ote' else ('lyot_hf_rms_nm', 'lyot_final', 'lyot_pupil', 'Lyot-plane OPD')
    names = list(runs.keys())
    fig = plt.figure(figsize=[16, 3.6 * (1 + len(names))])
    gs = fig.add_gridspec(1 + len(names), 3, height_ratios=[1.3] + [1] * len(names))

    ax = fig.add_subplot(gs[0, :])
    layer = which == 'lyot' and all('lyot_layer_hf_nm' in r for r in runs.values())
    for name in names:
        r = runs[name]
        if layer: # fitted layer only, edge-free (the logged metric of the total map has a floor from the stop edge for smooth layers)
            line, = ax.plot(r['lyot_layer_iters'], np.maximum(r['lyot_layer_hf_nm'], 1.), '.-', label=f'{name} (final {r["lyot_layer_hf_nm"][-1]:.1f} nm, total map {r[hf][-1]:.0f})')
            ax.plot(r['opd_iters'], r[hf], color=line.get_color(), lw=0.6, alpha=0.4)
        else:
            ax.plot(r['opd_iters'], r[hf], label=f'{name} (final {r[hf][-1]:.2f} nm)')
    for c in runs[names[0]]['stage_cutoffs']:
        ax.axvline(c, color='k', lw=0.5, ls='--')
    ax.set_xlabel('iteration'); ax.set_yscale('log')
    if layer:
        ax.set_ylabel(f'rms of the fitted Lyot layer > {cutoff} c/p [nm]')
        ax.set_title(f'High-frequency content of the fitted Lyot-plane layer, band-passed on the full array (thin: logged metric of the total map masked by the stop; dashed: stage cutoffs)')
    else:
        ax.set_ylabel(f'rms of {label} > {cutoff} c/p [nm]')
        ax.set_title(f'High-frequency content of the {label} (dashed: stage cutoffs)')
    ax.legend(fontsize=8)

    maps = {}
    for name in names:
        r = runs[name]
        pupil = r[pupil_key].astype(bool)
        opd = r[final].astype(np.float64)
        low, high = opd_bandpass(opd, pupil, cutoff)
        full = np.where(pupil, opd - opd[pupil].mean(), np.nan) * 1e9
        maps[name] = [full, np.where(pupil, low, np.nan) * 1e9, np.where(pupil, high, np.nan) * 1e9]
    vmax = [max(np.nanpercentile(np.abs(maps[name][col]), 99.5) for name in names) for col in range(3)] # one color scale per column, shared by all runs
    for row, name in enumerate(names, start=1):
        titles = [f'{name}: final {label}', f'<= {cutoff} c/p', f'> {cutoff} c/p']
        for col, (m, title) in enumerate(zip(maps[name], titles)):
            ax = fig.add_subplot(gs[row, col])
            v = vmax[col]
            im = ax.imshow(m, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
            ax.set_title(f'{title}, rms {np.nanstd(m):.1f} nm'); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, label='nm')
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()

def _residual_panel(ax, r, name, circle_px, color, label):
    res = r['residual_sci'][0] * np.where(r['px_mask'] == 0., np.nan, 1.)
    v = np.nanpercentile(np.abs(res), 99.)
    im = ax.imshow(res, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
    c = res.shape[-1] // 2
    ax.add_patch(plt.Circle((c + circle_px[0], c + circle_px[1]), 4, fill=False, color=color, lw=1))
    ax.set_title(f'{name}: science residual, {label}'); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, label='MJy/sr')

def fig_planet(runs, out_path, which):
    """
    Metrics of one planet over the science stage, for all runs, and the final science residuals with that planet circled.
    which = 'injected' (SNR and signal loss) or 'real' (SNR and recovered peak; the true flux of HIP 65426 b is not known).
    SNR as run_FLIP.py (signal and noise from the 1 px smoothed residual) when logged, else track_SNR's definition.
    """
    names = list(runs.keys())
    ncol = max(len(names), 2)
    fig = plt.figure(figsize=[max(4 * ncol, 14), 9])
    outer = fig.add_gridspec(2, 1, height_ratios=[1.2, 1])
    flux = any('flux_factors' in r and len(r['flux_factors']) for r in runs.values())
    top, bottom = outer[0].subgridspec(1, 3 if flux else 2), outer[1].subgridspec(1, ncol)
    ax1, ax2 = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    if flux:
        ax3 = fig.add_subplot(top[0, 2])
        for name in names:
            r = runs[name]
            if 'flux_factors' in r and len(r['flux_factors']):
                ax3.plot(r['opd_iters'], r['flux_factors'], label=f'{name} (final {r["flux_factors"][-1]:.3f})')
        for c in runs[names[0]]['stage_cutoffs']:
            ax3.axvline(c, color='k', lw=0.5, ls='--')
        ax3.set_xlabel('iteration'); ax3.set_ylabel('flux correction factor'); ax3.set_title('Fitted star flux factor (1 = flux matched in stages 1.1 / 2.1)'); ax3.legend(fontsize=8)
    smooth = all(('snrs_smooth' if which == 'injected' else 'real_snrs_smooth') in r for r in runs.values())
    for name in names:
        r = runs[name]
        if which == 'injected':
            snr = r['snrs_smooth'][:, 0] if smooth else r['snrs'][:, 0]
            second, second_label = r['signal_losses'][:, 0], 'recovered / injected peak'
        else:
            if not len(r.get('real_snrs', [])):
                continue
            snr = r['real_snrs_smooth'] if smooth else r['real_snrs']
            second, second_label = r.get('real_peaks', np.full(len(snr), np.nan)), 'recovered peak [MJy/sr]'
        ax1.plot(r['track_iters'], snr, marker='.', label=f'{name} (final {snr[-1]:.2f})')
        ax2.plot(r['track_iters'], second, marker='.', label=f'{name} (final {second[-1]:.3f})')
    title = 'Injected planet' if which == 'injected' else 'Real planet HIP 65426 b'
    ax1.set_xlabel('iteration'); ax1.set_ylabel('SNR'); ax1.set_title(f'{title}: SNR' + (' (run_FLIP.py definition)' if smooth else '')); ax1.legend(fontsize=8)
    ax2.set_xlabel('iteration'); ax2.set_ylabel(second_label); ax2.set_title(f'{title}: ' + ('signal loss' if which == 'injected' else 'recovered peak')); ax2.legend(fontsize=8)
    for col, name in enumerate(names):
        r = runs[name]
        ax = fig.add_subplot(bottom[0, col])
        if which == 'injected':
            _residual_panel(ax, r, name, r['fakes_px'][0], 'k', f'SNR {(r["snrs_smooth"] if smooth else r["snrs"])[-1, 0]:.2f}')
        else:
            _residual_panel(ax, r, name, r['real_planet_px'], 'g', f'SNR {(r["real_snrs_smooth"] if smooth else r["real_snrs"])[-1]:.1f}')
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()

def model_fields(run_dir, state_file, DEVICE='cuda'):
    """
    Re-run the forward model at a saved state and return the intensity (central wavelength) at the focal plane before the
    mask (oversampled, 2048^2) and at the Lyot plane before the stop (1024^2). The propagation follows
    optical_elements.DiffOpticalSystem.forward / DiffImageOptic.forward, stopped before each transmission.
    """
    import torch
    from model_building import assemble_JWST_NIRCam_coron
    from wavefronts import BroadbandWavefront
    from dl_utils import load_model_data, crop_to
    import torch.nn as nn
    r = np.load(os.path.join(run_dir, 'metrics.npz'), allow_pickle=True)
    st = torch.load(state_file, map_location=DEVICE, weights_only=False)
    data_dir = str(r['data_dir'])
    model = assemble_JWST_NIRCam_coron(data_dir, int(r['num_det_px']))
    _, _, _, _, wlen_weights, _ = load_model_data(data_dir=data_dir)
    wf = BroadbandWavefront(int(r['wf_npix']), float(r['diameter']), wlen_weights, nn.Parameter(torch.FloatTensor([1.])), nn.Parameter(torch.FloatTensor([0., 0.])).to(DEVICE)).to(DEVICE)
    wf.load_state_dict(st['wavefront'])
    # the saved OTE layer already contains the residual grid (and the prior): put it in the primary's OPD, keep its grid at zero
    model.load_state_dict({k: v for k, v in st['model'].items() if not k.startswith(('optical_system.JWSTPrimary.wfe_offsets', 'optical_system.NIRCamLyotstop.wfe_offsets'))}, strict=False) # the Lyot layer does not enter these fields
    primary = model.optical_system.JWSTPrimary
    primary.opd = st['ote_opd'].to(DEVICE)
    primary.wfe_offsets.grid.data.zero_()
    with torch.no_grad():
        phasor = wf.get_phasors()
        phasor = primary(phasor, wf.wavelengths, normalize=True)
        n = wf.npixels
        pad = (n * (model.oversample - 1)) // 2
        fpm_in = torch.fft.fftshift(torch.fft.ifft2(torch.nn.functional.pad(phasor, (pad,) * 4)), dim=[-2, -1]) # focal plane, before the mask
        fpm = model.optical_system['NIRCamcoronfocalplanemask']
        after = fpm_in * fpm.amplitude
        lyot_in = crop_to(torch.fft.fft2(torch.fft.fftshift(after, dim=[-2, -1])), n) # Lyot plane, before the stop
        c = len(wf.wavelengths) // 2
        return (fpm_in[0, c].abs() ** 2).cpu().numpy(), (lyot_in[0, c].abs() ** 2).cpu().numpy()

def fig_fields(runs, run_dirs, out_prefix, fpm_crop=160):
    """
    One row per run, one column per saved state: log10 intensity at the central wavelength, focal plane before the mask
    (central fpm_crop x fpm_crop pixels of the oversampled grid) and Lyot plane before the stop. Two figures.
    """
    names = list(runs.keys())
    iters = runs[names[0]]['state_iters']
    panels = {'fpm': {}, 'lyot': {}}
    for name, d in zip(names, run_dirs):
        for it in iters:
            f = os.path.join(d, 'states', f'iter{int(it)}.pt')
            if not os.path.exists(f):
                continue
            a, b = model_fields(d, f)
            c = a.shape[0] // 2; h = fpm_crop // 2
            panels['fpm'][(name, int(it))] = a[c - h:c + h, c - h:c + h]
            panels['lyot'][(name, int(it))] = b
    for plane, title in (('fpm', 'Focal plane before the mask'), ('lyot', 'Lyot plane before the stop')):
        fig, axes = plt.subplots(len(names), len(iters), figsize=[3.2 * len(iters), 3.2 * len(names)], squeeze=False)
        vmax = max(v.max() for v in panels[plane].values())
        for row, name in enumerate(names):
            for col, it in enumerate(iters):
                ax = axes[row, col]
                m = panels[plane].get((name, int(it)))
                if m is None:
                    ax.axis('off'); continue
                im = ax.imshow(np.log10(np.maximum(m / vmax, 1e-12)), origin='lower', cmap='magma', vmin=-8, vmax=0)
                ax.set_title(f'{name}, iter {int(it)}', fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.01, label='log10 intensity (central wavelength, relative to max over the figure)')
        fig.suptitle(title)
        plt.savefig(f'{out_prefix}_field_{plane}.png', dpi=110); plt.close()

def summary_table(runs):
    # One line per run: real planet SNR (run_FLIP definition) max over the science stage and final, recovered peak at the max,
    # injected planet SNR max / final, loss at the end of the reference stage, final OTE HF, Lyot layer HF (edge-free) and
    # rms, final flux factor, mean |residual| of the reference and science frames
    print(f'{"run":24s} {"realSNR max@it":>15s} {"final":>6s} {"peak":>5s} {"injSNR max/fin":>14s} {"refloss":>8s} {"oteHF":>6s} {"lyotHF":>7s} {"lyotLy":>7s} {"flux":>6s} {"res_ref":>8s} {"res_sci":>8s}')
    for name, r in runs.items():
        it = r['track_iters']
        real = r['real_snrs_smooth'] if len(r.get('real_snrs_smooth', [])) else np.full(len(it), np.nan)
        peaks = r['real_peaks'] if len(r.get('real_peaks', [])) else np.full(len(it), np.nan)
        kmax = int(np.nanargmax(real)) if np.isfinite(real).any() else 0
        inj = r['snrs_smooth'][:, 0]
        refloss = r['losses'][np.searchsorted(r['loss_iters'], r['stage_cutoffs'][2]) - 1]
        lyot_hf = r['lyot_layer_hf_nm'][-1] if 'lyot_layer_hf_nm' in r else r['lyot_hf_rms_nm'][-1]
        lyot_rms = r.get('lyot_layer_rms_nm', np.nan)
        flux = r['flux_factors'][-1] if 'flux_factors' in r else 1.
        mask = r['px_mask'] > 0
        print(f'{name:24s} {real[kmax]:8.1f} @{it[kmax]:5d} {real[-1]:6.1f} {peaks[kmax]:5.1f} {np.nanmax(inj):6.2f} / {inj[-1]:5.2f} {refloss:8.3f} {r["opd_hf_rms_nm"][-1]:6.0f} {lyot_hf:7.0f} {lyot_rms:7.0f} {flux:6.3f} {np.abs(r["residual_ref"][0][mask]).mean():8.3f} {np.abs(r["residual_sci"][0][mask]).mean():8.3f}')

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    name = 'fig'
    if args[0] == '-o':
        name, args = args[1], args[2:]
    fields = False
    if args[0] == '--fields':
        fields, args = True, args[1:]
    if args[0] == '--table': # only the summary table
        summary_table(load_runs(args[1:])); sys.exit()
    runs = load_runs(args)
    out = os.path.dirname(os.path.normpath(args[0]))
    fig_opd(runs, os.path.join(out, f'{name}_opd.png'))
    if all('lyot_final' in r for r in runs.values()):
        fig_opd(runs, os.path.join(out, f'{name}_lyot.png'), which='lyot')
    if fields:
        fig_fields(runs, args, os.path.join(out, name))
    fig_planet(runs, os.path.join(out, f'{name}_injected.png'), 'injected')
    if any(len(r.get('real_snrs', [])) for r in runs.values()):
        fig_planet(runs, os.path.join(out, f'{name}_real.png'), 'real')
    summary_table(runs)
